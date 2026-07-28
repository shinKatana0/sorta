"""Can the VLM verdict be learned from CLIP features? A measurement, not a feature.

The live run of 2026-07-28 left a by-product nobody had before: 7 895 frames labelled by
the model itself (`media_class` rows with `source='vlm'`). That is a training set obtained
for free — the expensive part has already been paid for.

The question this script answers with numbers: does a light classifier over CLIP features
reproduce what the deep tier decided? CLIP is computed on those frames by the fast tier
anyway, so the features cost nothing; a logistic regression trains in seconds and answers
in milliseconds.

The motive came out of F106. The gate threshold does not tune (the curve is flat, outcome
C) — not because 0.4 is perfect, but because **one number is a poor decision rule**. The
gate asks "how much does this frame look like a product, by a single CLIP score" and on
that one feature sends a third of the collection into an expensive model. A trained rule
looks at the whole feature vector.

Two possible uses, and the measurement has to tell them apart:

1. **Replacing the tier** where the probe is confident: the frame never goes to the VLM.
2. **A smart gate**: only the frames the probe is unsure about go to the VLM. That is the
   population cut the threshold never gave, but by a rule that means something.

Everything needed is on disk — the script makes no VLM call at all:

    label (the truth)     media_class rows with source='vlm' — what the model answered
    fast verdict          a snapshot of media_class taken BEFORE the deep run (--before)
    features              recomputed with the fast CLIP — minutes, not hours

Three things this script is careful about:

* The features are NOT a private copy of the prompts. `junk._CLIP_CLASSES`,
  `junk._DOCUMENT_CLASSES` and `junk._PRODUCT_CLASSES` are imported and run through the
  pipeline's own classifier: a copy would measure this script against itself instead of
  against the tier that actually runs. The feature vector is the three probability groups
  concatenated (see FEATURE_GROUPS) — nothing derived, nothing hand-picked. The image
  embedding is deliberately NOT added: the brief allows it if the probabilities turn out
  to be too few, but that is a second variant to be reported next to the first, not a
  quiet upgrade of a measurement that has already been read.
* The split is honest: stratified by label, fixed seed, the probe trains on the training
  part only and every number below is computed on the held-out part alone. Otherwise the
  report would measure the classifier's memory rather than its usefulness. There is no
  hyper-parameter search either — picking `C` by the held-out part is exactly how a
  measurement starts to lie.
* "Do nothing" is on the table. Without the row that says how many verdicts would have
  matched had the fast verdict simply been kept, any accuracy looks impressive.

`scikit-learn` is NOT a new dependency: it arrives with `hdbscan` (declared in
pyproject.toml, used by the face clustering), which is why an import of it here adds
nothing to the install. No new dependency is introduced by this script.

Privacy: counts only. No path, no basename, no file id is printed or stored — a table
about documents must not become a list of where somebody's papers are (the rule of
measure_ocr_gate.py / measure_vlm_speed.py / measure_candidate_gate.py before it).

The database is opened `mode=ro`: a measurement writes nothing.

Usage (from the repo root, with a GPU venv — `uv sync --extra gpu --extra vlm`):
    python scripts/measure_clip_probe.py --before report_output/verdicts_before_vlm.json
    python scripts/measure_clip_probe.py --before before.json --test-size 0.4 --seed 7
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sqlite3
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sorta import junk  # noqa: E402
from sorta.config import Config, load_config  # noqa: E402
from sorta.landmarks import batched, clip_classifier  # noqa: E402
from sorta.naming import naming_settings  # noqa: E402

# The features: the prompt groups of the fast tier, imported, in a fixed order. All three
# are already computed for a candidate frame during a run, so the probe would cost nothing
# extra in production either.
FEATURE_GROUPS: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = (
    ("base", junk._CLIP_CLASSES),          # photo / screenshot / meme
    ("document", junk._DOCUMENT_CLASSES),  # the document group with its anti-classes
    ("product", junk._PRODUCT_CLASSES),    # the product group with its anti-classes
)
N_FEATURES = sum(len(classes) for _name, classes in FEATURE_GROUPS)

DEFAULT_TEST_SIZE = 0.3
DEFAULT_SEED = 20260728

# The gate curve: send the N% least confident frames to the VLM. The brief asks for
# 10/20/30/50; 100% is kept as a control row — it must preserve every change, and a report
# that shows it is a report that checks itself.
GATE_GRID = (0.10, 0.20, 0.30, 0.50, 1.00)

# --- Pre-registered acceptance criteria (F107) -------------------------------
#
# Written down before the first run, so that the table cannot talk anybody into a
# conclusion afterwards.
#
# A — the probe replaces the tier on the confident part: agreement on the held-out part
#     >= 95% AND the share of `document -> non-document` <= 2% of the documents in it.
#     A document that became a photo is somebody's papers in a city folder, not "slightly
#     worse filing", which is why it has a criterion of its own.
# B — a smart gate only: A is not met, but there is an N <= 30% at which sending the N%
#     least confident frames to the VLM preserves >= 98% of the changed verdicts. That is
#     already two to three times cheaper than today's 32.6% of the population.
# C — neither: CLIP does not express what the VLM sees. A normal outcome, and a useful
#     one: it would mean the model's advantage is exactly what the CLIP features lack, and
#     the tier stays as it is.
MIN_AGREEMENT = 0.95
MAX_DOCUMENT_LEAK = 0.02
MIN_CHANGES_KEPT = 0.98
MAX_SMART_GATE_SHARE = 0.30

# Share of the collection the candidate gate opened for on the run of 2026-07-28
# (7 896 of 24 196) — what outcome B would be compared against.
CURRENT_CANDIDATE_SHARE = 0.326

DOCUMENT_VERDICT = "document"


@dataclass(frozen=True)
class Sample:
    """One labelled frame — its features and its verdicts, and nothing that identifies it.

    `label` is what the VLM answered (the truth this probe is trained on), `before` is the
    fast verdict from the snapshot, None — the frame is not in it (indexed after the
    snapshot was taken), so whether the tier changed anything for it is unknown and is
    counted apart instead of being guessed.
    """
    features: tuple[float, ...]
    label: str
    before: str | None


@dataclass(frozen=True)
class Answer:
    """One held-out frame after the probe answered it. Counts only, no identity.

    `confidence` is the probe's probability of its own answer — the ordering the smart
    gate would use: least confident first.
    """
    label: str
    predicted: str
    confidence: float
    before: str | None

    @property
    def correct(self) -> bool:
        return self.predicted == self.label

    @property
    def changed(self) -> bool:
        """The deep tier was useful here: it moved the verdict off the fast one."""
        return self.before is not None and self.before != self.label


@dataclass(frozen=True)
class ProbeRun:
    """What the honest split produced: the size it trained on, and the held-out answers."""
    trained_on: int
    answers: list[Answer]


@dataclass(frozen=True)
class Evaluation:
    """The confusion matrix on the held-out part: (VLM label, probe answer) -> frames.

    Everything the report says about agreement is arithmetic over these counts, so the
    matrix and the percentages can never disagree with each other.
    """
    pairs: Counter[tuple[str, str]]

    @property
    def total(self) -> int:
        return sum(self.pairs.values())

    @property
    def agreed(self) -> int:
        return sum(n for (label, pred), n in self.pairs.items() if label == pred)

    @property
    def agreement(self) -> float:
        """Share of the held-out frames the probe answered as the VLM did (1.0 if empty)."""
        return self.agreed / self.total if self.total else 1.0

    @property
    def labels(self) -> list[str]:
        seen = {label for label, _pred in self.pairs} | {pred for _label, pred in self.pairs}
        return sorted(seen)

    def per_class(self) -> dict[str, tuple[int, int]]:
        """label -> (answered as the VLM did, frames with that label)."""
        out: dict[str, tuple[int, int]] = {}
        for (label, pred), n in self.pairs.items():
            agreed, total = out.get(label, (0, 0))
            out[label] = (agreed + (n if label == pred else 0), total + n)
        return dict(sorted(out.items()))

    def document_leak(self) -> tuple[int, int]:
        """(documents the probe called something else, documents in the held-out part).

        The direction that matters. `photo -> document` is a milder mistake: it leaves a
        photo out of the city folders. `document -> photo` puts papers into them.
        """
        lost = sum(n for (label, pred), n in self.pairs.items()
                   if label == DOCUMENT_VERDICT and pred != DOCUMENT_VERDICT)
        total = sum(n for (label, _pred), n in self.pairs.items() if label == DOCUMENT_VERDICT)
        return lost, total


@dataclass(frozen=True)
class GateRow:
    """One row of the gate curve: what sending the N% least confident frames preserves."""
    share: float          # N, as a fraction of the held-out part
    to_vlm: int           # frames that would still be sent to the model
    kept: int             # changed verdicts preserved
    changed_total: int    # changed verdicts in the held-out part

    @property
    def lost(self) -> int:
        return self.changed_total - self.kept

    @property
    def kept_frac(self) -> float:
        """Share of the changed verdicts preserved (1.0 — an empty benefit loses nothing)."""
        return self.kept / self.changed_total if self.changed_total else 1.0


def load_before(path: Path) -> dict[int, str]:
    """The pre-tier snapshot -> {file_id: verdict}.

    The snapshot is produced outside this repo (the orchestrator dumps media_class before a
    deep run), so the shapes such a dump comes in are all accepted: a flat mapping of id to
    verdict, a list of rows (dicts with file_id/verdict, or [file_id, verdict] pairs), or
    either of those under a "rows" key. Anything else is an error rather than a silently
    empty baseline — without the snapshot the "do nothing" row is exactly the row that
    keeps the rest of the report honest.

    (measure_candidate_gate.py parses the same file; the loader is repeated instead of
    imported because scripts/ is not a package and a measurement should not import another
    measurement to stay runnable on its own.)
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "rows" in data:
        data = data["rows"]
    out: dict[int, str] = {}
    if isinstance(data, dict):
        for key, value in data.items():
            verdict = value.get("verdict") if isinstance(value, dict) else value
            out[int(key)] = str(verdict)
    elif isinstance(data, list):
        for row in data:
            if isinstance(row, dict):
                out[int(row["file_id"])] = str(row["verdict"])
            else:
                out[int(row[0])] = str(row[1])
    else:
        raise SystemExit(f"{path}: не похоже на снимок media_class "
                         f"(ожидается список строк или отображение id -> вердикт)")
    if not out:
        raise SystemExit(f"{path}: снимок пуст — сравнивать не с чем")
    return out


def stratified_split(labels: list[str], test_size: float,
                     seed: int) -> tuple[list[int], list[int]]:
    """(train indices, held-out indices) — stratified by label, reproducible from the seed.

    Every class is split in the same proportion, so a rare class (`document` is a few
    percent of the collection) cannot land entirely on one side and turn its column of the
    matrix into noise. A class of one frame stays in training: it is not enough to learn
    from, but a class the probe has never seen is a column of guaranteed zeros, which
    reports as a failure of CLIP rather than of the sample size.

    The two parts are disjoint and together are the whole sample — that is what makes
    "measured on the held-out part" mean anything.
    """
    if not 0.0 < test_size < 1.0:
        raise SystemExit(f"--test-size: ожидается доля в (0, 1), получено {test_size}")
    by_label: dict[str, list[int]] = defaultdict(list)
    for i, label in enumerate(labels):
        by_label[label].append(i)
    rng = random.Random(seed)
    train: list[int] = []
    test: list[int] = []
    for label in sorted(by_label):
        indices = list(by_label[label])
        rng.shuffle(indices)
        n_test = min(int(round(len(indices) * test_size)), len(indices) - 1)
        test.extend(indices[:n_test])
        train.extend(indices[n_test:])
    return sorted(train), sorted(test)


def run_probe(samples: list[Sample], test_size: float, seed: int) -> ProbeRun:
    """Train on the training part, answer the held-out part, return the answers.

    A plain logistic regression: simple, explainable, and — crucially — not tuned. The only
    knobs touched are the ones that make it converge at all (`max_iter`), never the ones
    that would be chosen by looking at the score it is about to be judged by.
    """
    from sklearn.linear_model import LogisticRegression

    if not samples:
        raise SystemExit("нет размеченных кадров (media_class.source='vlm') — нечего учить")
    labels = [s.label for s in samples]
    train_idx, test_idx = stratified_split(labels, test_size, seed)
    if len({labels[i] for i in train_idx}) < 2:
        raise SystemExit("в обучающей части меньше двух классов — обучать нечему")
    features = np.asarray([s.features for s in samples], dtype=np.float64)

    model = LogisticRegression(max_iter=1000, random_state=seed)
    model.fit(features[train_idx], [labels[i] for i in train_idx])

    if not test_idx:
        return ProbeRun(trained_on=len(train_idx), answers=[])
    probs = model.predict_proba(features[test_idx])
    predicted = [str(c) for c in model.classes_[np.argmax(probs, axis=1)]]
    confidence = np.max(probs, axis=1)
    answers = [
        Answer(label=samples[i].label, predicted=predicted[k],
               confidence=float(confidence[k]), before=samples[i].before)
        for k, i in enumerate(test_idx)
    ]
    return ProbeRun(trained_on=len(train_idx), answers=answers)


def confusion(answers: list[Answer]) -> Evaluation:
    """The held-out answers -> counts per (label, prediction) pair."""
    return Evaluation(Counter((a.label, a.predicted) for a in answers))


def gate_curve(answers: list[Answer], grid: tuple[float, ...] = GATE_GRID) -> list[GateRow]:
    """The smart gate: send the N% least confident frames to the VLM, keep the rest.

    A changed verdict is preserved either because the frame was sent to the model (which
    then answers what it answered) or because the probe happened to answer the same thing.
    The frames are ordered by confidence once and the gate takes a prefix of that order, so
    the curve is monotone in N by construction: a larger N sends a superset.
    """
    order = sorted(range(len(answers)), key=lambda i: (answers[i].confidence, i))
    changed_total = sum(1 for a in answers if a.changed)
    rows: list[GateRow] = []
    for share in grid:
        to_vlm = min(len(answers), math.ceil(share * len(answers)))
        gated = set(order[:to_vlm])
        kept = sum(1 for i, a in enumerate(answers)
                   if a.changed and (i in gated or a.correct))
        rows.append(GateRow(share=share, to_vlm=to_vlm, kept=kept,
                            changed_total=changed_total))
    return rows


def do_nothing(answers: list[Answer]) -> tuple[int, int, int]:
    """The baseline: (fast verdicts that already match the VLM, frames compared, unknown).

    Computed from the snapshot — the fast verdict against the VLM label — and never from
    the probe's answers. Frames missing from the snapshot have no fast verdict to keep, so
    they are counted apart rather than being counted as a match.
    """
    known = [a for a in answers if a.before is not None]
    matched = sum(1 for a in known if a.before == a.label)
    return matched, len(known), len(answers) - len(known)


def decide(evaluation: Evaluation, curve: list[GateRow]) -> tuple[str, str]:
    """(the outcome letter, one line saying why) — by the pre-registered criteria above."""
    leaked, documents = evaluation.document_leak()
    leak = leaked / documents if documents else 0.0
    agreement = evaluation.agreement
    if agreement >= MIN_AGREEMENT and leak <= MAX_DOCUMENT_LEAK:
        return "A", (
            f"согласие с VLM {agreement:.1%} при {MIN_AGREEMENT:.0%} и утечка документов "
            f"{leak:.1%} при {MAX_DOCUMENT_LEAK:.0%} ({leaked} из {documents}): зонд по "
            f"CLIP-признакам воспроизводит вердикт модели — уверенную часть можно не "
            f"возить в VLM (отдельной фичей, после того как таблицу прочитает человек)")
    good = [r for r in curve
            if r.share <= MAX_SMART_GATE_SHARE and r.kept_frac >= MIN_CHANGES_KEPT]
    if good:
        pick = min(good, key=lambda r: r.share)
        return "B", (
            f"условия A не выполнены (согласие {agreement:.1%} при {MIN_AGREEMENT:.0%}, "
            f"утечка документов {leak:.1%} при {MAX_DOCUMENT_LEAK:.0%}), но гейт по "
            f"неуверенности работает: при N={pick.share:.0%} сохраняется "
            f"{pick.kept_frac:.1%} изменённых вердиктов при {MIN_CHANGES_KEPT:.0%} — "
            f"это {pick.share:.0%} популяции вместо нынешних "
            f"{CURRENT_CANDIDATE_SHARE:.1%}, умный гейт вместо порога")
    best = max((r for r in curve if r.share <= MAX_SMART_GATE_SHARE),
               key=lambda r: r.kept_frac, default=None)
    tail = (f"лучший гейт до {MAX_SMART_GATE_SHARE:.0%} — N={best.share:.0%} с "
            f"{best.kept_frac:.1%} изменений при {MIN_CHANGES_KEPT:.0%}"
            if best else f"в сетке нет N <= {MAX_SMART_GATE_SHARE:.0%}")
    return "C", (
        f"ни A, ни B: согласие {agreement:.1%} при {MIN_AGREEMENT:.0%}, утечка документов "
        f"{leak:.1%} при {MAX_DOCUMENT_LEAK:.0%} ({leaked} из {documents}), {tail}. "
        f"CLIP не выражает того, что видит VLM — преимущество модели именно в том, чего в "
        f"этих признаках нет; ярус остаётся как есть, тему закрываем с цифрами")


def _pct(part: int, whole: int) -> str:
    return f"{100.0 * part / whole:.1f}%" if whole else "—"


def format_header(run: ProbeRun, total: int, test_size: float, seed: int) -> str:
    return "\n".join([
        "=" * 92,
        f"ЗОНД ПО CLIP-ПРИЗНАКАМ: выучивается ли вердикт VLM ({total} размеченных кадров, "
        f"{N_FEATURES} признаков)",
        f"обучение: {run.trained_on} кадров, отложено: {len(run.answers)} "
        f"(доля {test_size:g}, seed {seed}); все метрики ниже — только по отложенной части",
        "=" * 92,
    ])


def format_agreement(evaluation: Evaluation) -> str:
    """Agreement with the VLM: overall and per class. Counts only."""
    lines = [f"СОГЛАСИЕ С VLM: {evaluation.agreement:.1%} "
             f"({evaluation.agreed} из {evaluation.total})"]
    width = max((len(label) for label in evaluation.per_class()), default=0)
    for label, (agreed, total) in evaluation.per_class().items():
        lines.append(f"  {label:<{width}}  {agreed:>6d} из {total:>6d} "
                     f"({_pct(agreed, total):>6})")
    return "\n".join(lines)


def format_confusion(evaluation: Evaluation) -> str:
    """The mistakes themselves, per label pair — counters, no paths and no ids."""
    lines = ["МАТРИЦА ОШИБОК (метка VLM -> ответ зонда, счётчиками):"]
    per_label: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for (label, pred), n in evaluation.pairs.items():
        per_label[label].append((pred, n))
    for label in sorted(per_label):
        cells = ", ".join(f"{pred}: {n}" for pred, n in
                          sorted(per_label[label], key=lambda kv: (-kv[1], kv[0])))
        lines.append(f"  {label} -> {cells}")
    if len(lines) == 1:
        lines.append("  (отложенная часть пуста)")
    return "\n".join(lines)


def format_documents(evaluation: Evaluation) -> str:
    """`document -> anything`, on its own line whatever the outcome.

    An average over thousands of frames hides the dozens that are personal papers, and
    those are the ones that end up laid out by city.
    """
    leaked, total = evaluation.document_leak()
    leak = leaked / total if total else 0.0
    lines = [f"ДОКУМЕНТЫ, СТАВШИЕ НЕ ДОКУМЕНТАМИ: {leaked} из {total} "
             f"({_pct(leaked, total)}) при пороге {MAX_DOCUMENT_LEAK:.0%} — "
             f"{'в пределах' if leak <= MAX_DOCUMENT_LEAK else 'ВЫШЕ'} критерия"]
    for (label, pred), n in sorted(evaluation.pairs.items(), key=lambda kv: (-kv[1], kv[0])):
        if label == DOCUMENT_VERDICT and pred != DOCUMENT_VERDICT:
            lines.append(f"  document -> {pred}: {n}")
    if len(lines) == 1:
        lines.append("  (ни один документ не потерян)")
    return "\n".join(lines)


def format_gate_curve(rows: list[GateRow]) -> str:
    """The curve: N% least confident frames to the VLM -> changed verdicts preserved."""
    lines = ["КРИВАЯ ГЕЙТА (в VLM едут N% наименее уверенных кадров отложенной части):",
             f"{'N':>6} {'в VLM':>9} {'сохранено изменений':>28}"]
    for r in rows:
        lines.append(f"{r.share:>5.0%} {r.to_vlm:>9d} "
                     f"{r.kept:>15d} из {r.changed_total:<6d} ({r.kept_frac:>6.1%})")
    lines.append(f"порог исхода B: {MIN_CHANGES_KEPT:.0%} изменений при "
                 f"N <= {MAX_SMART_GATE_SHARE:.0%} (сейчас в VLM едет "
                 f"{CURRENT_CANDIDATE_SHARE:.1%} коллекции)")
    return "\n".join(lines)


def format_baseline(answers: list[Answer]) -> str:
    """"Do nothing": keep the fast verdict. Without this row any accuracy looks impressive."""
    matched, known, unknown = do_nothing(answers)
    changed = sum(1 for a in answers if a.changed)
    lines = [
        "НИЧЕГО НЕ ДЕЛАТЬ (оставить вердикт быстрого яруса из снимка «до»):",
        f"  совпало бы с VLM: {matched} из {known} ({_pct(matched, known)}); "
        f"изменённых вердиктов: {changed}",
    ]
    if unknown:
        lines.append(f"  нет в снимке «до»: {unknown} кадр(ов) — в базу не считаются "
                     f"(проиндексированы после снимка)")
    return "\n".join(lines)


def format_outcome(evaluation: Evaluation, curve: list[GateRow]) -> str:
    letter, why = decide(evaluation, curve)
    return f"ИСХОД {letter}: {why}"


def labelled_rows(db_path: str) -> list[sqlite3.Row]:
    """The frames the VLM answered, read-only: id, path and the verdict it gave.

    `mode=ro` is the contract of the brief — a measurement writes nothing. `source='vlm'`
    is the whole training set: a frame the fast tier decided was never shown to the model,
    so its verdict is not a label of what the model would have said.
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute(
            """SELECT f.id, f.path, mc.verdict AS verdict
               FROM files f JOIN media_class mc ON mc.file_id = f.id
               WHERE mc.source = 'vlm' AND f.dup_of IS NULL AND f.error IS NULL
               ORDER BY f.id"""
        ).fetchall()
    finally:
        conn.close()


def collect_samples(cfg: Config, rows: list[sqlite3.Row],
                    before: dict[int, str]) -> list[Sample]:  # pragma: no cover — ML
    """CLIP over the labelled frames -> the feature vectors of the probe.

    All three prompt groups are run with the pipeline's own classifier, in the order of
    FEATURE_GROUPS, and concatenated. The classifier caches image features by path, so the
    three calls decode and encode each frame once.
    """
    s = naming_settings(cfg)
    classifier = clip_classifier(s)
    groups = [[prompt for _cls, prompt in classes] for _name, classes in FEATURE_GROUPS]

    samples: list[Sample] = []
    done = 0
    for chunk in batched(rows, s.clip_batch_size):
        paths = [r["path"] for r in chunk]
        probs = [classifier(paths, prompts) for prompts in groups]
        for k, r in enumerate(chunk):
            features = np.concatenate([group[k] for group in probs])
            samples.append(Sample(
                features=tuple(float(x) for x in features),
                label=str(r["verdict"]),
                before=before.get(int(r["id"])),
            ))
        done += len(chunk)
        print(f"  CLIP {done}/{len(rows)}", end="\r", flush=True)
    print(" " * 40, end="\r")
    return samples


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--before", required=True,
                    help="JSON snapshot of media_class taken BEFORE the deep run")
    ap.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE,
                    help=f"share held out for the metrics (default {DEFAULT_TEST_SIZE:g})")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()

    cfg = load_config(args.config)
    before = load_before(Path(args.before))
    rows = labelled_rows(str(cfg.database))
    if not rows:
        raise SystemExit("в индексе нет ответов модели (media_class.source='vlm') — "
                         "учить зонд не на чем")
    print(f"размечено моделью: {len(rows)} кадров, снимок «до»: {len(before)} строк")

    samples = collect_samples(cfg, rows, before)
    run = run_probe(samples, args.test_size, args.seed)
    evaluation = confusion(run.answers)
    curve = gate_curve(run.answers)

    print()
    print(format_header(run, len(samples), args.test_size, args.seed))
    print(format_agreement(evaluation))
    print("-" * 92)
    print(format_confusion(evaluation))
    print("-" * 92)
    print(format_documents(evaluation))
    print("-" * 92)
    print(format_gate_curve(curve))
    print("-" * 92)
    print(format_baseline(run.answers))
    print("=" * 92)
    print(format_outcome(evaluation, curve))
    return 0


if __name__ == "__main__":
    sys.exit(main())
