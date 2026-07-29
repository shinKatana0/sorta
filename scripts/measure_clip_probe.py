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

F109 — two feature sets, both reported. The first run (F107) trained the probe on the
17 prompt probabilities and came out at 83.7% agreement, outcome C. That answer is honest
but narrow: seventeen numbers are a very compressed retelling of a picture, and they answer
questions posed in advance ("how much does this look like a document"), while the image
embedding carries everything the model sees at all. So the same probe is now run on both,
under the same split, the same seed and the same criteria:

    probs   the three prompt groups concatenated — 17 numbers, what F107 measured
    embed   the normalized image embedding of the same CLIP — ~768 numbers

They are never mixed into one vector: a probe over probabilities-plus-embedding would
answer neither question. And both rows are always printed side by side — a measurement that
shows only the winning variant has stopped being a measurement.

Everything needed is on disk — the script makes no VLM call at all:

    label (the truth)     media_class rows with source='vlm' — what the model answered
    fast verdict          a snapshot of media_class taken BEFORE the deep run (--before)
    features              recomputed with the fast CLIP — minutes, not hours

Three things this script is careful about:

* The features are NOT a private copy of the prompts. `junk._CLIP_CLASSES`,
  `junk._DOCUMENT_CLASSES` and `junk._PRODUCT_CLASSES` are imported and run through the
  pipeline's own classifier: a copy would measure this script against itself instead of
  against the tier that actually runs. The embedding comes from the very same object
  (`CachingFeatureClassifier.encode`, already L2-normalized), in the same pass over the
  same frames — one CLIP pass produces both variants, so they differ in the features and
  in nothing else.
* The split is honest: stratified by label, fixed seed, the probe trains on the training
  part only and every number below is computed on the held-out part alone. Otherwise the
  report would measure the classifier's memory rather than its usefulness. There is no
  hyper-parameter search either — picking `C` by the held-out part is exactly how a
  measurement starts to lie, and with ~768 features over ~5 500 training frames it is also
  exactly where over-fitting would be cured by peeking. `C` is fixed in the code (PROBE_C)
  and is the same for both variants.
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
    python scripts/measure_clip_probe.py --before before.json --features probs
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
from sorta.landmarks import (  # noqa: E402
    CachingFeatureClassifier,
    batched,
    clip_classifier,
)
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

# The two feature sets, and never their concatenation (F109). `both` measures two probes
# over the same frames and the same split, and prints both — see selected_variants().
PROBS = "probs"
EMBED = "embed"
BOTH = "both"
FEATURE_CHOICES = (PROBS, EMBED, BOTH)
DEFAULT_FEATURES = BOTH
VARIANT_TITLES = {
    PROBS: "вероятности по промптам",
    EMBED: "эмбеддинг изображения",
}

DEFAULT_TEST_SIZE = 0.3
DEFAULT_SEED = 20260728

# Regularization: fixed here, identical for both variants, and deliberately NOT a command
# line option. The embedding brings ~768 dimensions to ~5 500 training frames, where
# over-fitting is real — and the cure for it is regularization, not a search for the `C`
# that scores best on the part the report is about to be judged by. 1.0 is scikit-learn's
# own default, so the `probs` numbers stay exactly the ones F107 published.
PROBE_C = 1.0
PROBE_MAX_ITER = 1000

# The gate curve: send the N% least confident frames to the VLM. The brief asks for
# 10/20/30/50; 100% is kept as a control row — it must preserve every change, and a report
# that shows it is a report that checks itself.
GATE_GRID = (0.10, 0.20, 0.30, 0.50, 1.00)

# --- Pre-registered acceptance criteria (F107, unchanged by F109) -------------
#
# Written down before the first run, so that the table cannot talk anybody into a
# conclusion afterwards. F109 adds a second feature set and not a single new degree of
# freedom: the same thresholds, the same split, the same seed — otherwise the two rows
# would be two different experiments rather than two sets of features.
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
#     the tier stays as it is. Reached for BOTH feature sets, it closes the topic.
MIN_AGREEMENT = 0.95
MAX_DOCUMENT_LEAK = 0.02
MIN_CHANGES_KEPT = 0.98
MAX_SMART_GATE_SHARE = 0.30

# Share of the collection the candidate gate opened for on the run of 2026-07-28
# (7 896 of 24 196) — what outcome B would be compared against.
CURRENT_CANDIDATE_SHARE = 0.326

DOCUMENT_VERDICT = "document"

# A > B > C: which letter a joint outcome over the variants reports (see overall_decision).
_OUTCOME_ORDER = {"A": 0, "B": 1, "C": 2}


@dataclass(frozen=True)
class Sample:
    """One labelled frame — its features and its verdicts, and nothing that identifies it.

    Both feature sets of the frame are carried at once because both come out of the same
    CLIP pass, and because the comparison is only meaningful if the two probes see the very
    same frames on the very same side of the split.

    `label` is what the VLM answered (the truth this probe is trained on), `before` is the
    fast verdict from the snapshot, None — the frame is not in it (indexed after the
    snapshot was taken), so whether the tier changed anything for it is unknown and is
    counted apart instead of being guessed.
    """
    probs: tuple[float, ...]    # the prompt-group probabilities, N_FEATURES of them
    embed: tuple[float, ...]    # the normalized image embedding of the same CLIP
    label: str
    before: str | None

    def features(self, variant: str) -> tuple[float, ...]:
        """The feature vector of one variant. The two are never concatenated."""
        if variant == PROBS:
            return self.probs
        if variant == EMBED:
            return self.embed
        raise SystemExit(f"неизвестный набор признаков: {variant}")


@dataclass(frozen=True)
class FrameFeatures:
    """What one CLIP pass produced for one frame; `embed` is None — it did not decode.

    Kept apart from `Sample` because the embedding width is only known once some frame has
    actually been encoded: an undecodable frame gets a row of zeros of that width, the same
    row `CachingFeatureClassifier` hands the pipeline for it.
    """
    probs: tuple[float, ...]
    embed: tuple[float, ...] | None


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


@dataclass(frozen=True)
class VariantReport:
    """One feature set, measured end to end — and its own outcome letter.

    `dim` is printed everywhere the variant is: the reader has to see that 17 numbers are
    being compared with ~768 rather than have to guess it.
    """
    variant: str
    dim: int
    run: ProbeRun
    evaluation: Evaluation
    curve: list[GateRow]

    @property
    def decision(self) -> tuple[str, str]:
        return decide(self.evaluation, self.curve)

    @property
    def letter(self) -> str:
        return self.decision[0]


def selected_variants(features: str) -> tuple[str, ...]:
    """`--features` -> the feature sets to measure, in the order they are reported.

    `both` is two probes over two feature sets, never one probe over a concatenation:
    mixing the probabilities into the embedding would answer neither of the two questions,
    and the point of F109 is to know which of the two sets carries what.
    """
    if features == BOTH:
        return (PROBS, EMBED)
    if features not in (PROBS, EMBED):
        raise SystemExit(f"--features: ожидается {'|'.join(FEATURE_CHOICES)}, "
                         f"получено {features!r}")
    return (features,)


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

    Note what this function does NOT depend on: the features. Both variants of F109 are
    split by the same labels with the same seed, so they are measured on the same frames.
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


def feature_matrix(samples: list[Sample], variant: str) -> np.ndarray:
    """The frames' vectors of one variant, as one array — and a refusal if they are ragged.

    An empty width means the variant was never computed (asking for `embed` on samples
    built without one), and a mixed width means two different CLIP models produced the
    file. Both are errors here rather than a numpy exception three lines later.
    """
    widths = {len(s.features(variant)) for s in samples}
    if widths == {0}:
        raise SystemExit(f"признаки «{variant}» не посчитаны — измерять нечего")
    if len(widths) != 1:
        raise SystemExit(f"признаки «{variant}»: разная размерность у кадров "
                         f"{sorted(widths)} — так сравнивать нельзя")
    return np.asarray([s.features(variant) for s in samples], dtype=np.float64)


def run_probe(samples: list[Sample], test_size: float, seed: int,
              variant: str = PROBS) -> ProbeRun:
    """Train on the training part, answer the held-out part, return the answers.

    A plain logistic regression: simple, explainable, and — crucially — not tuned. The only
    knobs touched are the ones that make it converge at all (`PROBE_MAX_ITER`) and the fixed
    regularization (`PROBE_C`), never the ones that would be chosen by looking at the score
    it is about to be judged by. Both are the same for both variants: what differs between
    the two runs is the features and nothing else.
    """
    from sklearn.linear_model import LogisticRegression

    if not samples:
        raise SystemExit("нет размеченных кадров (media_class.source='vlm') — нечего учить")
    labels = [s.label for s in samples]
    train_idx, test_idx = stratified_split(labels, test_size, seed)
    if len({labels[i] for i in train_idx}) < 2:
        raise SystemExit("в обучающей части меньше двух классов — обучать нечему")
    features = feature_matrix(samples, variant)

    model = LogisticRegression(C=PROBE_C, max_iter=PROBE_MAX_ITER, random_state=seed)
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


def best_gate(curve: list[GateRow]) -> GateRow | None:
    """The row that preserves the most changed verdicts within the budget of outcome B.

    None — the grid has no N inside the budget at all, which is a statement about the grid
    and not about the probe, and the report says so in those words.
    """
    return max((r for r in curve if r.share <= MAX_SMART_GATE_SHARE),
               key=lambda r: r.kept_frac, default=None)


def do_nothing(answers: list[Answer]) -> tuple[int, int, int]:
    """The baseline: (fast verdicts that already match the VLM, frames compared, unknown).

    Computed from the snapshot — the fast verdict against the VLM label — and never from
    the probe's answers. Frames missing from the snapshot have no fast verdict to keep, so
    they are counted apart rather than being counted as a match.

    It follows that the baseline does not depend on the variant: the split is shared, so
    both probes are compared against the same row, and the report prints it once.
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
            # N — доля КАНДИДАТОВ, а не коллекции: сравнивать её с
            # CURRENT_CANDIDATE_SHARE (доля коллекции) значит сравнивать разные
            # знаменатели и занижать выигрыш втрое. Пишем оба числа явно.
            f"это {pick.share:.0%} нынешних кандидатов, то есть "
            f"{pick.share * CURRENT_CANDIDATE_SHARE:.1%} коллекции вместо "
            f"{CURRENT_CANDIDATE_SHARE:.1%} — умный гейт вместо порога")
    best = best_gate(curve)
    tail = (f"лучший гейт до {MAX_SMART_GATE_SHARE:.0%} — N={best.share:.0%} с "
            f"{best.kept_frac:.1%} изменений при {MIN_CHANGES_KEPT:.0%}"
            if best else f"в сетке нет N <= {MAX_SMART_GATE_SHARE:.0%}")
    return "C", (
        f"ни A, ни B: согласие {agreement:.1%} при {MIN_AGREEMENT:.0%}, утечка документов "
        f"{leak:.1%} при {MAX_DOCUMENT_LEAK:.0%} ({leaked} из {documents}), {tail}. "
        f"CLIP не выражает того, что видит VLM — преимущество модели именно в том, чего в "
        f"этих признаках нет; ярус остаётся как есть, тему закрываем с цифрами")


def measure_variant(samples: list[Sample], variant: str, test_size: float,
                    seed: int) -> VariantReport:
    """One feature set, from the split to the outcome letter."""
    run = run_probe(samples, test_size, seed, variant)
    return VariantReport(
        variant=variant,
        dim=len(samples[0].features(variant)),
        run=run,
        evaluation=confusion(run.answers),
        curve=gate_curve(run.answers),
    )


def overall_decision(reports: list[VariantReport]) -> tuple[str, str]:
    """The joint outcome over the measured feature sets, and which of them earned it.

    A single set reaching A or B is enough for the joint letter: the question was whether
    CLIP features can carry the VLM's verdict, and one set that carries it answers it yes.
    C is joint in the strict sense — every set had to fail, which is what the F109 brief
    means by closing the topic for CLIP features as a whole and not for one encoding of
    them. On a tie the earlier set wins, which is `probs`: the cheaper one, already
    computed by the tier.
    """
    if not reports:
        raise SystemExit("не измерен ни один набор признаков")
    best = min(reports, key=lambda r: _OUTCOME_ORDER[r.letter])
    letter, why = best.decision
    if letter != "C":
        return letter, (f"его даёт набор «{best.variant}» "
                        f"({VARIANT_TITLES[best.variant]}, {best.dim} признаков): {why}")
    rows = "; ".join(
        f"«{r.variant}» ({r.dim}) — согласие {r.evaluation.agreement:.1%}"
        for r in reports)
    return "C", (
        f"ни A, ни B ни для одного набора признаков: {rows}. Эмбеддинг несёт всё, что "
        f"модель вообще видит, и этого тоже не хватает — значит дело не в сжатии признаков "
        f"до вероятностей по промптам, а в том, что преимущество VLM лежит вне CLIP. "
        f"Ярус остаётся как есть, тема закрыта для обоих наборов признаков")


def _pct(part: int, whole: int) -> str:
    return f"{100.0 * part / whole:.1f}%" if whole else "—"


def format_header(reports: list[VariantReport], total: int, test_size: float,
                  seed: int) -> str:
    """The one header of the run: the sample, the split, and the widths being compared."""
    run = reports[0].run
    dims = ", ".join(f"{r.variant} — {r.dim}" for r in reports)
    return "\n".join([
        "=" * 92,
        f"ЗОНД ПО CLIP-ПРИЗНАКАМ: выучивается ли вердикт VLM ({total} размеченных кадров)",
        f"наборы признаков и их размерность: {dims}",
        f"обучение: {run.trained_on} кадров, отложено: {len(run.answers)} "
        f"(доля {test_size:g}, seed {seed}); разделение одно на все наборы, "
        f"все метрики ниже — только по отложенной части",
        "=" * 92,
    ])


def format_variant_header(report: VariantReport) -> str:
    """Which feature set the block below belongs to, and how wide it is."""
    return "\n".join([
        "-" * 92,
        f"ПРИЗНАКИ: {VARIANT_TITLES[report.variant]} ({report.variant}), "
        f"размерность {report.dim}",
        "-" * 92,
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


def format_comparison(reports: list[VariantReport]) -> str:
    """The variants side by side — every measured one, always, with its width.

    Printing only the winner would turn the measurement into an advertisement for whichever
    feature set happened to win, so the table has a row per variant and the reader compares
    them; the outcome column is per row, and the joint letter is printed under it.
    """
    lines = [
        "СРАВНЕНИЕ НАБОРОВ ПРИЗНАКОВ (печатаются все измеренные, «лучший» не выбирается):",
        f"{'признаки':<10}{'размерность':>12}{'согласие':>10}{'документы':>11}"
        f"{'гейт N<=30%':>16}{'исход':>7}",
    ]
    for r in reports:
        leaked, documents = r.evaluation.document_leak()
        leak = f"{leaked / documents:.1%}" if documents else "—"
        gate = best_gate(r.curve)
        cell = f"N={gate.share:.0%}: {gate.kept_frac:.1%}" if gate else "—"
        lines.append(f"{r.variant:<10}{r.dim:>12d}{r.evaluation.agreement:>10.1%}"
                     f"{leak:>11}{cell:>16}{r.letter:>7}")
    lines.append(f"пороги одни и те же для всех строк: согласие {MIN_AGREEMENT:.0%}, "
                 f"документы {MAX_DOCUMENT_LEAK:.0%}, изменения {MIN_CHANGES_KEPT:.0%} "
                 f"при N <= {MAX_SMART_GATE_SHARE:.0%}")
    return "\n".join(lines)


def format_outcome(report: VariantReport) -> str:
    letter, why = report.decision
    return f"ИСХОД {letter} ({report.variant}, {report.dim} признаков): {why}"


def format_overall_outcome(reports: list[VariantReport]) -> str:
    letter, why = overall_decision(reports)
    return f"ИСХОД {letter} (общий по наборам признаков): {why}"


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


def chunk_features(classifier: CachingFeatureClassifier, paths: list[str],
                   groups: list[list[str]]) -> list[FrameFeatures]:
    """One CLIP pass over a chunk of frames -> both feature sets of each of them.

    The frames are encoded ONCE (`encode`, which returns the L2-normalized image vector)
    and the prompt groups are then scored off those same vectors (`score` — a matmul and a
    softmax, no decoding). So the price of adding the embedding variant is not a second
    pass, and — more importantly — the two variants describe the same encoding of the same
    frames, which is the only way the comparison means anything.

    A frame that did not decode gets a row of zeros in the probabilities — exactly what
    `CachingFeatureClassifier.__call__` hands the pipeline for it — and None for the
    embedding, whose width is not known here (see collect_samples).
    """
    feats = classifier.encode(paths)
    valid = [i for i, f in enumerate(feats) if f is not None]
    zeros = (0.0,) * sum(len(prompts) for prompts in groups)
    out: list[FrameFeatures] = [FrameFeatures(probs=zeros, embed=None) for _ in paths]
    if not valid:
        return out
    stacked = np.stack([feats[i] for i in valid])
    scored = [classifier.score(stacked, prompts) for prompts in groups]
    for k, i in enumerate(valid):
        row = np.concatenate([group[k] for group in scored])
        embed = feats[i]
        assert embed is not None                      # `valid` says so
        out[i] = FrameFeatures(probs=tuple(float(x) for x in row),
                               embed=tuple(float(x) for x in embed))
    return out


def embeddings_of(frames: list[FrameFeatures]) -> list[tuple[float, ...]]:
    """The embeddings, with the frames that did not decode filled with zeros of that width.

    The width comes from the frames that did encode, so it is only known after the whole
    pass. A collection where nothing decoded gets empty vectors and `feature_matrix` then
    refuses to measure the variant rather than reporting on rows of nothing.
    """
    width = next((len(f.embed) for f in frames if f.embed is not None), 0)
    zeros = (0.0,) * width
    return [zeros if f.embed is None else f.embed for f in frames]


def collect_samples(cfg: Config, rows: list[sqlite3.Row],
                    before: dict[int, str]) -> list[Sample]:  # pragma: no cover — ML
    """CLIP over the labelled frames -> the feature vectors of both variants.

    All three prompt groups are run with the pipeline's own classifier, in the order of
    FEATURE_GROUPS, and concatenated; the embedding is the same classifier's image vector.
    One decode and one `encode_image` per frame for both (see chunk_features).
    """
    s = naming_settings(cfg)
    classifier = clip_classifier(s)
    if not isinstance(classifier, CachingFeatureClassifier):
        raise SystemExit("ожидался кэширующий классификатор CLIP "
                         "(landmarks.CachingFeatureClassifier) — эмбеддинг взять неоткуда")
    groups = [[prompt for _cls, prompt in classes] for _name, classes in FEATURE_GROUPS]

    frames: list[FrameFeatures] = []
    done = 0
    for chunk in batched(rows, s.clip_batch_size):
        frames.extend(chunk_features(classifier, [r["path"] for r in chunk], groups))
        done += len(chunk)
        print(f"  CLIP {done}/{len(rows)}", end="\r", flush=True)
    print(" " * 40, end="\r")

    embeddings = embeddings_of(frames)
    return [
        Sample(probs=frame.probs, embed=embedding, label=str(r["verdict"]),
               before=before.get(int(r["id"])))
        for r, frame, embedding in zip(rows, frames, embeddings)
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--before", required=True,
                    help="JSON snapshot of media_class taken BEFORE the deep run")
    ap.add_argument("--features", choices=FEATURE_CHOICES, default=DEFAULT_FEATURES,
                    help="which features the probe learns on: probs — the prompt "
                         "probabilities, embed — the image embedding, both — each of them "
                         f"separately, side by side (default {DEFAULT_FEATURES})")
    ap.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE,
                    help=f"share held out for the metrics (default {DEFAULT_TEST_SIZE:g})")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()

    variants = selected_variants(args.features)
    cfg = load_config(args.config)
    before = load_before(Path(args.before))
    rows = labelled_rows(str(cfg.database))
    if not rows:
        raise SystemExit("в индексе нет ответов модели (media_class.source='vlm') — "
                         "учить зонд не на чем")
    print(f"размечено моделью: {len(rows)} кадров, снимок «до»: {len(before)} строк")

    samples = collect_samples(cfg, rows, before)
    reports = [measure_variant(samples, v, args.test_size, args.seed) for v in variants]

    print()
    print(format_header(reports, len(samples), args.test_size, args.seed))
    # The baseline is the same for every variant (one split, one held-out part), so it is
    # printed once, above them: it is what both of them are compared against.
    print(format_baseline(reports[0].run.answers))
    for report in reports:
        print(format_variant_header(report))
        print(format_agreement(report.evaluation))
        print("-" * 92)
        print(format_confusion(report.evaluation))
        print("-" * 92)
        print(format_documents(report.evaluation))
        print("-" * 92)
        print(format_gate_curve(report.curve))
        print("-" * 92)
        print(format_outcome(report))
    print("=" * 92)
    print(format_comparison(reports))
    print("=" * 92)
    print(format_overall_outcome(reports))
    return 0


if __name__ == "__main__":
    sys.exit(main())
