"""What the candidate gate never shows the model — and what that costs in documents.

The deep tier only ever looks at frames the fast CLIP suspected of being goods
(`naming.product_candidate_min = 0.4`). On the live run of 2026-07-28 that was 7 896
frames of 24 196; the other 16 300 the model has never seen. We know what the tier
FIXES (2 592 changed verdicts, 2 202 of them into `product`). Nobody has ever measured
what it MISSES.

The asymmetry is the point. A photo that lands in _Documents is found when somebody goes
through that folder; a document that lands in a city folder is never found — it looks
like one more frame among thousands. And the gate decides on a single feature, "how much
does this look like a product", so a document that does not look like goods cannot reach
the model by construction.

The method, and the two things it is careful about:

1. The population is exactly the frames the tier processed and the gate did NOT let
   through — `media_class.tier = 'vlm' AND source != 'vlm'`. Those are the frames the
   model was never asked about; a frame with `source='vlm'` is one it answered.
2. The sample is random with a fixed seed, never the first N rows: the collection is
   ordered by time, so the first 300 rows would be one trip.
3. Every sampled frame is shown to the model with the SAME prompt and read with the SAME
   parser the tier uses — both imported from `junk.py`. A private copy of the prompt
   would measure a different question than the one the product asks.
4. The answer is compared with the fast verdict already stored for that frame.

Cost: ~0.78 s per frame (measured on the run of 2026-07-28), so 300 frames ≈ 4 minutes
of GPU.

Two things the arithmetic below refuses to blur:

* An answer that names no label is counted on its own line, never as agreement. The
  tier maps an unrecognized answer to `personal_photo` on purpose (conservative — see
  junk._VLM_LABEL_TO_VERDICT), and taking that fallback for an answer would make this
  measurement quieter the worse the model behaves.
* A frame whose fast verdict is `screenshot` or `meme` is outside the model's three
  labels: it cannot agree with the fast tier no matter what it sees. Such frames are in
  the population (the tier processed them) and are reported separately instead of being
  counted as disagreement — otherwise the report would price the vocabulary gap and call
  it a gate that is too narrow.

Privacy: counts and labels only. No path, no basename, no file id and no recognized
content is printed or stored — the model is shown documents exactly as the product tier
already shows it documents, and only flags come back out (the rule of
measure_ocr_gate.py / measure_vlm_speed.py / measure_candidate_gate.py before it).

The database is opened `mode=ro`: a measurement writes nothing.

Usage (from the repo root, with a GPU venv — `uv sync --extra gpu --extra vlm`):
    python scripts/measure_gate_misses.py                  # 300 frames, seed 20260729
    python scripts/measure_gate_misses.py --sample 500
    python scripts/measure_gate_misses.py --sample 50 --seed 1   # a quick smoke run
"""
from __future__ import annotations

import argparse
import random
import sqlite3
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sorta import junk, naming  # noqa: E402
from sorta.config import Config, load_config  # noqa: E402

# The sample the brief pre-registered. Below it the percentages are still printed, with a
# warning: 300 frames put a 1% effect at three frames, and less than that is noise.
MIN_SAMPLE = 300

# Fixed in the brief, so a rerun draws the same frames as the run being argued about.
DEFAULT_SEED = 20260729

# Seconds per frame in the deep tier, measured on the live run of 2026-07-28 (95 minutes
# over 7 896 candidates). The price of widening the gate is this number times the
# population — the same constant measure_candidate_gate.py prices the curve with.
SEC_PER_FRAME = 0.78

# The verdicts the model's three labels map to (junk._VLM_LABEL_TO_VERDICT). A fast
# verdict outside this set cannot be confirmed by the model — see out_of_vocabulary.
MODEL_VERDICTS = frozenset(junk._VLM_LABEL_TO_VERDICT.values())

PHOTO_VERDICT = "photo"
DOCUMENT_VERDICT = "document"

# --- Pre-registered acceptance criteria (F110) -------------------------------
#
# Written down before the first run, so the numbers cannot talk anybody into an outcome
# afterwards.
#
# A — the gate is too narrow: disagreements >= 5% of the compared frames, whatever they
#     are. The next feature is widening it, and this run already prices that in minutes.
# B — narrow for documents only: fewer than 5% disagreements, but `photo -> document`
#     >= 1% (~160 documents over the unseen population). Then it is not the whole gate
#     that widens — the model gets a separate cheap question about documents on a wider
#     population.
# C — the gate is set correctly: under both. A good result, and a real one: it means 0.4
#     selects for something rather than at random.
#
# The two conditions of A overlap with B on purpose (the brief writes A as an OR). B is
# the narrower case and names a different action, so it is checked first — see decide().
MISMATCH_MIN = 0.05
DOCUMENT_MISS_MIN = 0.01


@dataclass(frozen=True)
class Answer:
    """One frame of the unseen population after the model was finally asked about it.

    `fast` is the verdict the fast tier stored for it; `deep` is the verdict the model's
    answer maps to, and None when there is no answer to map — the reply named no label,
    the frame did not decode, or the model raised on it. None is never agreement.

    Nothing here identifies the frame: no id, no path, no answer text.
    """
    fast: str
    deep: str | None

    @property
    def out_of_vocabulary(self) -> bool:
        """The fast verdict is one the model has no label for (screenshot, meme)."""
        return self.fast not in MODEL_VERDICTS

    @property
    def comparable(self) -> bool:
        return self.deep is not None and not self.out_of_vocabulary

    @property
    def agreed(self) -> bool:
        return self.comparable and self.fast == self.deep


@dataclass(frozen=True)
class Tally:
    """The whole report as counters — every percentage below is arithmetic over these.

    `pairs` holds only the comparable frames whose verdicts differ, so the breakdown and
    the number of disagreements can never contradict each other.
    """
    population: int          # rows of the unseen population in the index
    sample: int              # frames actually shown to the model
    agreed: int
    pairs: Counter[tuple[str, str]]
    unparsed: int            # the model named no label — counted apart, never as agreement
    out_of_vocabulary: int   # fast verdict the model has no label for

    @property
    def mismatched(self) -> int:
        return sum(self.pairs.values())

    @property
    def compared(self) -> int:
        """Frames where the two verdicts could differ at all — the denominator below."""
        return self.agreed + self.mismatched

    @property
    def mismatch_frac(self) -> float:
        """Share of compared frames where the model disagrees (0.0 — nothing compared).

        The denominator is the compared frames rather than the whole sample: an unparsed
        answer carries no verdict and a screenshot has no label the model could confirm,
        so both would only dilute the rate. It also makes the criterion slightly easier
        to trip, which is the safe direction — outcome A is "go and look".
        """
        return self.mismatched / self.compared if self.compared else 0.0

    @property
    def agreement(self) -> float:
        """Share of compared frames where the model confirms the fast tier (1.0 if none)."""
        return self.agreed / self.compared if self.compared else 1.0

    @property
    def documents(self) -> int:
        """`photo -> document`: personal papers sitting in a city folder today."""
        return self.pairs[(PHOTO_VERDICT, DOCUMENT_VERDICT)]

    @property
    def document_frac(self) -> float:
        return self.documents / self.compared if self.compared else 0.0

    @property
    def documents_forecast(self) -> float:
        """That share over the whole unseen population, in frames.

        An upper estimate: the share is measured on the comparable frames, and the
        population also holds screenshots and memes, which cannot produce this pair.
        """
        return self.document_frac * self.population

    @property
    def population_minutes(self) -> float:
        """What showing the whole unseen population to the model would cost."""
        return self.population * SEC_PER_FRAME / 60.0


def model_verdict(answer: str) -> str | None:
    """The model's raw answer -> a verdict, or None when it named no label at all.

    The parsing is the tier's own (`junk._vlm_label`), plus the one thing the tier
    deliberately hides: an answer it does not recognize becomes `personal_photo`, which
    is the right conservative default for a pipeline and the wrong one for a
    measurement — every failure of the model would arrive as agreement with the fast
    tier. So the fallback is DETECTED rather than reimplemented: the answer parses as
    `personal_photo` only if it actually says so.
    """
    label = junk._vlm_label(answer)
    if label == junk._VLM_FALLBACK_LABEL and label not in answer.lower():
        return None
    return junk._VLM_LABEL_TO_VERDICT[label]


def unseen_rows(db_path: str) -> list[sqlite3.Row]:
    """The population: frames the deep tier processed and the gate did not let through.

    `tier='vlm'` — a deep run handled the row; `source != 'vlm'` — the model was never
    asked about it (junk.classify writes `source='vlm'` exactly for the frames it
    answered). Rows with no `tier` are outside the question: no deep run has touched them.

    `mode=ro` is the contract of the brief — a measurement writes nothing.
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute(
            """SELECT f.id, f.path, mc.verdict AS verdict
               FROM files f JOIN media_class mc ON mc.file_id = f.id
               WHERE mc.tier = 'vlm' AND mc.source != 'vlm'
                 AND f.dup_of IS NULL AND f.error IS NULL
               ORDER BY f.id"""
        ).fetchall()
    finally:
        conn.close()


def take_sample(rows: Sequence[sqlite3.Row], n: int, seed: int) -> list[sqlite3.Row]:
    """`n` frames of the population, random and reproducible from the seed.

    Random and not the first n: the collection is ordered by id, i.e. by the time the
    files were indexed, so a prefix of it would be one trip and the report would describe
    that trip. Frames that are no longer on disk are dropped — there is nothing to show
    the model — and dropped AFTER the shuffle, so their absence does not reorder the rest.
    """
    picked = list(rows)
    random.Random(seed).shuffle(picked)
    return [r for r in picked if Path(r["path"]).exists()][:n]


def recording(describe: Callable[[Sequence[Any], str, int], str],
              sink: list[str]) -> Callable[[Sequence[Any], str, int], str]:
    """The runtime, with a copy of every raw answer appended to `sink`.

    The tier's classifier returns a parsed label and throws the raw reply away, which is
    exactly the reply this measurement needs to see (an answer that named no label is a
    line of the report, not a `personal_photo`). Wrapping the RUNTIME instead of copying
    the classifier keeps the prompt, the decode, the token budget and the parser the
    tier's own.

    The sink holds one frame's answer at a time (`ask_model` empties it before every
    frame) and is never printed or written anywhere: it is what the model said about
    somebody's documents.
    """
    def describe_and_record(frames: Sequence[Any], prompt: str, max_new_tokens: int) -> str:
        answer = describe(frames, prompt, max_new_tokens)
        sink.append(str(answer))
        return answer

    return describe_and_record


def ask_model(classifier: junk.VlmClassifyFn, sink: list[str],
              rows: Sequence[sqlite3.Row]) -> list[Answer]:
    """Show every frame to the model and pair its answer with the stored fast verdict.

    The classifier is the tier's (`junk.vlm_classifier_from`), so a frame that does not
    decode never reaches the model — the sink stays empty for it and the frame is counted
    as unanswered rather than as the fallback label. A model error on one frame does not
    end the measurement, the same way it does not end a run: it is one more frame with no
    answer.

    The sink is emptied before every frame, so no transcript of the model's answers ever
    accumulates — one answer is alive at a time and only its parsed label survives it.
    """
    out: list[Answer] = []
    for done, row in enumerate(rows, 1):
        sink.clear()
        try:
            classifier(row["path"])
        except Exception:  # noqa: BLE001 — one frame must not end the measurement
            pass
        raw = sink[-1] if sink else None
        out.append(Answer(fast=str(row["verdict"]),
                          deep=model_verdict(raw) if raw is not None else None))
        print(f"  VLM {done}/{len(rows)}", end="\r", flush=True)
    sink.clear()
    print(" " * 40, end="\r")
    return out


def tally(answers: Sequence[Answer], population: int) -> Tally:
    """The counters of the report. Every frame lands in exactly one of the four buckets."""
    pairs: Counter[tuple[str, str]] = Counter()
    agreed = unparsed = out_of_vocabulary = 0
    for a in answers:
        if a.deep is None:
            unparsed += 1
        elif a.out_of_vocabulary:
            out_of_vocabulary += 1
        elif a.agreed:
            agreed += 1
        else:
            pairs[(a.fast, a.deep)] += 1
    return Tally(population=population, sample=len(answers), agreed=agreed, pairs=pairs,
                 unparsed=unparsed, out_of_vocabulary=out_of_vocabulary)


def decide(t: Tally) -> tuple[str, str]:
    """(the outcome letter, one line saying why) — by the pre-registered criteria above.

    B is checked before A because the brief writes A as an OR that swallows it: the case
    "few disagreements, but documents among them" has an action of its own (a cheap
    document question on a wider population), and reporting it as A would send the next
    feature after the whole gate.
    """
    if not t.compared:
        return "C", ("сравнивать не с чем: ни один кадр выборки не дал ответа, который "
                     "можно сопоставить с быстрым вердиктом")
    documents_bad = t.document_frac >= DOCUMENT_MISS_MIN
    if t.mismatch_frac < MISMATCH_MIN and documents_bad:
        return "B", (
            f"расхождений {t.mismatch_frac:.1%} — меньше {MISMATCH_MIN:.0%}, но "
            f"photo -> document {t.document_frac:.1%} при {DOCUMENT_MISS_MIN:.0%} "
            f"(~{t.documents_forecast:.0f} документов на непросмотренной популяции): "
            f"гейт узок именно по документам — расширять не его целиком, а задавать "
            f"модели отдельный дешёвый вопрос про документы на более широкой популяции")
    if t.mismatch_frac >= MISMATCH_MIN:
        return "A", (
            f"расхождений {t.mismatch_frac:.1%} при пороге {MISMATCH_MIN:.0%} "
            f"(photo -> document {t.document_frac:.1%}, ~{t.documents_forecast:.0f} "
            f"документов на непросмотренной популяции): гейт узок, следующей фичей — "
            f"его расширение, цена полного прогона популяции "
            f"~{t.population_minutes:.0f} мин")
    return "C", (
        f"расхождений {t.mismatch_frac:.1%} при {MISMATCH_MIN:.0%} и photo -> document "
        f"{t.document_frac:.1%} при {DOCUMENT_MISS_MIN:.0%} "
        f"(~{t.documents_forecast:.0f} документов на популяции): гейт настроен верно — "
        f"то, что модель не видит, ей и не нужно видеть; порог 0.4 отбирает не случайно, "
        f"тему закрываем с цифрами")


def _pct(part: int, whole: int) -> str:
    return f"{100.0 * part / whole:.1f}%" if whole else "—"


def format_summary(t: Tally) -> str:
    """The head of the report: the population, the sample and the two shares."""
    return "\n".join([
        "=" * 92,
        f"ЧЕГО ГЕЙТ НЕ ПОКАЗЫВАЕТ МОДЕЛИ: {t.sample} кадров показано из "
        f"{t.population} непросмотренных",
        "=" * 92,
        f"РАСХОЖДЕНИЯ С БЫСТРЫМ ВЕРДИКТОМ: {t.mismatched} из {t.compared} "
        f"({t.mismatch_frac:.1%}) при пороге исхода A {MISMATCH_MIN:.0%}",
        f"СОГЛАСИЕ С БЫСТРЫМ ЯРУСОМ: {t.agreed} из {t.compared} ({t.agreement:.1%}) — "
        f"это и есть оценка того, стоит ли расширять гейт",
    ])


def format_pairs(t: Tally) -> str:
    """The breakdown, per label pair. Counters only — no path, no id, no answer text."""
    lines = ["РАЗБИВКА ПО ПАРАМ (быстрый вердикт -> ответ модели), счётчиками:"]
    for (fast, deep), count in sorted(t.pairs.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"  {fast} -> {deep}: {count}")
    if len(lines) == 1:
        lines.append("  (расхождений нет: модель подтвердила каждый сопоставимый кадр)")
    lines.append(f"  сумма по парам: {t.mismatched} — сходится с числом расхождений")
    return "\n".join(lines)


def format_documents(t: Tally) -> str:
    """`photo -> document` on its own line, whatever the outcome.

    An average over three hundred frames hides the three that are personal papers, and
    those are the ones already laid out by city — a document in a city folder is not
    found by anybody going through a folder, unlike a photo in _Documents.
    """
    return "\n".join([
        f"PHOTO -> DOCUMENT: {t.documents} из {t.compared} ({t.document_frac:.1%}) "
        f"при пороге {DOCUMENT_MISS_MIN:.0%}",
        f"  прогноз на непросмотренную популяцию ({t.population} кадров): "
        f"~{t.documents_forecast:.0f} документов лежат сегодня в папках городов",
        "  (оценка сверху: доля считается по сопоставимым кадрам, а в популяции есть и "
        "скриншоты/мемы, которые такой пары дать не могут)",
    ])


def format_skipped(t: Tally) -> str:
    """The frames that carry no comparison — apart, and never as agreement."""
    return "\n".join([
        "НЕ УЧАСТВУЮТ В СРАВНЕНИИ:",
        f"  ответ не разобран: {t.unparsed} ({_pct(t.unparsed, t.sample)} выборки) — "
        f"модель не назвала метки, кадр не декодировался или упал; согласием НЕ считается",
        f"  вне словаря модели: {t.out_of_vocabulary} "
        f"({_pct(t.out_of_vocabulary, t.sample)} выборки) — быстрый вердикт "
        f"screenshot/meme, у модели такой метки нет и подтвердить его нечем",
    ])


def format_price(t: Tally) -> str:
    """What widening the gate to the whole unseen population would cost in wall time."""
    return (f"ЦЕНА РАСШИРЕНИЯ: {t.population} кадров x {SEC_PER_FRAME:.2f} с = "
            f"~{t.population_minutes:.0f} мин GPU на полный прогон непросмотренной "
            f"популяции через модель")


def format_outcome(t: Tally) -> str:
    letter, why = decide(t)
    return f"ИСХОД {letter}: {why}"


def build_classifier(cfg: Config,
                     sink: list[str]) -> junk.VlmClassifyFn:  # pragma: no cover — ML
    """The tier's own classifier over the shared runtime, with the raw answers observable.

    This is exactly what `junk.qwen_vlm_classifier` builds — `vlm_classifier_from` over
    `naming.shared_vlm(model)` at `vlm.max_edge` — with the runtime wrapped by
    `recording`. The wrapper is a plain function rather than a SplitVlm, so the classifier
    takes the serial path: 300 frames at ~0.78 s is four minutes, and a pipelined pass
    would only make it harder to say which answer belongs to which frame.
    """
    return junk.vlm_classifier_from(
        recording(naming.shared_vlm(cfg.vlm.model), sink), max_edge=cfg.vlm.max_edge)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--sample", type=int, default=MIN_SAMPLE,
                    help=f"frames to show the model (default {MIN_SAMPLE} — the "
                         f"pre-registered minimum)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED,
                    help=f"the sampling seed (default {DEFAULT_SEED}, fixed in the brief)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    rows = unseen_rows(str(cfg.database))
    if not rows:
        raise SystemExit(
            "непросмотренная популяция пуста: в индексе нет строк с tier='vlm' и "
            "source != 'vlm' — глубокий ярус на этой коллекции не запускался, "
            "измерять нечего")
    sample = take_sample(rows, args.sample, args.seed)
    if not sample:
        raise SystemExit(f"популяция — {len(rows)} кадров, но ни одного файла нет на "
                         f"диске: показывать модели нечего")
    print(f"непросмотренная популяция (tier='vlm', source != 'vlm'): {len(rows)} кадров")
    print(f"выборка: {len(sample)} кадров, seed {args.seed} (случайная, не первые по id)")
    print(f"модель: {cfg.vlm.model}, ~{SEC_PER_FRAME:.2f} с/кадр -> "
          f"~{len(sample) * SEC_PER_FRAME / 60.0:.0f} мин")
    if len(sample) < MIN_SAMPLE:
        print(f"ВНИМАНИЕ: кадров меньше {MIN_SAMPLE} — на такой выборке доли ниже "
              f"считаются, но исход по ним не доказан")

    sink: list[str] = []
    answers = ask_model(build_classifier(cfg, sink), sink, sample)
    result = tally(answers, population=len(rows))

    print()
    print(format_summary(result))
    print("-" * 92)
    print(format_pairs(result))
    print("-" * 92)
    print(format_documents(result))
    print("-" * 92)
    print(format_skipped(result))
    print("-" * 92)
    print(format_price(result))
    print("=" * 92)
    print(format_outcome(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
