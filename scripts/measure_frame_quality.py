"""Price the frame-quality cascade (F113/F130): what each tier sees, and what it got right.

Three of the numbers this feature ships with are guesses until somebody looks at a
distribution — the pet threshold, the sharpness band, and the CLIP score below which a
frame counts as "subject unclear". CLIP's accuracy on "is there a cat in this frame" has
never been measured in this project, so assuming it is exactly the mistake F110 warned
about. This script does not choose any of them. It prints what the collection actually
looks like — the score distribution, what each candidate threshold would fire on, and how
big the VLM band would be — and the thresholds in config.yaml stay a decision for a human
in front of that table.

Nothing is reimplemented here: `junk.clip_prompts`, `junk.pet_verdict`,
`junk.laplacian_variance`, `junk.preview_sharpness_detector` and `junk.uncertain_band` are
the same functions the pipeline runs, driven off the same `junk.quality_settings`. A
private copy of the arithmetic would drift and price a cascade that does not exist.

Privacy: nothing printed identifies a frame. No path, no basename; the cache stores file
ids and aggregates only (the same rule scripts/measure_ocr_gate.py follows).

F130 adds the block the pet cascade is accepted or rejected on. Its numbers — precision
97-99%, recall 66% — are a PREDICTION extrapolated from the F122 labelling, not a
measurement of this feature, and the brief pre-commits to reporting whichever way the
measurement goes. Two things are needed for that and both are here:

* `--features verify` prices the candidate threshold (how many frames the model is shown,
  how long that takes) and, after a live run, reports what the stored answers actually
  changed — labels removed from frames CLIP scored high, labels added below its threshold;
* `--labels` computes precision and recall over hand-labelled frames, BEFORE the check
  (the CLIP threshold alone, which is how the 92% / 54% of F122 was produced) and AFTER
  it, weighted back to the collection by score band so the two are comparable to F122 and
  to each other. `--write-labels` writes the stratified worksheet to fill in.

F157 adds the sweep `features.blur_review_max` is read off, in the same block: the blur
slice is an ORDERING, so what the grid prints is how many frames its first page would hold
at each value, and where to stop reading stays a decision for the person reading it.

F155 adds the sweep `features.face_sharpness_max` has to be read off: the same laplacian
measured INSIDE the face boxes, over the frames that have one. It prints under `--features
sharpness`, next to the whole-frame distribution it is meant to be compared with, and like
every table here it chooses nothing — the signal is ~25% precise and covers only the third
of a collection that has a face, so what the grid shows is the size of a review list.

Usage (from the repo root, with the venv python):
    python scripts/measure_frame_quality.py --features pets
    python scripts/measure_frame_quality.py --features pets sharpness band
    python scripts/measure_frame_quality.py --sample 500 --cache frame_quality.json
    python scripts/measure_frame_quality.py --cache frame_quality.json   # replay, no CLIP
    python scripts/measure_frame_quality.py --features verify             # after a run
    python scripts/measure_frame_quality.py --features verify --labels pets.json

A run costs one CLIP pass over the sample (the pet prompts ride in the same call the junk
stage makes) plus a preview decode per frame for the laplacian — minutes on a few hundred
frames. `--cache` writes the per-frame aggregates out so a different grid can be tried
without paying for the model again. The stored VLM answers are read from the DB on every
run, cache or no cache: a replay must not describe the state of the index before the run
that produced those answers.
"""
from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sorta import junk  # noqa: E402
from sorta.config import Config, load_config  # noqa: E402
from sorta.landmarks import batched, clip_classifier  # noqa: E402
from sorta.naming import naming_settings  # noqa: E402

# The brief's floor: fewer frames than this cannot support a threshold anybody should
# trust, so the default sits above it and going below is a deliberate flag.
MIN_SAMPLE = 200
DEFAULT_SAMPLE = 500

# The grid the pet threshold is read off. Wide on purpose — nobody knows yet whether the
# useful cut is at 0.3 or at 0.9, and a narrow grid centred on the current default would
# be the guess this script exists to avoid.
DEFAULT_PET_GRID = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
# F130: the grid the CANDIDATE threshold is read off — how many frames the check is shown.
# It reaches lower than the grid above because that is the whole point of the cascade: the
# selection can go down once something verifies what it selects.
DEFAULT_CANDIDATE_GRID = (0.1, 0.2, 0.3, 0.4, 0.5, 0.7)
# The measured cost of one local VLM answer, the number every minute below is counted with.
VLM_SECONDS_PER_FRAME = 0.78
# Percentiles are how a laplacian distribution is read: it has no natural units, so what
# matters is where the mass sits relative to itself.
PERCENTILES = (5, 10, 25, 50, 75, 90, 95)

# F130: the score bands a labelling sample is stratified over, and the bands the answers
# are weighted back to the collection with. Uniform in score rather than in count: the
# population is enormously top-heavy in the low bands, so a uniform-in-frames sample would
# spend every label on frames nobody was ever going to mark.
SCORE_BANDS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001)

FEATURES = ("pets", "sharpness", "band", "verify")

# F155: the grid the face-sharpness threshold is read off. It reaches far past the value
# the brief measured (200) on purpose: what the 68-frame sample showed is a direction, and
# a grid centred on the number it produced would only confirm it.
DEFAULT_FACE_GRID = (50.0, 100.0, 200.0, 300.0, 400.0, 600.0)

# F157: the grid `features.blur_review_max` is read off — how long the FIRST PAGE of the
# blur list would be at each value. It is not a threshold sweep in the sense the two grids
# above are: the slice is a ranking and every frame below is in it sooner or later, so what
# this prints is the size of a reading job, not the size of a verdict.
DEFAULT_BLUR_GRID = (90.0, 200.0, 300.0, 450.0, 700.0)

# F155 bumped this: `Frame` gained face_sharpness, so a cache written before it has one
# column too few and would load with the fields shifted.
CACHE_VERSION = 2


@dataclass(frozen=True)
class Frame:
    """The per-frame aggregate the tables need — and nothing that identifies a frame.

    `pet_class` is the winning pet subclass REGARDLESS of any threshold, and `pet_score`
    its score: the sweep below has to be able to ask what a threshold LOWER than the
    configured one would have fired on, which a class already filtered by the configured
    one cannot answer.
    """
    file_id: int
    pet_class: str | None
    pet_score: float
    sharpness: float | None  # None — the frame did not decode
    subject_score: float     # the junk-group probability of "a photograph"
    # F130: what the stored `frame_quality.pet_vlm` says about this frame — real |
    # depiction | none, or None for "the check never asked". Read from the DB rather than
    # from the cache, so a replay describes the index as it is now; defaulted so every
    # cache written before this feature still loads.
    pet_vlm: str | None = None
    # F155: the laplacian inside the sharpest FACE of the frame, or None — no face on it,
    # no faces run behind it, or a crop too small to measure. A different population from
    # `sharpness` above (a third of a collection, not all of it), which is why the sweep
    # below counts against the frames that have the number rather than against all of them.
    face_sharpness: float | None = None


def sample_rows(db_path: str, n: int, seed: int) -> list[sqlite3.Row]:
    """`n` canonical photos that still exist on disk, deterministic for a given seed."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """SELECT f.id, f.path FROM files f
               WHERE f.dup_of IS NULL AND f.error IS NULL AND f.media_type = 'photo'
               ORDER BY f.id"""
        ).fetchall()
    finally:
        conn.close()
    random.Random(seed).shuffle(rows)
    return [r for r in rows if Path(r["path"]).exists()][:n]


def percentiles(values: list[float], points: tuple[int, ...] = PERCENTILES
                ) -> list[tuple[int, float]]:
    """(percentile, value) pairs by nearest rank — no numpy interpolation games.

    Nearest rank because the number is going to be copied into a config file as a
    threshold, and a value that no frame actually has is a worse answer than one that
    some frame does.
    """
    if not values:
        return []
    ordered = sorted(values)
    out: list[tuple[int, float]] = []
    for p in points:
        rank = max(0, min(len(ordered) - 1, round(p / 100.0 * len(ordered)) - 1))
        out.append((p, ordered[rank]))
    return out


@dataclass(frozen=True)
class PetRow:
    """One row of the pet table: what a threshold would fire on."""
    threshold: float
    fired: int
    by_class: dict[str, int]


def sweep_pets(frames: list[Frame], thresholds: list[float]) -> list[PetRow]:
    """How many frames each threshold calls a pet, split by class.

    The rule replayed here is `junk.pet_verdict`'s own — the class is written when its
    score reaches the threshold — so the table cannot disagree with what the stage writes.
    """
    rows: list[PetRow] = []
    for threshold in thresholds:
        by_class: dict[str, int] = {}
        fired = 0
        for f in frames:
            if f.pet_class is not None and f.pet_score >= threshold:
                fired += 1
                by_class[f.pet_class] = by_class.get(f.pet_class, 0) + 1
        rows.append(PetRow(threshold, fired, by_class))
    return rows


def _pct(part: int, whole: int) -> str:
    return f"{100.0 * part / whole:.1f}%" if whole else "—"


def format_pets(frames: list[Frame], rows: list[PetRow], current: float) -> str:
    """The pet block: the score distribution first, the thresholds after it."""
    scores = [f.pet_score for f in frames]
    out = [
        "=" * 88,
        f"ЖИВОТНЫЕ (CLIP, {len(frames)} кадров): распределение скора",
        "  " + "  ".join(f"p{p}={v:.3f}" for p, v in percentiles(scores)),
        f"{'порог':>6} {'сработало':>18} {'по классам':>40}",
    ]
    for r in rows:
        mark = "*" if abs(r.threshold - current) < 1e-9 else " "
        classes = ", ".join(f"{cls} {n}" for cls, n in sorted(r.by_class.items())) or "—"
        out.append(f"{r.threshold:>5.2f}{mark}{r.fired:>10d} "
                   f"({_pct(r.fired, len(frames)):>6}) {classes:>40}")
    out.append("=" * 88)
    out.append("* — порог из конфига (features.pet_threshold). Класс пишется, только если "
               "скор >= порога;\nскор в БД пишется всегда, поэтому порог можно "
               "пересмотреть без нового прохода.")
    out.append("ГЛАЗАМИ: откройте несколько кадров каждого класса в UI и решите сами — "
               "точность CLIP\nна этой задаче никто не мерил, и таблица её не измеряет, "
               "она показывает только охват.")
    return "\n".join(out)


def format_sharpness(frames: list[Frame], q: junk.QualitySettings,
                     blur_thresholds: list[float], blur_current: float) -> str:
    """The sharpness block: the distribution the band has to be read off.

    F157 adds the second table — the depth of the blur list's first page at each candidate
    value of `features.blur_review_max`. The slice is an ORDERING, so a row here says how
    many frames a person would be handed before the button, and nothing about how many of
    them are blurred: on 300 hand-labelled frames precision runs 29% down to 12% across
    this grid while recall climbs 12% to 82%, which is why the number is a page depth.
    """
    values = [f.sharpness for f in frames if f.sharpness is not None]
    missing = len(frames) - len(values)
    low, high = q.sharpness_band
    inside = sum(1 for v in values if low <= v <= high)
    out = [
        "=" * 88,
        f"РЕЗКОСТЬ (дисперсия лапласиана, превью {q.sharpness_max_edge}px, "
        f"{len(values)} кадров)",
        "  " + "  ".join(f"p{p}={v:.1f}" for p, v in percentiles(values)),
        f"  текущая полоса {low:.1f}..{high:.1f}: внутри {inside} кадров "
        f"({_pct(inside, len(values))})",
    ]
    if missing:
        out.append(f"  не декодировалось: {missing} — резкость NULL, это не «0»")
    out.append(f"{'порог':>7} {'первая страница':>18} {'от кадров с резкостью':>26}")
    for t in blur_thresholds:
        mark = "*" if abs(t - blur_current) < 1e-9 else " "
        fired = sum(1 for v in values if v < t)
        out.append(f"{t:>7.1f}{mark}{fired:>17d} {_pct(fired, len(values)):>25}")
    out.append("=" * 88)
    out.append("Число зависит от разрешения (features.sharpness_max_edge) — меняя его, "
               "полосу\nнужно перемерить. Ниже полосы кадр смазан, выше — резкий; "
               "внутри решает VLM.")
    out.append("* — features.blur_review_max: это ГЛУБИНА ПЕРВОЙ СТРАНИЦЫ списка "
               "размытых, а не\nграница «размыто». Список отсортирован от самых мягких, "
               "и «показать ещё» идёт\nдальше по нему.")
    return "\n".join(out)


@dataclass(frozen=True)
class FaceRow:
    """One row of the face-sharpness sweep: what a threshold would put into the list."""
    threshold: float
    fired: int


def sweep_face(frames: list[Frame], thresholds: list[float]) -> list[FaceRow]:
    """How many frames WITH A FACE each threshold calls a blur candidate.

    Below and not at or below: the number is the top of an open list, the same reading
    `features.blur_review_max` has, and a frame exactly at the threshold is not in it.
    """
    return [FaceRow(t, sum(1 for f in frames
                           if f.face_sharpness is not None and f.face_sharpness < t))
            for t in thresholds]


def format_face_sharpness(frames: list[Frame], rows: list[FaceRow],
                          current: float) -> str:
    """F155: the face block — the distribution and the sweep the threshold is read off.

    Separate from the block above because it is a separate POPULATION: only frames a face
    was found on, a third of a collection, and every share below is counted against that
    third rather than against the whole. Mixing the two into one table would make the
    coverage of this signal look like the coverage of the other one.
    """
    values = [f.face_sharpness for f in frames if f.face_sharpness is not None]
    out = [
        "=" * 88,
        f"РЕЗКОСТЬ ЛИЦА (дисперсия лапласиана внутри вырезки, {len(values)} кадров "
        f"с лицом из {len(frames)})",
    ]
    if not values:
        out.append("  ни одного кадра с лицом — сначала нужен прогон стадии faces")
        out.append("=" * 88)
        return "\n".join(out)
    out.append("  " + "  ".join(f"p{p}={v:.1f}" for p, v in percentiles(values)))
    out.append(f"{'порог':>7} {'кандидатов':>14} {'от кадров с лицом':>22}")
    for r in rows:
        mark = "*" if abs(r.threshold - current) < 1e-9 else " "
        out.append(f"{r.threshold:>7.1f}{mark}{r.fired:>13d} "
                   f"{_pct(r.fired, len(values)):>21}")
    out.append("=" * 88)
    out.append("* — порог из конфига (features.face_sharpness_max). Это РАНЖИРОВАНИЕ, "
               "не вердикт:\nна размеченной выборке верны около четверти помеченных, "
               "и кадры без лиц этим\nсигналом не покрыты вовсе. Столбец в БД пишется "
               "всегда, поэтому порог можно\nпересмотреть без нового прохода.")
    return "\n".join(out)


def format_band(frames: list[Frame], q: junk.QualitySettings) -> str:
    """The band block: how many frames the cheap tiers did NOT settle, and why.

    Split by REASON because the two conditions are independent knobs: if the whole band is
    the subject condition, moving the sharpness numbers changes nothing.

    F186 retired the consumer of this band — the frame-quality question the model was
    asked about the frames inside it — so the block no longer prices a model pass. What it
    still does is what `features.sharpness_band_*` and `features.subject_score_min` are
    chosen by, which is why the block stayed when the question went.
    """
    total = len(frames)
    low, high = q.sharpness_band
    by_sharp = sum(1 for f in frames
                   if f.sharpness is not None and low <= f.sharpness <= high)
    by_subject = sum(1 for f in frames if f.subject_score < q.subject_score_min)
    inside = sum(1 for f in frames if junk.uncertain_band(f.sharpness, f.subject_score, q))
    return "\n".join([
        "=" * 88,
        f"НЕУВЕРЕННАЯ ПОЛОСА ({total} кадров)",
        f"  по резкости ({low:.1f}..{high:.1f}):       {by_sharp:>6d} "
        f"({_pct(by_sharp, total)})",
        f"  по сюжету (CLIP < {q.subject_score_min:.2f}):     {by_subject:>6d} "
        f"({_pct(by_subject, total)})",
        f"  итого в полосе (ИЛИ):          {inside:>6d} ({_pct(inside, total)})",
        "=" * 88,
        "Полоса больше никого ни о чём не спрашивает: вопрос к модели снят (F186).\n"
        "Это доля кадров, которую дешёвые признаки не разделили, — по ней и выбираются "
        "пороги.",
    ])


# --- F130: the cascade — what it costs, what it changed, and whether it was right -------


def read_pet_vlm(db_path: str, frames: list[Frame]) -> list[Frame]:
    """`frame_quality.pet_vlm` attached to the frames — the answers a live run stored.

    Read from the index and never from the cache: the whole reason to run this block is a
    run that has happened since, and a replay that described the state before it would be
    reporting on the wrong pass. A DB without the column (an index older than F130) simply
    answers nothing, so the block prints "the check has not run" instead of failing.
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        answers = {int(r["file_id"]): r["pet_vlm"] for r in conn.execute(
            "SELECT file_id, pet_vlm FROM frame_quality WHERE pet_vlm IS NOT NULL")}
    except sqlite3.OperationalError:
        answers = {}
    finally:
        conn.close()
    return [Frame(f.file_id, f.pet_class, f.pet_score, f.sharpness, f.subject_score,
                  answers.get(f.file_id)) for f in frames]


def format_candidates(frames: list[Frame], grid: list[float], current: float) -> str:
    """How many frames each candidate threshold sends to the model, and for how long.

    The table the candidate threshold is chosen from, and the only honest way to choose it:
    the price is linear in the count and the count is only knowable from the scores this
    collection actually has.
    """
    total = len(frames)
    out = [
        "=" * 88,
        f"КАНДИДАТЫ НА ПРОВЕРКУ VLM ({total} кадров в выборке)",
        f"{'порог':>6} {'кадров':>10} {'доля':>8} {'время на выборке':>18} "
        f"{'на 20 000 кадров':>18}",
    ]
    for threshold in grid:
        fired = sum(1 for f in frames if f.pet_score >= threshold)
        share = fired / total if total else 0.0
        mark = "*" if abs(threshold - current) < 1e-9 else " "
        out.append(
            f"{threshold:>5.2f}{mark}{fired:>10d} {_pct(fired, total):>8} "
            f"{fired * VLM_SECONDS_PER_FRAME / 60:>15.1f} мин "
            f"{20000 * share * VLM_SECONDS_PER_FRAME / 60:>15.1f} мин")
    out.append("=" * 88)
    out.append("* — порог из конфига (features.pet_candidate_threshold). "
               f"Цена кадра — {VLM_SECONDS_PER_FRAME} с, замер F113.")
    return "\n".join(out)


def format_answers(frames: list[Frame], q: junk.QualitySettings) -> str:
    """What the stored answers did to the labels — in both directions, counted separately.

    Both directions matter and they are different claims. Removing a label from a high
    score is precision (the plush toys); adding one below the threshold is recall (the
    animals a threshold cannot reach). A table that only totalled them would hide whichever
    half failed.
    """
    answered = [f for f in frames if f.pet_vlm is not None]
    if not answered:
        return "\n".join([
            "=" * 88,
            "ОТВЕТЫ VLM: в базе их нет — проверка (features.pets_verify) ещё не прогонялась"
            " на этой выборке",
            "=" * 88,
        ])
    by_answer: dict[str, int] = {}
    for f in answered:
        by_answer[f.pet_vlm or ""] = by_answer.get(f.pet_vlm or "", 0) + 1
    threshold = q.pet_threshold
    before = sum(1 for f in frames if junk.pet_label(None, f.pet_score, threshold))
    after = sum(1 for f in frames
                if junk.pet_label(f.pet_vlm, f.pet_score, threshold))
    removed = sum(1 for f in frames
                  if junk.pet_label(None, f.pet_score, threshold)
                  and not junk.pet_label(f.pet_vlm, f.pet_score, threshold))
    added = sum(1 for f in frames
                if not junk.pet_label(None, f.pet_score, threshold)
                and junk.pet_label(f.pet_vlm, f.pet_score, threshold))
    return "\n".join([
        "=" * 88,
        f"ОТВЕТЫ VLM ({len(answered)} кадров из {len(frames)} спрошено)",
        "  " + ", ".join(f"{name} {count}" for name, count in sorted(by_answer.items())),
        f"  помечено животными до проверки (порог {threshold:.2f}): {before:>5d}",
        f"  снято проверкой (изображение или нет животного):        {removed:>5d}",
        f"  добавлено проверкой (ниже порога, но живое):            {added:>5d}",
        f"  помечено после проверки:                                {after:>5d}",
        "=" * 88,
        "Метка снимается и ставится в обе стороны — это и есть каскад. Точность и полнота "
        "требуют\nразметки: --labels (см. --write-labels).",
    ])


@dataclass(frozen=True)
class Accuracy:
    """Precision and recall over a weighted sample, plus the counts they came from."""
    marked: float     # weighted frames carrying the label
    correct: float    # of those, frames that really hold a live animal
    truth: float      # weighted frames that really hold one, marked or not

    @property
    def precision(self) -> float:
        return self.correct / self.marked if self.marked else 0.0

    @property
    def recall(self) -> float:
        return self.correct / self.truth if self.truth else 0.0


def band_weights(frames: list[Frame], labelled: set[int]) -> dict[int, float]:
    """Weight per LABELLED frame: how many frames of the population it stands for.

    The sample is stratified by score band because the population is top-heavy — a uniform
    sample would spend every label in the bands nobody marks — and a stratified sample only
    answers a question about the collection once each label is weighted back by the size of
    its band. This is the F122 arithmetic, made repeatable instead of done once by hand.

    A band with frames but no labels contributes nothing and is reported by the caller: it
    is a hole in the sample, not a zero.
    """
    weights: dict[int, float] = {}
    for low, high in zip(SCORE_BANDS, SCORE_BANDS[1:]):
        band = [f for f in frames if low <= f.pet_score < high]
        picked = [f for f in band if f.file_id in labelled]
        if not picked:
            continue
        share = len(band) / len(picked)
        for f in picked:
            weights[f.file_id] = share
    return weights


def accuracy(frames: list[Frame], labels: dict[int, bool], threshold: float,
             *, verified: bool) -> Accuracy:
    """Weighted precision/recall of the label rule — with the stored answers, or without.

    `verified=False` replays the rule as it was before the cascade (the CLIP score alone),
    which is what the 92% / 54% of F122 measured; `verified=True` is the same rule with
    `pet_vlm` in hand. Both call `junk.pet_label`, the pipeline's own function, so the
    table cannot come to disagree with what the stage writes.
    """
    weights = band_weights(frames, set(labels))
    marked = correct = truth = 0.0
    for f in frames:
        weight = weights.get(f.file_id)
        if weight is None or f.file_id not in labels:
            continue
        is_animal = labels[f.file_id]
        has_label = junk.pet_label(f.pet_vlm if verified else None,
                                   f.pet_score, threshold) is not None
        marked += weight * has_label
        correct += weight * (has_label and is_animal)
        truth += weight * is_animal
    return Accuracy(marked, correct, truth)


def format_accuracy(frames: list[Frame], labels: dict[int, bool],
                    q: junk.QualitySettings) -> str:
    """The block the feature is accepted or rejected on: before the check, and after it."""
    known = {fid for fid in labels if any(f.file_id == fid for f in frames)}
    if len(known) < MIN_SAMPLE:
        head = (f"ВНИМАНИЕ: размечено {len(known)} кадров выборки, меньше {MIN_SAMPLE} — "
                f"по такой разметке вывод не делают")
    else:
        head = f"размечено {len(known)} кадров выборки"
    rows = [("до проверки (только CLIP)",
             accuracy(frames, labels, q.pet_threshold, verified=False)),
            ("после проверки (CLIP + VLM)",
             accuracy(frames, labels, q.pet_threshold, verified=True))]
    out = [
        "=" * 88,
        f"ТОЧНОСТЬ И ПОЛНОТА (порог {q.pet_threshold:.2f}, взвешено по полосам оценки)",
        f"  {head}",
        f"{'':>30} {'точность':>10} {'полнота':>10}",
    ]
    for name, a in rows:
        out.append(f"{name:>30} {a.precision:>9.1%} {a.recall:>10.1%}")
    out.append("=" * 88)
    out.append("Замер F122 без проверки: точность 92%, полнота 54%. Если проверка полноту "
               "не подняла\nили точность просела — это результат, а не повод двигать "
               "ожидание: порог отбора\nоткатывается, а числа записываются в бриф.")
    return "\n".join(out)


def write_label_template(path: Path, frames: list[Frame], per_band: int,
                         seed: int) -> int:
    """A worksheet to fill in: `{file_id: null}`, stratified over the score bands.

    File ids and nothing else — the same privacy rule the cache follows. Whoever fills it
    in opens those frames in the web app and replaces each null with true (a live animal is
    in the frame) or false. Returns how many frames were written.
    """
    rng = random.Random(seed)
    picked: list[int] = []
    for low, high in zip(SCORE_BANDS, SCORE_BANDS[1:]):
        band = [f.file_id for f in frames if low <= f.pet_score < high]
        rng.shuffle(band)
        picked.extend(sorted(band[:per_band]))
    path.write_text(json.dumps({str(fid): None for fid in picked}, indent=1),
                    encoding="utf-8")
    return len(picked)


def load_labels(path: Path) -> dict[int, bool]:
    """The filled-in worksheet -> {file_id: is there really a live animal}.

    Frames still holding `null` are simply not labelled yet and are dropped — a partially
    filled sheet has to be usable, and treating an unanswered frame as `false` would invent
    the very labels the sheet exists to collect.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    return {int(fid): bool(value) for fid, value in data.items() if value is not None}


def save_cache(path: Path, frames: list[Frame]) -> None:
    """Per-frame aggregates for a later replay. File ids only — never paths."""
    path.write_text(json.dumps({
        "version": CACHE_VERSION,
        "frames": [[f.file_id, f.pet_class, f.pet_score, f.sharpness, f.subject_score,
                    f.face_sharpness]
                   for f in frames],
    }), encoding="utf-8")


def load_cache(path: Path) -> list[Frame]:
    """A cache of another version is an error, not a silently wrong table."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("version") != CACHE_VERSION:
        raise SystemExit(f"{path}: кэш версии {data.get('version')}, ожидается "
                         f"{CACHE_VERSION} — перемерить с --refresh")
    return [Frame(int(fid), pet, float(score),
                  None if sharp is None else float(sharp), float(subject),
                  face_sharpness=None if face is None else float(face))
            for fid, pet, score, sharp, subject, face in data["frames"]]


def read_faces(db_path: str, file_ids: list[int]) -> dict[int, junk.FaceBoxes]:
    """F155: the face boxes of the sample, straight out of `junk.read_face_boxes`.

    Read-only and in a connection of its own — the sample was taken in one that is already
    closed, and this script must never be the thing that writes to a user's index.
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        return junk.read_face_boxes(conn, file_ids)
    finally:
        conn.close()


def measure(cfg: Config, rows: list[sqlite3.Row],
            q: junk.QualitySettings) -> list[Frame]:  # pragma: no cover — ML
    """One CLIP pass (junk prompts + the pet group, as the stage makes it) + the laplacian.

    `clip_prompts(pets=True)` on purpose even when the toggle is off in the config: the
    whole point is to see what the pet group would say before deciding to switch it on.

    F155: the face crop rides in the same call the frame laplacian is taken by — one decode,
    both numbers, exactly as the stage does it, so the sweep prices the real signal.
    """
    s = naming_settings(cfg)
    classifier = clip_classifier(s)
    prompts = junk.clip_prompts(True)
    sharpness = junk.preview_sharpness_detector(q.sharpness_max_edge)
    faces = read_faces(str(cfg.database), [int(r["id"]) for r in rows])

    frames: list[Frame] = []
    done = 0
    for chunk in batched(rows, s.clip_batch_size):
        probs = classifier([r["path"] for r in chunk], prompts)
        for r, p in zip(chunk, probs):
            # threshold 0.0: the winning class always comes back, and which thresholds
            # would have kept it is what the sweep is for.
            pet_class, pet_score = junk.pet_verdict(p, 0.0)
            subject = float(junk._group_probs(p, junk._JUNK_GROUP)[0])
            measured = sharpness(r["path"], faces.get(int(r["id"]), junk.NO_FACES))
            frames.append(Frame(r["id"], pet_class, pet_score, measured.frame, subject,
                                face_sharpness=measured.face))
        done += len(chunk)
        print(f"  CLIP+резкость {done}/{len(rows)}", end="\r", flush=True)
    print(" " * 40, end="\r")
    return frames


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--features", nargs="+", default=list(FEATURES), choices=FEATURES,
                    help="which blocks to print (default: all three)")
    ap.add_argument("--sample", type=int, default=DEFAULT_SAMPLE,
                    help=f"frames to measure (default {DEFAULT_SAMPLE}, floor {MIN_SAMPLE})")
    ap.add_argument("--seed", type=int, default=20260729)
    ap.add_argument("--pet-thresholds", type=float, nargs="+",
                    default=list(DEFAULT_PET_GRID), help="the pet-score grid")
    ap.add_argument("--candidate-thresholds", type=float, nargs="+",
                    default=list(DEFAULT_CANDIDATE_GRID),
                    help="the grid features.pet_candidate_threshold is read off")
    ap.add_argument("--face-thresholds", type=float, nargs="+",
                    default=list(DEFAULT_FACE_GRID),
                    help="the grid features.face_sharpness_max is read off")
    ap.add_argument("--blur-thresholds", type=float, nargs="+",
                    default=list(DEFAULT_BLUR_GRID),
                    help="the grid features.blur_review_max is read off (F157)")
    ap.add_argument("--labels", help="JSON {file_id: true|false} — is there really a live "
                                     "animal in the frame; enables the accuracy block")
    ap.add_argument("--write-labels", help="write a stratified worksheet to fill in "
                                           "(file ids only) and exit")
    ap.add_argument("--per-band", type=int, default=20,
                    help="frames per score band in the worksheet (default 20)")
    ap.add_argument("--cache", help="JSON with the per-frame aggregates: written after a "
                                    "measurement, replayed instead of one")
    ap.add_argument("--refresh", action="store_true",
                    help="measure again even if the cache exists")
    args = ap.parse_args()

    cfg = load_config(args.config)
    q = junk.quality_settings(cfg)
    cache = Path(args.cache) if args.cache else None

    if cache and cache.exists() and not args.refresh:
        frames = load_cache(cache)
        print(f"кэш: {len(frames)} кадров из {cache}")
    else:
        rows = sample_rows(str(cfg.database), args.sample, args.seed)
        if not rows:
            raise SystemExit("нет подходящих файлов в индексе — нечего мерить")
        if len(rows) < MIN_SAMPLE:
            print(f"ВНИМАНИЕ: кадров всего {len(rows)}, меньше {MIN_SAMPLE} — "
                  f"по такой выборке порог не выбирают")
        print(f"выборка: {len(rows)} кадров")
        frames = measure(cfg, rows, q)
        if cache:
            save_cache(cache, frames)
            print(f"кэш записан: {cache} (только file_id и агрегаты, без путей)")

    # F130: the answers live in the index, not in the cache — see read_pet_vlm.
    frames = read_pet_vlm(str(cfg.database), frames)

    if args.write_labels:
        written = write_label_template(
            Path(args.write_labels), frames, args.per_band, args.seed)
        print(f"размечать: {written} кадров записано в {args.write_labels} "
              f"(только file_id; замените null на true/false)")
        return

    print()
    if "pets" in args.features:
        print(format_pets(frames, sweep_pets(frames, sorted(args.pet_thresholds)),
                          q.pet_threshold))
    if "sharpness" in args.features:
        print(format_sharpness(frames, q, sorted(args.blur_thresholds),
                               cfg.features.blur_review_max))
        # F155: the same block, over the face crops — a different population and a
        # different threshold, printed next to the one it is meant to be compared with.
        print(format_face_sharpness(
            frames, sweep_face(frames, sorted(args.face_thresholds)),
            cfg.features.face_sharpness_max))
    if "band" in args.features:
        print(format_band(frames, q))
    if "verify" in args.features:
        print(format_candidates(frames, sorted(args.candidate_thresholds),
                                q.pet_candidate_threshold))
        print(format_answers(frames, q))
        if args.labels:
            print(format_accuracy(frames, load_labels(Path(args.labels)), q))


if __name__ == "__main__":
    main()
