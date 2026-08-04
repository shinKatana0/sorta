"""F163 phase 0: price "not your photograph" — the downloaded and the generated — first.

The finding this exists for (2026-08-03). The bucket the owner filled with "screenshot /
receipt / screen" turned out to hold one real screenshot in twenty-two; what it actually
holds is a class the product does not have. Of the thirteen frames the owner looked at by
eye, eleven were downloaded wallpapers and generated pictures — files nobody photographed.
There is no `media_class.verdict` for that: `product`, `screenshot`, `document` and `meme`
are all about something else, and ~570 frames of the archive are laid out by city as if
somebody had stood there and pressed a shutter.

WHAT THIS SCRIPT IS NOT. It is not the feature and it writes nothing. Phase 1 gets its own
brief once the tables below hold numbers, and the brief already says what it may not do:
no new verdict — a verdict moves files, and the accuracy of this class is exactly what is
unknown.

The metadata signal is REAL AND WEAK, and that is already measured, so nobody has to
rediscover it. On the thirteen frames "no camera EXIF" separated them 13 of 13 — on a
sample selected by that same feature. On 500 honestly labelled frames it gives 31% precision
at 55% recall, and no size threshold saves it, because messengers strip EXIF and a real
photograph a friend sent is indistinguishable from a downloaded picture by metadata alone.
That is why metadata enters these tables as a CANDIDATE SELECTOR (table 3) and as the
baseline to beat (table 4), never as the answer.

What the script prints, and in which order a person reads it:

1. POPULATION — how many canonical frames have no camera EXIF, how many are small, and
   what today's classes say about those slices. Part of the class may already be caught as
   `meme` or `screenshot`, and then the feature is smaller than it looks.
2. WORDINGS AND DEPTH — precision and recall of a zero-shot query over the vectors F128
   already stores, one row per (wording, depth). Several wordings, each measured on its
   own: an ENSEMBLE OF WORDINGS IS CLOSED — measured, no effect, do not reopen. The depths
   double, because depth is the one confirmed lever of recall (+25 p.p. per doubling).
3. METADATA AS A SELECTOR — the same rows with the candidates narrowed to frames with no
   camera EXIF, against the same rows over the whole population. The question is whether
   narrowing concentrates the class or only loses it.
4. BASELINE — what today's classes and the metadata rules alone score on the very same
   frames. Without it every number above reads as an improvement.

NO THRESHOLD AND NO DEPTH IS CHOSEN HERE, and none may be proposed before the table is
read: F130 put 0.30 in a brief and the measurement made it the worst row of its own table.

THE SAMPLE, AND WHY IT IS WEIGHTED. `--write-sample` writes a worksheet of file ids
stratified over RANK BANDS of the pooled ranking — pooled over all wordings (the best rank
a frame gets from any of them), so the sample is not selected by the one wording it would
then flatter. That is the error the brief names four times over for a single day. Bands are
sampled at very different rates, so every number below is a WEIGHTED estimate: a labelled
frame stands for `band size / labelled in that band` frames of the ranking. Unweighted
counts are printed next to each estimate, so the arithmetic stays visible.

Nothing is reimplemented: `detect.rank_candidates`,
`junk.read_clip_embeddings`, `junk.embedding_model` and `junk.unit_rows` are the pipeline's
own. A private copy of the arithmetic would price a query the stage cannot run.

Privacy: nothing printed identifies a frame — counts and aggregates only, the rule
measure_junk_rescue.py / measure_detector.py follow. The worksheet holds file ids alone.

The database is opened `mode=ro`: a measurement writes nothing, in any mode.

Usage (from the repo root, with the venv python):
    python scripts/measure_downloaded.py --population
    python scripts/measure_downloaded.py --write-sample sample.json --per-band 25
    python scripts/measure_downloaded.py --labels sample.json
"""
from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sorta import detect, junk  # noqa: E402
from sorta.config import Config, load_config  # noqa: E402
from sorta.naming import naming_settings  # noqa: E402


@dataclass(frozen=True)
class Wording:
    """One way of asking for the class: a name for the table and the prompt itself.

    ONE PROMPT PER WORDING, never a group of them. Combining wordings into an ensemble was
    measured and gave nothing, and the point of this table is which single phrasing the
    model answers best — an ensemble would hide exactly that.
    """

    name: str
    prompt: str


# The four wordings the brief names, in English because the vectors of `clip_embeddings`
# come from the CLASSIFICATION model (`ViT-L-14`), which answers English and does not
# answer Russian — 22% precision at top-5 against 98% for the search model (F141). The
# names are what the table prints; the prompts are what the text tower sees.
WORDINGS: tuple[Wording, ...] = (
    Wording("обои", "a wallpaper image downloaded from the internet"),
    Wording("сгенерированное", "an ai-generated image"),
    Wording("рисунок", "a digital drawing or an illustration"),
    Wording("скачанное", "a picture downloaded from the internet, not taken with a camera"),
)

# The depths priced, each one double the last: depth is the only confirmed lever of recall
# (+25 p.p. per doubling, measured three times), so the rows have to be readable as pairs.
# They bracket the estimated population of the class (~570 frames of the archive) from both
# sides — a grid, not a proposal: no depth is chosen in this script.
DEFAULT_DEPTHS = (250, 500, 1000, 2000)

# The rank bands the worksheet is stratified over — by RANK and not by score, because a
# score has no absolute meaning (F129) and the depth above is what cuts the ranking. The
# last band is opened up to the size of the ranking at runtime (see `rank_bands`): without
# a sampled tail the recall denominator would only hold frames the query already liked.
SAMPLE_BANDS = (0, 100, 250, 500, 1000, 2000, 5000)

# The sizes a "small file" is asked about, in megapixels. Straight from the brief's own
# table, so the rows here can be compared with the 500 frames labelled before this script
# existed.
SMALL_MEGAPIXELS = (1.0, 3.0, 5.0)


@dataclass(frozen=True)
class Frame:
    """One canonical frame, as the two questions of this measurement see it.

    `verdict` is what `media_class` says today (None — the stage never classified it),
    `megapixels` is None when the index has no size for the frame: not asked is not the
    same fact as small, and a rule about small files must not silently claim the unknown
    ones.
    """

    file_id: int
    verdict: str | None
    has_camera: bool
    megapixels: float | None

    def smaller_than(self, limit: float) -> bool:
        """True when the frame is KNOWN to be smaller than `limit` megapixels."""
        return self.megapixels is not None and self.megapixels < limit


@dataclass(frozen=True)
class SliceRow:
    """One slice of the population and what today's classes call the frames in it."""

    name: str
    frames: int
    total: int
    verdicts: Counter[str]

    @property
    def share(self) -> float:
        return self.frames / self.total if self.total else 0.0


@dataclass(frozen=True)
class Estimate:
    """What some selection holds, counted twice: by hand and by the sampling design.

    `labelled` and `correct` are frames of the worksheet — what a person could recount.
    `marked_w` and `correct_w` are the same two numbers weighted by how many frames of the
    ranking each labelled frame stands for, which is what makes precision and recall
    statements about the COLLECTION rather than about the worksheet. `population_w` is the
    weighted count of the whole class and is the recall denominator everywhere, including
    the narrowed pools of table 3 — a selector that loses frames has to show it.
    """

    labelled: int
    correct: int
    marked_w: float
    correct_w: float
    population_w: float

    @property
    def precision(self) -> float | None:
        return self.correct_w / self.marked_w if self.marked_w else None

    @property
    def recall(self) -> float | None:
        return self.correct_w / self.population_w if self.population_w else None


@dataclass(frozen=True)
class QueryRow:
    """One (wording, depth) cell of the query tables, plus the size of what it cut."""

    wording: str
    depth: int
    candidates: int
    estimate: Estimate


@dataclass(frozen=True)
class RuleRow:
    """One baseline rule — today's classes, or a metadata rule — over the same frames."""

    name: str
    marked: int
    estimate: Estimate


def is_reclassified(frame: Frame) -> bool:
    """What the pipeline already calls something other than a photograph."""
    return frame.verdict is not None and frame.verdict != junk.QUALITY_VERDICT


def small(limit: float) -> Callable[[Frame], bool]:
    """A predicate for `smaller than `limit` megapixels`.

    A factory rather than a lambda written inside a comprehension: the loop variable of a
    comprehension is captured by reference, which is how a table of thresholds silently
    becomes several copies of the last one.
    """
    return lambda frame: frame.smaller_than(limit)


def small_without_camera(limit: float) -> Callable[[Frame], bool]:
    """The brief's own rule: no camera in EXIF AND smaller than `limit` megapixels."""
    return lambda frame: not frame.has_camera and frame.smaller_than(limit)


def no_camera(frame: Frame) -> bool:
    return not frame.has_camera


# The baseline. The first row is what the pipeline says today; the rest are the metadata
# rules of the brief, repeated here so the new sample can be compared with the 500 frames
# labelled before. A rule about size never fires on a frame whose size is unknown.
RULES: tuple[tuple[str, Callable[[Frame], bool]], ...] = (
    ("нынешние классы (не «photo»)", is_reclassified),
    ("нет камеры", no_camera),
    *tuple((f"нет камеры и меньше {limit:g} Мп", small_without_camera(limit))
           for limit in SMALL_MEGAPIXELS),
)


def open_ro(db_path: str) -> sqlite3.Connection:
    """The index, read-only — a measurement writes nothing, in any mode."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def has_camera_exif(make: object, model: object) -> bool:
    """True when EXIF names a camera at all — either field, non-blank.

    Either field, because the two are filled independently by the phones and the cameras
    of the collection, and a frame that names one of them was taken by something. A string
    of spaces is not a camera: exiftool hands those back for tags that exist and are empty.
    """
    return any(str(value).strip() for value in (make, model) if value is not None)


def megapixels(width: object, height: object) -> float | None:
    """Frame size in megapixels, or None when the index does not know it.

    None rather than 0.0 — the same rule `frame_quality` follows for every nullable column
    of the schema: a frame nobody measured must not read as the smallest one in the
    collection, which is precisely the band the rules below fire in.
    """
    try:
        pixels = int(width) * int(height)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return pixels / 1e6 if pixels > 0 else None


def read_frames(conn: sqlite3.Connection) -> list[Frame]:
    """The canonical photographs of the index with the two facts this measurement needs.

    The F120 population minus its verdict filter: the ranking below can only reach frames
    the stage called `photo` (`clip_embeddings` holds those alone), but table 1 is the
    question of how many of the class are ALREADY caught as something else, and a table
    that dropped the other verdicts could not answer it.
    """
    return [
        Frame(file_id=int(r["id"]),
              verdict=None if r["verdict"] is None else str(r["verdict"]),
              has_camera=has_camera_exif(r["camera_make"], r["camera_model"]),
              megapixels=megapixels(r["width"], r["height"]))
        for r in conn.execute(
            """SELECT f.id, f.camera_make, f.camera_model, f.width, f.height,
                      mc.verdict AS verdict
               FROM files f LEFT JOIN media_class mc ON mc.file_id = f.id
               WHERE f.dup_of IS NULL AND f.error IS NULL AND f.media_type = 'photo'
               ORDER BY f.id""")
    ]


def ranked_ids(frames: list[Frame]) -> list[int]:
    """The frames a query can reach: the ones the stage calls a photograph.

    `clip_embeddings` is purged of every other verdict on each run, so this is also the
    only population that has vectors — and, being the population, it is why the baseline
    row "today's classes" scores zero by construction. That is not a bug in the baseline;
    it is the shape of the problem, and the table says so in those words.
    """
    return [f.file_id for f in frames if f.verdict == junk.QUALITY_VERDICT]


def population_table(frames: list[Frame]) -> list[SliceRow]:
    """Table 1: the metadata slices against what the classifier says about them today."""
    slices: list[tuple[str, Callable[[Frame], bool]]] = [
        ("вся коллекция", lambda frame: True),
        ("нет камеры", no_camera),
    ]
    slices.extend((f"меньше {limit:g} Мп", small(limit)) for limit in SMALL_MEGAPIXELS)
    slices.extend((f"нет камеры и меньше {limit:g} Мп", small_without_camera(limit))
                  for limit in SMALL_MEGAPIXELS)
    rows = []
    for name, predicate in slices:
        inside = [f for f in frames if predicate(f)]
        rows.append(SliceRow(
            name=name, frames=len(inside), total=len(frames),
            verdicts=Counter("нет класса" if f.verdict is None else f.verdict
                             for f in inside)))
    return rows


def rank_bands(total: int) -> list[tuple[int, int]]:
    """The rank bands of the sampling design, with the last one opened up to `total`.

    The declared edges stop at 5 000 because that is where a query's head ends; the tail
    is one band of its own, however long it is. Without it the worksheet would hold no
    frame the query dislikes, and a recall computed over such a sample cannot be wrong —
    which is the same trap as the thirteen frames of the brief.
    """
    edges = [edge for edge in SAMPLE_BANDS if edge < total] or [0]
    if edges[-1] < total:
        edges.append(total)
    return [(low, high) for low, high in zip(edges, edges[1:]) if high > low]


def band_of_rank(rank: int, bands: list[tuple[int, int]]) -> tuple[int, int] | None:
    """The band a rank falls into; None when it is outside the ranking entirely."""
    for low, high in bands:
        if low <= rank < high:
            return low, high
    return None


def design_bands(ranks: dict[int, int]) -> list[tuple[int, int]]:
    """The bands of the sampling design — as many as the ranked population supports."""
    return rank_bands(len(ranks))


def band_sizes(ranks: dict[int, int]) -> Counter[tuple[int, int]]:
    """How many frames sit in each band — COUNTED, never taken as the width of the band.

    A pooled rank is not a permutation: two frames can both be first, each for a different
    wording, so several frames share a rank and the tail bands hold fewer frames than their
    edges suggest. The weights below divide by this count, so the estimator stays right
    whatever shape the pooling gives the design.
    """
    bands = design_bands(ranks)
    return Counter(band for rank in ranks.values()
                   if (band := band_of_rank(rank, bands)) is not None)


def pooled_ranks(rankings: dict[str, list[int]]) -> dict[int, int]:
    """file_id -> its BEST rank over all wordings — the design the worksheet is drawn on.

    Pooled and not taken from one wording, because a sample stratified by one query's
    ranking measures that query on its own favourites. The brief records four measurements
    ruined by exactly that in a single day; this is the cheapest available guard against
    the fifth.
    """
    best: dict[int, int] = {}
    for ranking in rankings.values():
        for rank, file_id in enumerate(ranking):
            if rank < best.get(file_id, len(ranking) + 1):
                best[file_id] = rank
    return best


def write_sample(path: Path, ranks: dict[int, int], per_band: int, seed: int) -> int:
    """The worksheet: `{file_id: null}`, `per_band` frames from each band of the design.

    Whoever fills it in opens those frames in the web app and replaces each null with true
    (nobody photographed this — it was downloaded, generated or drawn) or false (a real
    photograph, whatever its metadata says). File ids and nothing else: the privacy rule
    the tables follow.
    """
    rng = random.Random(seed)
    picked: list[int] = []
    for low, high in design_bands(ranks):
        band = [file_id for file_id, rank in ranks.items() if low <= rank < high]
        rng.shuffle(band)
        picked.extend(sorted(band[:per_band]))
    path.write_text(json.dumps({str(file_id): None for file_id in sorted(picked)}, indent=1),
                    encoding="utf-8")
    return len(picked)


def read_labels(path: str | None) -> dict[int, bool]:
    """The filled-in worksheet: `{file_id: true|false}`; unanswered nulls are dropped.

    Dropped rather than read as false: an unanswered frame is a frame nobody looked at,
    and counting it as "a real photograph" would invent precisely the marks the worksheet
    exists to collect.
    """
    if not path:
        return {}
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {int(key): bool(value) for key, value in raw.items() if value is not None}


def band_weights(ranks: dict[int, int], labels: dict[int, bool]) -> dict[int, float]:
    """file_id -> how many frames of the ranking this labelled frame stands for.

    The bands are sampled at wildly different rates (25 frames out of the first 100 and 25
    out of the last fifteen thousand), so a plain average over the worksheet is an average
    over the head of the ranking and nothing else. The weight is the size of the band
    divided by how many of its frames were actually labelled — the estimator that turns
    the worksheet back into a statement about the collection.

    A labelled frame outside the ranking (no stored vector, or reclassified since the
    worksheet was written) gets no weight and takes part in nothing: it is reported as
    missing instead of being folded in at some invented rate.
    """
    bands = design_bands(ranks)
    sizes = band_sizes(ranks)
    inside = {file_id: band for file_id, rank in ranks.items()
              if file_id in labels and (band := band_of_rank(rank, bands)) is not None}
    labelled_per_band = Counter(inside.values())
    return {file_id: sizes[band] / labelled_per_band[band]
            for file_id, band in inside.items()}


def estimate(selected: list[int], labels: dict[int, bool],
             weights: dict[int, float]) -> Estimate:
    """What one selection of frames holds, by hand and by the sampling design."""
    inside = [file_id for file_id in selected if file_id in weights]
    correct = [file_id for file_id in inside if labels[file_id]]
    return Estimate(
        labelled=len(inside),
        correct=len(correct),
        marked_w=sum(weights[file_id] for file_id in inside),
        correct_w=sum(weights[file_id] for file_id in correct),
        population_w=sum(weight for file_id, weight in weights.items() if labels[file_id]),
    )


def query_rows(rankings: dict[str, list[int]], labels: dict[int, bool],
               weights: dict[int, float], depths: tuple[int, ...]) -> list[QueryRow]:
    """Tables 2 and 3: one row per (wording, depth) over whichever pool was ranked."""
    rows = []
    for wording in WORDINGS:
        ranking = rankings.get(wording.name, [])
        for depth in depths:
            head = ranking[:depth]
            rows.append(QueryRow(wording=wording.name, depth=depth, candidates=len(head),
                                 estimate=estimate(head, labels, weights)))
    return rows


def rule_rows(frames: list[Frame], labels: dict[int, bool],
              weights: dict[int, float]) -> list[RuleRow]:
    """Table 4: the baseline rules, over the labelled frames and the same weights."""
    known = {f.file_id: f for f in frames}
    rows = []
    for name, predicate in RULES:
        marked = [file_id for file_id in weights
                  if file_id in known and predicate(known[file_id])]
        rows.append(RuleRow(name=name, marked=len(marked),
                            estimate=estimate(marked, labels, weights)))
    return rows


def population_estimate(weights: dict[int, float], labels: dict[int, bool]) -> float:
    """The weighted size of the class in the ranked population — the brief's ~570."""
    return sum(weight for file_id, weight in weights.items() if labels[file_id])


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{100.0 * value:.0f}%"


def format_population(rows: list[SliceRow]) -> str:
    """Table 1: how big each metadata slice is and what the classifier calls it today."""
    out = [
        "=" * 92,
        "НАСЕЛЕНИЕ: метаданные против нынешних классов (канонические кадры)",
        f"{'срез':<28} {'кадров':>9} {'доля':>7}   классы",
    ]
    for r in rows:
        classes = ", ".join(f"{verdict}: {count}"
                            for verdict, count in sorted(r.verdicts.items(),
                                                         key=lambda kv: (-kv[1], kv[0])))
        out.append(f"{r.name:<28} {r.frames:>9d} {_pct(r.share):>7}   {classes or '—'}")
    out.append("=" * 92)
    out.append("ЧИТАТЬ ПЕРВОЙ: часть класса уже ловится как «screenshot» или «meme», и на "
               "столько же\nменьше фича. Кадр без размера в индексе не считается мелким — "
               "«не измерено» это не «мало».")
    return "\n".join(out)


def format_query(rows: list[QueryRow], title: str) -> str:
    """Tables 2 and 3: precision and recall per wording and depth, weighted estimates."""
    out = [
        "=" * 92,
        title,
        f"{'формулировка':<18} {'глубина':>8} {'кандидатов':>11} {'размечено':>10} "
        f"{'верных':>7} {'точность':>9} {'полнота':>8}",
    ]
    for r in rows:
        e = r.estimate
        out.append(f"{r.wording:<18} {r.depth:>8d} {r.candidates:>11d} {e.labelled:>10d} "
                   f"{e.correct:>7d} {_pct(e.precision):>9} {_pct(e.recall):>8}")
    out.append("=" * 92)
    out.append("Каждая формулировка меряется отдельно: АНСАМБЛЬ ФОРМУЛИРОВОК ЗАКРЫТ "
               "замером — эффекта\nнет, не открывать. Глубина идёт удвоениями и читается "
               "парами строк: это единственный\nподтверждённый рычаг полноты. Точность и "
               "полнота — оценки по взвешенной выборке,\nрядом с ними стоят сырые "
               "счётчики, из которых они посчитаны.")
    return "\n".join(out)


def format_gate(gated: list[QueryRow], full: list[QueryRow]) -> str:
    """Table 3's own question: does "no camera" concentrate the class or only lose it?"""
    by_key = {(r.wording, r.depth): r for r in full}
    out = [
        "=" * 92,
        "МЕТАДАННЫЕ КАК ОТБОРЩИК: кандидаты «нет камеры» против всей популяции",
        f"{'формулировка':<18} {'глубина':>8} {'точность':>9} {'полнота':>8}   "
        f"{'точность всюду':>15} {'полнота всюду':>14}",
    ]
    for r in gated:
        other = by_key.get((r.wording, r.depth))
        out.append(
            f"{r.wording:<18} {r.depth:>8d} {_pct(r.estimate.precision):>9} "
            f"{_pct(r.estimate.recall):>8}   "
            f"{_pct(other.estimate.precision) if other else '—':>15} "
            f"{_pct(other.estimate.recall) if other else '—':>14}")
    out.append("=" * 92)
    out.append("Полнота в обоих случаях считается от ВСЕГО класса, а не от суженной "
               "популяции:\nотборщик, который теряет кадры, обязан это показать. Метаданные "
               "на честной выборке —\n31–41% точности: годятся отбирать кандидатов, не "
               "годятся выносить вердикт.")
    return "\n".join(out)


def format_baseline(rows: list[RuleRow], population: float) -> str:
    """Table 4: what today's classes and the metadata rules score on the same frames."""
    out = [
        "=" * 92,
        "БАЗОВАЯ ЛИНИЯ: нынешние классы и правила по метаданным на тех же кадрах",
        f"{'правило':<30} {'помечено':>10} {'размечено':>10} {'верных':>7} "
        f"{'точность':>9} {'полнота':>8}",
    ]
    for r in rows:
        e = r.estimate
        out.append(f"{r.name:<30} {r.marked:>10d} {e.labelled:>10d} {e.correct:>7d} "
                   f"{_pct(e.precision):>9} {_pct(e.recall):>8}")
    out.append("-" * 92)
    out.append(f"оценка населения класса: ~{population:.0f} кадров ранжированной популяции "
               f"(взвешенная оценка)")
    out.append("=" * 92)
    out.append("Строка нынешних классов равна нулю ПО ПОСТРОЕНИЮ: ранжируются только кадры, "
               "которые\nклассификатор уже назвал фотографиями. Это и есть форма задачи — "
               "сколько ловится\nв других срезах, показывает таблица 1.")
    return "\n".join(out)


def text_features(cfg: Config) -> dict[str, np.ndarray]:  # pragma: no cover — ML
    """Each wording through the project's own text tower, as one unit row apiece.

    One encoder call for all of them and then a row each — the wordings are compared, never
    combined: an ensemble was measured, gave nothing, and would hide which phrasing works.
    """
    encoder = junk.clip_text_encoder(naming_settings(cfg))
    encoded = junk.unit_rows(np.asarray(encoder([w.prompt for w in WORDINGS]),
                                        dtype=np.float32))
    return {w.name: encoded[i:i + 1] for i, w in enumerate(WORDINGS)}


def rankings_of(vectors: dict[int, np.ndarray], features: dict[str, np.ndarray],
                depth: int) -> dict[str, list[int]]:
    """One ranking per wording, best first — the stage's own `detect.rank_candidates`.

    Its tie-breaking (by file_id) is what makes a repeated run select the same frames, and
    a worksheet drawn on a ranking that reshuffles between runs would measure nothing.
    """
    return {name: detect.rank_candidates(vectors, rows, depth)
            for name, rows in features.items()}


def main() -> int:  # pragma: no cover — the driver; the tables above are what is tested
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--labels", help="the filled-in worksheet {file_id: true|false}")
    ap.add_argument("--write-sample", help="write a worksheet to review by eye and exit")
    ap.add_argument("--per-band", type=int, default=25,
                    help="frames per rank band in the worksheet (default 25 — seven bands, "
                         "175 frames, the 150-200 the brief asks for)")
    ap.add_argument("--population", action="store_true",
                    help="print table 1 and stop — the only mode that needs no model")
    ap.add_argument("--depths", default=",".join(str(d) for d in DEFAULT_DEPTHS),
                    help="the candidate depths to price (each double the last)")
    ap.add_argument("--seed", type=int, default=20260803)
    args = ap.parse_args()

    depths = tuple(sorted({int(part) for part in args.depths.replace(",", " ").split()}))
    if not depths:
        raise SystemExit("--depths: пустая сетка")
    cfg = load_config(args.config)
    labels = read_labels(args.labels)

    conn = open_ro(str(cfg.database))
    try:
        frames = read_frames(conn)
        if not frames:
            raise SystemExit("в индексе нет канонических фотографий — нечего мерить")
        print(f"коллекция: {len(frames)} кадров, размечено {len(labels)}")
        print()
        print(format_population(population_table(frames)))
        if args.population:
            return 0
        ids = ranked_ids(frames)
        vectors = junk.read_clip_embeddings(
            conn, junk.embedding_model(naming_settings(cfg)), ids)
    finally:
        conn.close()

    if not vectors:
        raise SystemExit(
            "нет сохранённых CLIP-векторов для этих кадров — сначала нужен прогон junk "
            "с features.store_embeddings")

    features = text_features(cfg)
    # Ranked to the full depth of the population once: the tables take their own heads of
    # it, and the pooled ranks the worksheet is drawn on need the tail as well.
    rankings = rankings_of(vectors, features, len(vectors))
    ranks = pooled_ranks(rankings)

    if args.write_sample:
        written = write_sample(Path(args.write_sample), ranks, args.per_band, args.seed)
        print(f"\nсмотреть глазами: {written} кадров записано в {args.write_sample} "
              f"(только file_id; замените null на true/false)")
        return 0
    if not labels:
        print("\nБЕЗ РАЗМЕТКИ таблицы точности не печатаются: считать их не из чего.\n"
              "Сделайте выборку (--write-sample), разметьте её и запустите с --labels.")
        return 0

    weights = band_weights(ranks, labels)
    missing = len(labels) - len(weights)
    without_camera = {f.file_id for f in frames if not f.has_camera}
    gated = rankings_of({file_id: vec for file_id, vec in vectors.items()
                         if file_id in without_camera}, features, len(vectors))
    rows = query_rows(rankings, labels, weights, depths)

    print()
    print(f"ранжировано {len(vectors)} кадров с вектором, из них без EXIF камеры "
          f"{len(vectors.keys() & without_camera)}; размеченных в ранжировании "
          f"{len(weights)}" + (f", вне его {missing}" if missing else ""))
    print()
    print(format_query(rows,
                       "ФОРМУЛИРОВКИ И ГЛУБИНА: точность и полнота запроса по векторам"))
    print(format_gate(query_rows(gated, labels, weights, depths), rows))
    print(format_baseline(rule_rows(frames, labels, weights),
                          population_estimate(weights, labels)))
    print("ВЫБИРАЙТЕ ПО ТАБЛИЦЕ, а не заранее: в F130 предложенное в брифе значение "
          "оказалось\nхудшей строкой собственной таблицы. Фаза 1 начинается со СРЕЗА, "
          "который смотрит человек,\nа не с нового вердикта: вердикт двигает файлы.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
