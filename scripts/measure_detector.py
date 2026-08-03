"""Price the F154 cascade BEFORE its threshold and its depth are chosen.

The finding this exists for: an object detector beats what the pipeline already has on
exactly one slice of three (200 hand-labelled frames, 2026-08-02, confidence 0.5):

    ANIMALS   62% precision, 87% recall   the CLIP label: 71% / 33%
    PEOPLE    42% precision, 96% recall   faces (F152): ~100% precision
    FOOD      20% precision, 15% recall   COCO has a banana, not a meal

...and that a full pass costs 83.8 ms x 22 096 = 30.8 minutes. So the detector runs over
CANDIDATES — the top `features.detector_candidates` frames of a zero-shot animal query over
the stored CLIP vectors — and the two numbers that decide what the feature is worth are the
confidence threshold and that depth. NEITHER MAY BE GUESSED. F130 is why the rule is
written down: its brief proposed 0.30 for the animal gate and the measurement made 0.30 the
WORST row of the table, below the CLIP-only baseline it was meant to improve.

What this script prints, and in which order a person reads it:

1. the candidate depth — how many frames each depth selects, what the detector then costs
   at the measured 83.8 ms, and (with labels) what share of the known animals that depth
   still contains. This is the recall ceiling: the detector cannot find an animal the query
   never showed it.
2. the confidence threshold — precision and recall at 0.3 / 0.5 / 0.7 (or any grid),
   against the labelled sample, next to the CLIP label's own numbers on the same frames.

Both tables need a LABELLED SAMPLE — that is the point of the exercise, and there is no way
around it. `--write-sample` produces a worksheet of file ids stratified over the query
ranking; whoever fills it in opens those frames in the web app and replaces each null with
true (there is a real animal in this photograph) or false. Without one, the script still
prints table 1 minus its recall column: the population and the cost are facts of the index.

`--detect` actually runs the detector over the candidates and stores what it finds in
`detections`, exactly as the stage would (one row per frame, boxes above the storage floor)
— so the tables below can be replayed at any threshold afterwards without a second pass.
Nothing else is written: no verdict, no label, no `frame_quality` row.

Nothing is reimplemented here: `detect.rank_candidates`, `detect.animal_boxes`,
`detect.detector_settings` and `junk.read_clip_embeddings` are the pipeline's own. A
private copy of the arithmetic would price a cascade the stage does not have.

Privacy: nothing printed identifies a frame — counts and aggregates only, the rule
measure_ocr_gate.py / measure_junk_rescue.py follow. The worksheet holds file ids alone.

The database is opened `mode=ro` unless `--detect` is given, and a measurement writes
nothing else even then.

Usage (from the repo root, with the venv python):
    python scripts/measure_detector.py --write-sample sample.json --per-band 25
    python scripts/measure_detector.py --detect --labels sample.json
    python scripts/measure_detector.py --labels sample.json --thresholds 0.3,0.5,0.7
"""
from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sorta import detect, junk  # noqa: E402
from sorta.config import Config, load_config  # noqa: E402
from sorta.naming import naming_settings, utcnow_iso  # noqa: E402

# The grid the brief names, plus the ends of the useful range. The chosen row (0.6, F162)
# sits inside it on purpose: a table you can find the configured number in is what makes
# the other rows readable.
DEFAULT_THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)

# The depths worth pricing. 4 000 is the default of `features.detector_candidates` (F162);
# the rest bracket it in both directions — down to where the recall ceiling collapses and
# up to where it stops moving — because what this column shows is a trade of minutes
# against the query's recall ceiling.
DEFAULT_DEPTHS = (500, 1000, 2000, 4000, 10000)

# Seconds per candidate frame, measured on 2026-08-02: 83.8 ms with the mobilenet backbone.
DETECT_SECONDS_PER_FRAME = 0.0838

# The bands a review by eye is stratified over — by RANK in the query, not by score, because
# the ranking is what the depth setting cuts and a score has no absolute meaning (F129).
SAMPLE_BANDS = (0, 100, 500, 1000, 2000, 5000)


@dataclass(frozen=True)
class DepthRow:
    """What one candidate depth selects, what it costs, and what it can still reach."""

    depth: int
    candidates: int
    labelled: int          # labelled frames inside this depth
    animals: int           # of them, the ones a human called an animal
    animals_total: int     # animals in the whole labelled sample

    @property
    def minutes(self) -> float:
        return self.candidates * DETECT_SECONDS_PER_FRAME / 60.0

    @property
    def ceiling(self) -> float | None:
        """The recall the detector cannot exceed at this depth — the query's own."""
        if not self.animals_total:
            return None
        return self.animals / self.animals_total


@dataclass(frozen=True)
class ThresholdRow:
    """Precision and recall of one confidence threshold over the labelled sample."""

    threshold: float
    marked: int            # frames the detector calls an animal
    correct: int           # of them, the ones a human agrees with
    animals_total: int     # animals in the labelled sample (the recall denominator)

    @property
    def precision(self) -> float | None:
        return self.correct / self.marked if self.marked else None

    @property
    def recall(self) -> float | None:
        return self.correct / self.animals_total if self.animals_total else None


def open_db(db_path: str, writable: bool) -> sqlite3.Connection:
    """The index. Read-only unless `--detect` asked to store what the detector finds."""
    if writable:
        from sorta.db import connect

        return connect(db_path)
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def population(conn: sqlite3.Connection) -> dict[int, str]:
    """file_id -> path over the F120 population: canonical photographs, `photo` or unrated.

    The same predicate `junk._DETECTOR_POPULATION_SQL` uses, and it is imported rather than
    retyped for the reason the whole script is built this way: a measurement that selects a
    different population from the stage prices a feature nobody has.
    """
    return {int(r["id"]): str(r["path"]) for r in conn.execute(
        junk._DETECTOR_POPULATION_SQL, (junk.QUALITY_VERDICT,))}


def ranking(cfg: Config, conn: sqlite3.Connection, ids: list[int],
            depth: int) -> list[int]:  # pragma: no cover — ML (the text tower)
    """The candidate list the stage would build, at `depth` — the query, not a threshold."""
    model = junk.embedding_model(naming_settings(cfg))
    vectors = junk.read_clip_embeddings(conn, model, ids)
    if not vectors:
        raise SystemExit(
            "в clip_embeddings нет векторов текущей модели — сначала нужен прогон "
            "junk с features.store_embeddings")
    encoder = junk.clip_text_encoder(naming_settings(cfg))
    features = junk.unit_rows(
        np.asarray(encoder(list(detect.ANIMAL_QUERY_PROMPTS)), dtype=np.float32))
    return detect.rank_candidates(vectors, features, depth)


def stored_boxes(conn: sqlite3.Connection, model: str) -> dict[int, list[detect.Detection]]:
    """What a previous `--detect` (or a live run) left in `detections`, for this detector."""
    return {int(r["file_id"]): detect.unpack_boxes(r["boxes"])
            for r in conn.execute(
                "SELECT file_id, boxes FROM detections WHERE model = ?", (model,))}


def run_detector(conn: sqlite3.Connection, paths: dict[int, str], candidates: list[int],
                 s: detect.DetectorSettings) -> int:  # pragma: no cover — ML
    """Examine the candidates that have no stored answer, and store what was found.

    The stage's own writing rule: a row for every frame examined, including the ones with
    nothing on them (that row is what stops the next run from paying again), boxes above
    the storage floor, the model in the row. Returns how many frames were examined.
    """
    known = stored_boxes(conn, s.model)
    todo = [file_id for file_id in candidates if file_id not in known]
    if not todo:
        return 0
    detector = detect.torchvision_detector(s.model)
    now = utcnow_iso()
    examined = 0
    with conn:
        for file_id in todo:
            try:
                found = detect.animal_boxes(list(detector(paths[file_id])),
                                            detect.STORE_FLOOR)
            except Exception as exc:  # noqa: BLE001 — one frame, not the measurement
                print(f"  кадр file_id={file_id} не разобран ({exc}) — пропускаю")
                continue
            best = detect.best_animal(found, s.threshold)
            conn.execute(junk._DETECTIONS_UPSERT, (
                file_id, None if best is None else best.label,
                None if best is None else float(best.score),
                detect.pack_boxes(found), s.model, now))
            examined += 1
    return examined


def read_labels(path: str | None) -> dict[int, bool]:
    """The worksheet: `{file_id: true|false}`; the nulls nobody filled in are dropped."""
    if not path:
        return {}
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {int(key): bool(value) for key, value in raw.items() if value is not None}


def depth_table(candidates: list[int], labels: dict[int, bool],
                depths: tuple[int, ...]) -> list[DepthRow]:
    """What each depth selects and what it can still reach — the recall CEILING.

    Read this before the threshold table: no confidence setting can find an animal the
    query never put in front of the detector, so this column bounds the one below.
    """
    animals_total = sum(1 for value in labels.values() if value)
    rows = []
    for depth in depths:
        head = candidates[:depth]
        inside = [file_id for file_id in head if file_id in labels]
        rows.append(DepthRow(depth, len(head), len(inside),
                             sum(1 for file_id in inside if labels[file_id]),
                             animals_total))
    return rows


def threshold_table(boxes: dict[int, list[detect.Detection]], labels: dict[int, bool],
                    thresholds: tuple[float, ...]) -> list[ThresholdRow]:
    """Precision and recall per confidence, over the labelled frames the detector saw.

    The rule replayed here is `detect.best_animal`'s own — a frame is an animal when it
    holds an animal box at or above the threshold — so the table cannot disagree with what
    the stage does. The recall denominator is every animal in the SAMPLE, including the
    ones the detector was never shown: that is what makes the number comparable with the
    depth ceiling above.
    """
    animals_total = sum(1 for value in labels.values() if value)
    rows = []
    for threshold in thresholds:
        marked = [file_id for file_id, found in boxes.items()
                  if file_id in labels and detect.best_animal(found, threshold)]
        rows.append(ThresholdRow(threshold, len(marked),
                                 sum(1 for file_id in marked if labels[file_id]),
                                 animals_total))
    return rows


def clip_baseline(conn: sqlite3.Connection, labels: dict[int, bool],
                  threshold: float) -> ThresholdRow:
    """What the label the pipeline writes today scores on the same frames.

    Without it the table has no baseline and every number in it reads as an improvement.
    Read off `frame_quality.pet_score` — the stored score, so this costs no pass — through
    the same `junk.pet_label` rule the stage applies.
    """
    marked = 0
    correct = 0
    for row in junk.read_frame_quality(conn, list(labels)).values():
        if junk.pet_label(row.pet_vlm, row.pet_score, threshold) is None:
            continue
        marked += 1
        correct += bool(labels.get(row.file_id))
    return ThresholdRow(threshold, marked, correct,
                        sum(1 for value in labels.values() if value))


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{100.0 * value:.0f}%"


def format_depths(rows: list[DepthRow], current: int) -> str:
    """Table 1: the depth, its price, and the recall it puts a ceiling on."""
    out = [
        "=" * 92,
        "ГЛУБИНА КАНДИДАТОВ: сколько кадров отбирает запрос, чем это оплачено",
        f"{'глубина':>9} {'кандидатов':>12} {'время':>12} {'размечено':>11} "
        f"{'животных':>10} {'потолок полноты':>17}",
    ]
    for r in rows:
        mark = "*" if r.depth == current else " "
        out.append(f"{r.depth:>8d}{mark}{r.candidates:>12d} {r.minutes:>9.1f} мин "
                   f"{r.labelled:>11d} {r.animals:>10d} {_pct(r.ceiling):>17}")
    out.append("=" * 92)
    out.append(f"* — глубина из конфига (features.detector_candidates). Цена кадра — "
               f"{DETECT_SECONDS_PER_FRAME * 1000:.1f} мс, замер 2026-08-02.")
    out.append("ПОТОЛОК ПОЛНОТЫ — доля размеченных животных, попавших в этот срез "
               "запроса.\nНиже него детектор не опустит, но и выше не поднимется: "
               "кадр, которого он не видел,\nне будет найден ни при каком пороге.")
    return "\n".join(out)


def format_thresholds(rows: list[ThresholdRow], baseline: ThresholdRow | None,
                      current: float) -> str:
    """Table 2: the threshold, chosen from here and not before."""
    out = [
        "=" * 92,
        "ПОРОГ УВЕРЕННОСТИ ДЕТЕКТОРА: точность и полнота на размеченной выборке",
        f"{'порог':>7} {'помечено':>10} {'верно':>8} {'точность':>10} {'полнота':>9}",
    ]
    for r in rows:
        mark = "*" if abs(r.threshold - current) < 1e-9 else " "
        out.append(f"{r.threshold:>6.2f}{mark}{r.marked:>10d} {r.correct:>8d} "
                   f"{_pct(r.precision):>10} {_pct(r.recall):>9}")
    if baseline is not None:
        out.append("-" * 92)
        out.append(f"{'метка CLIP':>7} {baseline.marked:>10d} {baseline.correct:>8d} "
                   f"{_pct(baseline.precision):>10} {_pct(baseline.recall):>9}")
    out.append("=" * 92)
    out.append("* — порог из конфига (features.detector_threshold). Строка «метка CLIP» — "
               "то, что\nпайплайн пишет сегодня на тех же кадрах: без неё любая цифра "
               "выше читается как улучшение.")
    out.append("ВЫБИРАЙТЕ ПО ТАБЛИЦЕ, а не заранее: в F130 предположенные 0.30 оказались "
               "худшей\nстрокой — ниже базовой линии, которую каскад должен был поднять.")
    return "\n".join(out)


def write_sample(path: Path, candidates: list[int], per_band: int, seed: int) -> int:
    """A worksheet to review by eye: `{file_id: null}`, stratified over the query ranking.

    Stratified by RANK and not uniformly: the head of the ranking is where the animals are
    and the tail is what decides the depth, so a uniform sample would spend every look on
    frames no depth will ever select. File ids and nothing else — the privacy rule the
    tables follow.
    """
    rng = random.Random(seed)
    picked: list[int] = []
    for low, high in zip(SAMPLE_BANDS, SAMPLE_BANDS[1:]):
        band = candidates[low:high]
        rng.shuffle(band)
        picked.extend(sorted(band[:per_band]))
    path.write_text(json.dumps({str(file_id): None for file_id in picked}, indent=1),
                    encoding="utf-8")
    return len(picked)


def parse_grid(text: str) -> tuple[float, ...]:
    values = tuple(sorted({float(part) for part in text.replace(",", " ").split()}))
    if not values:
        raise SystemExit("--thresholds: пустая сетка")
    return values


def main() -> int:  # pragma: no cover — the driver; the tables above are what is tested
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--thresholds",
                    default=",".join(f"{t:g}" for t in DEFAULT_THRESHOLDS),
                    help="the confidence grid (default 0.3..0.7)")
    ap.add_argument("--depths", default=",".join(str(d) for d in DEFAULT_DEPTHS),
                    help="the candidate depths to price")
    ap.add_argument("--labels", help="the filled-in worksheet {file_id: true|false}")
    ap.add_argument("--detect", action="store_true",
                    help="run the detector over the candidates and store what it finds "
                         "in `detections` (the only writing mode; needs the weights)")
    ap.add_argument("--write-sample", help="write a worksheet to review by eye and exit")
    ap.add_argument("--per-band", type=int, default=25,
                    help="frames per rank band in the worksheet (default 25)")
    ap.add_argument("--seed", type=int, default=20260803)
    args = ap.parse_args()

    thresholds = parse_grid(args.thresholds)
    depths = tuple(int(v) for v in parse_grid(args.depths))
    cfg = load_config(args.config)
    s = detect.detector_settings(cfg)
    labels = read_labels(args.labels)

    conn = open_db(str(cfg.database), writable=args.detect)
    try:
        paths = population(conn)
        if not paths:
            raise SystemExit("в индексе нет кадров с вердиктом photo — нечего мерить")
        candidates = ranking(cfg, conn, list(paths), max(depths))
        if args.write_sample:
            written = write_sample(Path(args.write_sample), candidates,
                                   args.per_band, args.seed)
            print(f"смотреть глазами: {written} кадров записано в {args.write_sample} "
                  f"(только file_id; замените null на true/false)")
            return 0
        if args.detect:
            head = candidates[:s.candidates]
            print(f"детектор по {len(head)} кандидатам "
                  f"(~{len(head) * DETECT_SECONDS_PER_FRAME / 60.0:.1f} мин)...")
            print(f"разобрано кадров: {run_detector(conn, paths, head, s)}")
        boxes = stored_boxes(conn, s.model)
        baseline = clip_baseline(conn, labels, float(cfg.features.pet_threshold)) \
            if labels else None
    finally:
        conn.close()

    print(f"коллекция: {len(paths)} фотографий, кандидатов ранжировано "
          f"{len(candidates)}, размечено {len(labels)}, "
          f"кадров с боксами {len(boxes)}")
    print()
    print(format_depths(depth_table(candidates, labels, depths), s.candidates))
    if not labels:
        print("\nБЕЗ РАЗМЕТКИ таблица порогов не печатается: точность и полноту не из "
              "чего считать.\nСделайте выборку (--write-sample), разметьте её и "
              "запустите снова с --labels.")
        return 0
    if not boxes:
        print("\nБОКСОВ НЕТ: сначала прогон детектора — `--detect` (или обычный прогон "
              "junk\nс features.detector). Пороги считаются по сохранённым боксам, "
              "второй проход не нужен.")
        return 0
    print(format_thresholds(threshold_table(boxes, labels, thresholds), baseline,
                            s.threshold))
    return 0


if __name__ == "__main__":
    sys.exit(main())
