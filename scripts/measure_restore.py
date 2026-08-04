"""F169, phase 0 — what "try to improve" really does, on three frame populations.

The button of F149 scales every frame to one ceiling before the model
(`features.restore_max_edge`, 1024 px until this feature made it a setting), and the
model is x4. For a SMALL frame that is a clean gain: 800 px in, 3200 px out, nothing
given up. For a full-sized one it is a trade nobody was told about —

    4032 x 3024 (12 Mpx)  ->  1024 x 768  ->  4096 x 3072

— the same size out, through a quarter and back, with the real detail of the original
dropped on the way in and plausible detail drawn in its place.

WHAT MAY BE DECIDED FROM THIS SCRIPT'S OUTPUT, and what may not. It prints what the
action costs and what it produces; it does NOT print whether the result is better. That
judgement is a person's, on the blind pairs it lays out, and the reason is written into
F149's own history: the first probe used `swin2SR-classical-sr`, a model trained on clean
bicubic downscaling, and its numbers flattered a result a human eye then rejected. A
broken instrument flatters the outcome, so nothing here reduces "better" to a metric.

What it prints:

1. THE THREE POPULATIONS SEPARATELY — frames under 1024 px, 1024-2500 px, over 2500 px.
   They ask different questions, and one average over the three would answer none of
   them (the mistake of one bucket for "a screenshot" and "the screenshot bin").
2. PER POPULATION, WHAT BECAME OF THE FRAME: the size in and out, the weight, the time,
   the peak VRAM — and the share of the original's own pixels the model was even shown.
3. THE BASELINE IS THE ORIGINAL ITSELF, printed as the first row of every table. Without
   it "it got sharper" has nothing to be sharper than.
4. WHAT RAISING THE CEILING BUYS: the same rows at 2048 and at the full frame, so the
   memory wall is a measured number here rather than an assumption (the x4 output of a
   4000 px frame is ~780 Mpx, which is expected to fail — a failure is a row, not a
   crash).
5. BLIND PAIRS FOR THE EYES: two pictures at the same size, side by side, nothing saying
   which is which, per frame and per ceiling. The whole frame (`pair_NN.jpg`) and — where
   the copy came back at the size of its original, which is the population in question —
   the same middle of both at native scale (`pair_NN_crop.jpg`), because a 12 Mpx frame
   shrunk to fit a sheet is a frame whose lost detail cannot be seen. The mapping goes
   into `key.json`, which is meant to be opened AFTER looking.

Privacy: no path, no basename and no thumbnail of a frame is printed to the console; the
sheets are numbered (`pair_01.jpg`), and the key holds file ids alone — the rule
`scripts/measure_ocr_gate.py` and `scripts/measure_detector.py` follow.

The originals are opened read-only and never written: this script measures the action,
it does not perform it (nothing is saved beside anybody's photograph).

Usage (from the repo root, with a GPU venv — `uv sync --extra gpu --extra vlm`):
    python scripts/measure_restore.py                        # 6 frames per population
    python scripts/measure_restore.py --sample 10 --out measure_restore
    python scripts/measure_restore.py --edges 1024 2048 0    # 0 — the full frame
"""
from __future__ import annotations

import argparse
import io
import json
import random
import sqlite3
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))  # the repo root — for `sorta`
sys.path.insert(0, str(_HERE))         # ...and this directory — for the sibling script

from sorta import imaging, restore  # noqa: E402
from sorta.config import load_config  # noqa: E402

# The peak-memory reader of the speed measurements, imported rather than copied: a second
# version of it would report a different number for the same run (reserved vs allocated),
# and the two reports have to stay comparable.
from measure_vlm_speed import _vram_peak_mb as vram_peak_mb  # noqa: E402

# The three populations, split at the shipped ceiling and at the size above which a frame
# is a full camera shot. Under 1024 the ceiling never fires at all — that is the case the
# action was built for and the case that must not be broken by anything decided here.
SMALL, MID, BIG = "small", "mid", "big"
POPULATIONS = (SMALL, MID, BIG)
SMALL_MAX = 1024
MID_MAX = 2500
POPULATION_LABEL = {
    SMALL: "мелкие (< 1024 px)",
    MID: "средние (1024-2500 px)",
    BIG: "полноразмерные (> 2500 px)",
}

# The ceilings compared. The first is what ships, so "nothing changes" stays a possible
# outcome of the run; 0 means "no ceiling at all", the full frame, which is the variant
# the brief expects to fail on memory and wants measured rather than assumed.
DEFAULT_EDGES = (1024, 2048, 0)
FULL_FRAME = 0
DEFAULT_SAMPLE = 6

# Both halves of a blind pair are drawn at the SAME size on purpose: a bigger picture
# gives itself away, and then the comparison measures the layout instead of the pixels.
SHEET_EDGE = 900
# The 1:1 window into a pair that came back at the size it went in — native scale, no
# resampling of either half, because a resample is what hides the difference.
CROP_BOX = 700
SHEET_GAP = 16
SHEET_BACKGROUND = (24, 24, 24)
ORIGINAL, PROCESSED = "оригинал", "обработано"


@dataclass(frozen=True)
class FrameRun:
    """One frame through one ceiling — or, with `max_edge=None`, the frame as it lies.

    `error` is a row and not an exception: "the full frame did not fit into memory" is
    one of the answers this script exists to produce, and a traceback would end the run
    before the populations that DO fit were measured.
    """
    file_id: int
    population: str
    max_edge: int | None
    source_size: tuple[int, int]
    input_size: tuple[int, int]
    output_size: tuple[int, int]
    weight_bytes: int
    seconds: float
    peak_vram_mb: float | None = None
    error: str | None = None

    @property
    def source_edge(self) -> int:
        return max(self.source_size) if self.source_size else 0

    @property
    def is_baseline(self) -> bool:
        return self.max_edge is None


def population_of(edge: int) -> str:
    """Which of the three bands a frame with this longer side belongs to."""
    if edge <= SMALL_MAX:
        return SMALL
    if edge <= MID_MAX:
        return MID
    return BIG


def truth_kept(source_edge: int, max_edge: int | None) -> float:
    """The share of the original's OWN pixels the model was shown (1.0 — all of them).

    Areal, not linear: the ceiling cuts each side, so a frame halved on the way in hands
    the model a quarter of what was really there. This is the one number here that needs
    no eye — it is arithmetic over sizes, and it says how much of the copy CANNOT be the
    photograph regardless of how it looks.
    """
    if source_edge <= 0:
        return 0.0
    if not max_edge:          # None (the original) or 0 (no ceiling) — nothing was cut
        return 1.0
    return min(1.0, (max_edge / source_edge) ** 2)


def weigh_jpeg(image: Image.Image) -> int:
    """The bytes the copy would take on disk, at the quality the feature really writes.

    Encoded into memory rather than to a file: what is being measured is the weight of
    the result, and nothing about this script may leave a copy beside a photograph.
    """
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, "JPEG", quality=restore.JPEG_QUALITY)
    return buffer.tell()


def source_size(path: Path) -> tuple[int, int]:
    """(width, height) off the header — no pixels are decoded, and (0, 0) if it will not
    open."""
    try:
        with Image.open(path) as im:
            return int(im.width), int(im.height)
    except Exception:  # noqa: BLE001 — an unreadable frame is a row, like everything here
        return 0, 0


def baseline_run(path: Path, file_id: int) -> FrameRun:
    """The frame as it lies on disk — the row every other row is read against."""
    size = source_size(path)
    weight = path.stat().st_size if path.exists() else 0
    return FrameRun(file_id=file_id, population=population_of(max(size, default=0)),
                    max_edge=None, source_size=size, input_size=size, output_size=size,
                    weight_bytes=int(weight), seconds=0.0)


def run_frame(upscale: restore.UpscaleFn, path: Path, file_id: int,
              max_edge: int) -> tuple[FrameRun, Image.Image | None]:
    """One frame through the model at one ceiling: the row, and the picture it produced.

    The decode is the pipeline's own (`imaging.decode_rgb` with `apply_orientation`, what
    `restore.restore_frame` calls), so the table prices the action that runs rather than a
    private imitation of it. The picture comes back for the blind sheets and is never
    written next to the original.
    """
    size = source_size(path)
    population = population_of(max(size, default=0))
    vram_peak_mb(reset=True)
    started = time.perf_counter()
    image = imaging.decode_rgb(path, max_edge or None, apply_orientation=True)
    if image is None:
        return FrameRun(file_id=file_id, population=population, max_edge=max_edge,
                        source_size=size, input_size=(0, 0), output_size=(0, 0),
                        weight_bytes=0, seconds=0.0,
                        error="кадр не читается"), None
    try:
        processed = upscale(image)
    except Exception as exc:  # noqa: BLE001 — out of memory IS one of the answers
        return FrameRun(file_id=file_id, population=population, max_edge=max_edge,
                        source_size=size, input_size=image.size, output_size=(0, 0),
                        weight_bytes=0, seconds=time.perf_counter() - started,
                        peak_vram_mb=vram_peak_mb(),
                        error=f"{type(exc).__name__}: {exc}"), None
    seconds = time.perf_counter() - started
    run = FrameRun(file_id=file_id, population=population, max_edge=max_edge,
                   source_size=size, input_size=image.size, output_size=processed.size,
                   weight_bytes=weigh_jpeg(processed), seconds=seconds,
                   peak_vram_mb=vram_peak_mb())
    return run, processed


# --- the tables ----------------------------------------------------------------------


@dataclass(frozen=True)
class EdgeSummary:
    """What one ceiling did to one population — medians over the frames of that band."""
    max_edge: int | None
    frames: int
    failed: int
    input_size: tuple[int, int]
    output_size: tuple[int, int]
    weight_bytes: float
    seconds: float
    peak_vram_mb: float | None
    truth_kept: float
    error: str | None = None


def _median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def _median_size(sizes: list[tuple[int, int]]) -> tuple[int, int]:
    if not sizes:
        return (0, 0)
    return (int(_median([float(w) for w, _ in sizes])),
            int(_median([float(h) for _, h in sizes])))


def summarize(runs: list[FrameRun]) -> EdgeSummary:
    """The row for one ceiling. A failed frame counts in `failed` and in nothing else —
    averaging a crash into a time would price a run that never happened."""
    max_edge = runs[0].max_edge if runs else None
    good = [r for r in runs if r.error is None]
    failed = [r for r in runs if r.error is not None]
    peaks = [r.peak_vram_mb for r in runs if r.peak_vram_mb is not None]
    return EdgeSummary(
        max_edge=max_edge, frames=len(good), failed=len(failed),
        input_size=_median_size([r.input_size for r in good]),
        output_size=_median_size([r.output_size for r in good]),
        weight_bytes=_median([float(r.weight_bytes) for r in good]),
        seconds=_median([r.seconds for r in good]),
        peak_vram_mb=max(peaks) if peaks else None,
        truth_kept=_median([truth_kept(r.source_edge, r.max_edge) for r in good]),
        error=failed[0].error if failed and not good else None)


def summaries(runs: list[FrameRun], edges: list[int]) -> list[EdgeSummary]:
    """The baseline first, then one row per ceiling — the order the table is read in."""
    out: list[EdgeSummary] = []
    for max_edge in [None, *edges]:
        rows = [r for r in runs if r.max_edge == max_edge]
        if rows:
            out.append(summarize(rows))
    return out


def _size(size: tuple[int, int]) -> str:
    return f"{size[0]}x{size[1]}" if size[0] and size[1] else "—"


def _mb(value: float) -> str:
    return f"{value / (1024 * 1024):.2f} МБ" if value else "—"


def _edge_label(max_edge: int | None) -> str:
    if max_edge is None:
        return "оригинал"
    return "целиком" if max_edge == FULL_FRAME else str(max_edge)


def format_population_table(population: str, runs: list[FrameRun],
                            edges: list[int]) -> str:
    """One population, one table: the original first, then every ceiling under it."""
    rows = summaries(runs, edges)
    frames = max((r.frames + r.failed for r in rows), default=0)
    out = [
        "=" * 92,
        f"ПОПУЛЯЦИЯ «{POPULATION_LABEL[population]}»: кадров {frames}",
        f"{'предел':>10} {'вход':>12} {'выход':>12} {'вес':>10} {'время':>9} "
        f"{'пик VRAM':>10} {'правды':>8}",
    ]
    if not rows:
        out += ["  (в выборке нет кадров этой популяции)", "=" * 92]
        return "\n".join(out)
    for row in rows:
        if row.error is not None:
            out.append(f"{_edge_label(row.max_edge):>10} {'—':>12} {'—':>12} "
                       f"не получилось: {row.error}")
            continue
        vram = f"{row.peak_vram_mb:.0f} МБ" if row.peak_vram_mb is not None else "—"
        seconds = "—" if row.max_edge is None else f"{row.seconds:.2f} с"
        out.append(
            f"{_edge_label(row.max_edge):>10} {_size(row.input_size):>12} "
            f"{_size(row.output_size):>12} {_mb(row.weight_bytes):>10} {seconds:>9} "
            f"{vram:>10} {row.truth_kept:>7.0%}")
        if row.failed:
            out.append(f"{'':>10} из них не получилось: {row.failed}")
    out.append("=" * 92)
    return "\n".join(out)


def format_verdict_prompt(out_dir: Path) -> str:
    """The block that hands the decision back to the person, with the choices named.

    The script stops here deliberately. Every number above is a cost; not one of them
    says whether a rebuilt copy is worth having, and a script that guessed at that would
    be repeating the mistake this whole measurement exists because of.
    """
    return "\n".join([
        "ВЕРДИКТ ЗАПИСЫВАЕТ ЧЕЛОВЕК, А НЕ СКРИПТ.",
        f"Слепые пары лежат в {out_dir}: две картинки в одном размере, без подписей.",
        "pair_NN.jpg — кадр целиком; pair_NN_crop.jpg — та же середина обеих картинок",
        "в масштабе 1:1 (там, где копия вернулась размером с оригинал: на уменьшенном",
        "листе разница в настоящей детализации просто не видна).",
        "Посмотрите их ДО того, как откроете key.json — ключ там же.",
        "",
        "Вопрос по каждой паре один: какая из двух картинок ближе к тому, что было?",
        "«Резче» и «лучше» — разные вещи; дорисованная детализация выглядит убедительно",
        "именно тогда, когда она выдумана.",
        "",
        "Что означает ответ (порядок решения из брифа F169):",
        "  на полноразмерных ХУЖЕ оригинала  -> D: действие закрывается для больших",
        "     кадров и заводится отдельная фича на деблюр — «увеличить» и «навести",
        "     фокус» это разные задачи и разные модели;",
        "  на полноразмерных СОПОСТАВИМО     -> C: тайлинг в родном разрешении плюс",
        "     возврат к исходному размеру (суперсэмплинг), и только после этого F168",
        "     выпускает кнопку в остальные срезы.",
        "Мелкая популяция от исхода не зависит: там предел не срабатывает вовсе.",
    ])


# --- the blind pairs -----------------------------------------------------------------


def sheet_half_size(size: tuple[int, int], box: int = SHEET_EDGE) -> tuple[int, int]:
    """The size both halves of a sheet are drawn at — `size` fitted into a square box."""
    edge = max(size) or 1
    scale = box / edge
    return (max(1, round(size[0] * scale)), max(1, round(size[1] * scale)))


def blind_sheet(original: Image.Image, processed: Image.Image, flipped: bool,
                box: int = SHEET_EDGE) -> Image.Image:
    """The two pictures side by side, at the same size, with nothing saying which is which.

    EXACTLY the same size, computed from the original and imposed on both: the x4 output
    is always the bigger one, so a sheet that merely shrank each half to fit would hand
    the answer over before the first pair is open. No caption, no border, and no order to
    learn either — `flipped` comes from the seeded generator, so "the original was on the
    left last time" is not a thing anybody can notice.
    """
    left, right = (processed, original) if flipped else (original, processed)
    target = sheet_half_size(original.size, box)
    halves = [image.convert("RGB").resize(target) for image in (left, right)]
    width = sum(h.width for h in halves) + SHEET_GAP
    height = max(h.height for h in halves)
    sheet = Image.new("RGB", (width, height), SHEET_BACKGROUND)
    sheet.paste(halves[0], (0, 0))
    sheet.paste(halves[1], (halves[0].width + SHEET_GAP, 0))
    return sheet


def centre_crop(image: Image.Image, box: int = CROP_BOX) -> Image.Image:
    """The middle `box` x `box` of the picture — the whole picture if it is smaller."""
    side = min(box, image.width, image.height)
    left, top = (image.width - side) // 2, (image.height - side) // 2
    return image.crop((left, top, left + side, top + side))


def same_scale(one: tuple[int, int], other: tuple[int, int],
               tolerance: float = 0.1) -> bool:
    """Are these two the same picture at the same size (a rebuild, not an enlargement)?

    Which is the whole question above the ceiling: a frame there is reduced and blown
    back up to about its own size, so the copy and the original are directly comparable
    pixel for pixel. Below the ceiling the copy is genuinely four times bigger and no
    such comparison exists.
    """
    if not max(one) or not max(other):
        return False
    return abs(max(one) / max(other) - 1.0) <= tolerance


def write_blind_pairs(pairs: list[tuple[FrameRun, Image.Image, Image.Image]],
                      out_dir: Path, seed: int) -> list[dict]:
    """Write the sheets for every (run, original, processed) and return the key.

    The sheets are NUMBERED, never named after the frame: a basename is what would
    identify somebody's photograph in a folder meant to be looked through, and a
    filename saying `_restored` would end the blindness before the first pair is open.

    Two sheets where two are possible. `pair_NN.jpg` is the whole frame, which is how a
    person meets a photograph — and, on a 12 Mpx frame shrunk to fit a sheet, is also how
    a difference in real detail disappears, because shrinking is itself a way of making
    detail agree. So whenever the copy came back at the size of its original (exactly the
    population this measurement is about), `pair_NN_crop.jpg` shows the same middle of
    both at native scale, where there is nowhere for the difference to hide. The flip is
    the SAME for both sheets of a pair: two sheets of one frame disagreeing about which
    side is which would tell the answer instead of hiding it.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    key: list[dict] = []
    for n, (run, original, processed) in enumerate(pairs, start=1):
        flipped = rng.random() < 0.5
        sheets = {"full": f"pair_{n:02d}.jpg"}
        blind_sheet(original, processed, flipped).save(
            out_dir / sheets["full"], "JPEG", quality=restore.JPEG_QUALITY)
        if same_scale(original.size, processed.size):
            sheets["crop"] = f"pair_{n:02d}_crop.jpg"
            blind_sheet(centre_crop(original), centre_crop(processed), flipped,
                        box=CROP_BOX).save(
                out_dir / sheets["crop"], "JPEG", quality=restore.JPEG_QUALITY)
        key.append({
            "sheets": sheets,
            "left": PROCESSED if flipped else ORIGINAL,
            "right": ORIGINAL if flipped else PROCESSED,
            "file_id": run.file_id,
            "population": run.population,
            "max_edge": run.max_edge,
        })
    return key


# --- the sample ----------------------------------------------------------------------


def sample_frames(db_path: str, per_population: int,
                  seed: int) -> dict[str, list[tuple[int, str]]]:
    """`per_population` frames from each band: canonical photographs still on disk.

    Deterministic for a given seed, so a second run at another ceiling talks about the
    same frames. Sizes come from the index (`files.width/height`) because the point is to
    choose the band without decoding a collection first.
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """SELECT id, path, width, height FROM files
               WHERE dup_of IS NULL AND error IS NULL AND media_type = 'photo'
                 AND width > 0 AND height > 0
               ORDER BY id""").fetchall()
    finally:
        conn.close()
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    picked: dict[str, list[tuple[int, str]]] = {p: [] for p in POPULATIONS}
    for row in shuffled:
        band = population_of(max(int(row["width"]), int(row["height"])))
        if len(picked[band]) >= per_population or not Path(row["path"]).exists():
            continue
        picked[band].append((int(row["id"]), str(row["path"])))
    return picked


def measure(upscale: restore.UpscaleFn, frames: dict[str, list[tuple[int, str]]],
            edges: list[int]) -> tuple[list[FrameRun], list[tuple[FrameRun, Image.Image,
                                                                 Image.Image]]]:
    """Every frame at every ceiling, plus the baseline row and the pairs for the eyes."""
    runs: list[FrameRun] = []
    pairs: list[tuple[FrameRun, Image.Image, Image.Image]] = []
    for population in POPULATIONS:
        for file_id, path in frames[population]:
            src = Path(path)
            runs.append(baseline_run(src, file_id))
            # Full size, not sheet size: the crop half of a pair is a 1:1 window into the
            # original, and a pre-shrunk one would have nothing left to show there.
            original = imaging.decode_rgb(src, None, apply_orientation=True)
            for max_edge in edges:
                run, processed = run_frame(upscale, src, file_id, max_edge)
                runs.append(run)
                if processed is not None and original is not None:
                    pairs.append((run, original, processed))
    return runs, pairs


def main() -> int:  # pragma: no cover — needs the weights and a collection
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--sample", type=int, default=DEFAULT_SAMPLE,
                    help=f"frames per population (default {DEFAULT_SAMPLE})")
    ap.add_argument("--seed", type=int, default=20260804)
    ap.add_argument("--edges", nargs="+", type=int, default=list(DEFAULT_EDGES),
                    help="ceilings to compare, the shipped one first; 0 — the full frame")
    ap.add_argument("--out", default="measure_restore",
                    help="where the blind pairs and key.json go (default measure_restore)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    frames = sample_frames(str(cfg.database), args.sample, args.seed)
    total = sum(len(v) for v in frames.values())
    if not total:
        raise SystemExit("в индексе нет подходящих кадров — мерить нечего")
    print(f"модель: {cfg.features.restore_model}")
    print(f"предел в конфиге: features.restore_max_edge = {cfg.features.restore_max_edge}")
    for population in POPULATIONS:
        print(f"выборка, {POPULATION_LABEL[population]}: {len(frames[population])} кадров")
    print("веса грузятся один раз, дальше — по кадру на предел...")

    upscale = restore.shared_upscaler(cfg.features.restore_model)
    runs, pairs = measure(upscale, frames, args.edges)

    print()
    for population in POPULATIONS:
        print(format_population_table(
            population, [r for r in runs if r.population == population], args.edges))
    out_dir = Path(args.out)
    key = write_blind_pairs(pairs, out_dir, args.seed)
    (out_dir / "key.json").write_text(
        json.dumps(key, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print(format_verdict_prompt(out_dir))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
