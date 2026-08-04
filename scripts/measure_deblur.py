"""F183, phase 0 — sharpness WITHOUT enlargement: is a 1:1 model the right instrument?

F169 measured the button of F149 and closed one half of the question with numbers: on a
SMALL frame the x4 model works and is chosen — blurred small frames came back better than
bicubic in 62% of the blind pairs against 10%, 37:6, p < 0.001, and the gain grows as the
frame gets smaller. Nothing here touches that population, ever: below the ceiling the frame
is enlarged and nothing of it is given up.

What stayed open is the OTHER case, and it was the original reason for all of this: a good
full-sized frame that is simply soft. There —

    4032 x 3024 (12 Mpx)  ->  1024 x 768  ->  4096 x 3072

— the frame is squeezed to a quarter of its side and rebuilt from the reduced copy, and
the blind pairs on fidelity came back 35/35/30 and 15/65/20: nobody could call the copy
closer to what was there. All of that arithmetic follows from the MULTIPLIER, not from the
hardware: a model that returns what it was given (12 Mpx in, 12 Mpx out) has no ceiling to
pick, nothing to tile and no choice to make between "rebuild from a quarter" and "leave big
frames alone".

THE QUESTION THIS SCRIPT ASKS IS FIDELITY, NOT "BETTER". Twice in one day a measurement
that asked "which one is better" gave the wrong answer, in both directions: "sharper" and
"closer to what was there" are different things, and invented detail is convincing exactly
when it is invented. Every blind pair here asks one question — which of the two is closer
to the frame that was taken.

WHAT IS DELIBERATELY NOT DECIDED HERE. No model is named by the brief and none is named
below: F149's first probe took `swin2SR-classical-sr`, trained on clean bicubic
downscaling, scored well and turned out to be useless on real smear. A broken instrument
flatters the result, so the candidates are named on the command line (`--models`), and
every one of them is checked before it is priced (see `probe_one_to_one`): a model that
enlarges is not this class of model, and a model that returns its input unchanged is a
null result wearing the costume of "did no harm".

THE RISK THAT IS CHECKED FIRST. Deblurring models are trained mostly on MOTION blur
(GoPro-style datasets: camera shake, a moving subject). This population is "soft frames by
the owner's strict criterion", and among them there are missed focus, shallow depth of
field and plain weak optics — different degradations, and a model trained on one may do
nothing at all for another. So the sample is SPLIT (`degradation_of`) and every table is
printed per type. One average over the three would answer none of them — the mistake that
turned "screenshots leak" into a disagreement about definitions.

What it prints:

1. THE PRICE, on 1, 4 and 12 megapixels: milliseconds per frame and peak VRAM, with the
   growth named out loud (`growth_verdict`). The premise of this feature is that the cost
   grows LINEARLY with the pixels instead of quadratically with the x4 output; if it does
   not, the premise is wrong and the run says to stop.
2. THE POPULATION SPLIT BY DEGRADATION — motion smear, missed focus, general softness —
   with a table for each.
3. THE CURRENT MODEL BESIDE IT on the same frames, at the shipped ceiling. Without a
   baseline any picture looks like an improvement.
4. HOW MANY FRAMES THIS TOUCHES: the blurred slice as the product itself defines it
   (`sorter.quality_slice_where`), and the part of it above the ceiling — which is the
   only part this feature is about.
5. BLIND PAIRS FOR THE EYES: the original and a result, same size, no caption, and the
   sheets SHUFFLED across arms so their order gives away neither which model made a copy
   nor which side it is on. `key.json` is meant to be opened after looking, not before.

Privacy: no path, no basename and no thumbnail is printed. Sheets are numbered
(`pair_01.jpg`) and the key holds file ids alone — the rule `scripts/measure_restore.py`
and `scripts/measure_ocr_gate.py` follow.

The originals are opened read-only and never written: this script measures the action, it
does not perform it (nothing is saved beside anybody's photograph).

Usage (from the repo root, with a GPU venv — `uv sync --extra gpu --extra vlm`):
    python scripts/measure_deblur.py --models <repo/weights> [<repo/weights> ...]
    python scripts/measure_deblur.py --models <weights> --sample 16 --out measure_deblur
    python scripts/measure_deblur.py --models <weights> --no-baseline --megapixels 1 4 12
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sqlite3
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))  # the repo root — for `sorta`
sys.path.insert(0, str(_HERE))         # ...and this directory — for the sibling scripts

from sorta import imaging, junk, restore, sorter  # noqa: E402
from sorta.config import FeaturesConfig, load_config  # noqa: E402

# The blind sheets, the 1:1 crop and the peak-memory reader come from the F169 measurement
# rather than being written a second time. Both runs are read against each other — a second
# version of "the same middle of both pictures at native scale" would differ in some detail
# and the two verdicts would stop being comparable, which is the whole reason this feature
# exists at all.
from measure_restore import (  # noqa: E402
    CROP_BOX,
    ORIGINAL,
    PROCESSED,
    blind_sheet,
    centre_crop,
    same_scale,
    source_size,
    truth_kept,
    vram_peak_mb,
    weigh_jpeg,
)

RestoreFn = restore.UpscaleFn

# --- the degradations -----------------------------------------------------------------
#
# Three kinds of soft, and they are three because they are produced by three different
# accidents and are repaired — if at all — by different training sets. A model taught on
# camera shake has never been shown a missed focus.

MOTION, DEFOCUS, SOFT, UNKNOWN = "motion", "defocus", "soft", "unknown"
DEGRADATIONS = (MOTION, DEFOCUS, SOFT, UNKNOWN)
DEGRADATION_LABEL = {
    MOTION: "смаз движения",
    DEFOCUS: "промах фокуса",
    SOFT: "общая мягкость",
    UNKNOWN: "не удалось определить",
}

# A smear has a DIRECTION: it destroys the detail along the way the camera moved and leaves
# the detail across it, so the gradients of the frame collapse onto one axis. Missed focus
# and weak optics have no direction at all. This is the number that separates them —
# (l1 - l2) / (l1 + l2) over the structure tensor, 0 for a frame that is equally detailed
# in every direction and 1 for one that holds detail in a single direction only.
#
# 0.35 is a threshold and not a discovery, which matters twice over: real frames are
# anisotropic on their own (a horizon, a fence, a skyline), so this splits the sample
# APPROXIMATELY and says so. The type of every frame goes into `key.json` beside its blind
# pair, so a person who looks at the pictures can correct the split rather than inherit it.
MOTION_ANISOTROPY = 0.35

# Missed focus against general softness is a matter of DEGREE, on the very scale the
# product already ranks the blurred list by. So the line is drawn as a fraction of
# `features.blur_review_max` (the window the slice opens to) instead of a constant of its
# own: a person who moves the window moves this with it, and no number here can drift away
# from the list it is supposed to describe.
DEFOCUS_SHARE = 0.5

# The anisotropy is taken at NATIVE scale over the middle of the frame, and the sharpness
# is taken over a preview at `features.sharpness_max_edge` — on purpose, and they are not
# interchangeable. A three-pixel smear does not survive a 4x downscale, so direction has to
# be looked for in the frame's own pixels; while a laplacian variance is scale-dependent,
# and the only number `blur_review_max` may be compared against is one taken exactly as the
# indexing stage takes it.
ANISOTROPY_CROP = 1024

# Below this many frames of one type, that type has no answer here — it has an anecdote.
# Printed as a caveat rather than enforced: a run that refused to show what it measured
# would be worse than one that shows it with the count beside it.
MIN_PER_TYPE = 5

# --- the price ------------------------------------------------------------------------

# The three sizes of brief item 1. 12 Mpx is an ordinary phone frame of this collection and
# the size at which the x4 arm is expected to fail; 1 Mpx is there to give the growth a
# short end to be measured against.
COST_MEGAPIXELS = (1, 4, 12)

# How far the cost per megapixel may drift across those sizes and still be called linear.
# Not 1.0: a fixed cost per call (weights on the card, the processor's own work) shows up
# as a HIGHER per-megapixel price on the small frame, which is the harmless direction.
LINEAR_TOLERANCE = 1.5

# --- the arms -------------------------------------------------------------------------

ARM_ORIGINAL = "оригинал"
ARM_BASELINE = "нынешняя x4"


@dataclass(frozen=True)
class Arm:
    """One instrument on one frame: what it is called, what it is shown, what it does.

    `max_edge` is the ceiling on the way IN — `None` for a 1:1 candidate, which is the
    whole point of it (the frame goes as it lies), and `features.restore_max_edge` for the
    baseline, because that is what the shipped button really does.
    """
    name: str
    max_edge: int | None
    process: RestoreFn | None = None

    @property
    def is_original(self) -> bool:
        return self.process is None


@dataclass(frozen=True)
class Degradation:
    """Which of the three kinds of soft this frame is, and the two numbers that decided."""
    kind: str
    sharpness: float | None
    anisotropy: float | None


@dataclass(frozen=True)
class FrameRun:
    """One frame through one arm. An error is a ROW, not an exception.

    "It did not fit into memory" and "the weights are broken on this frame" are answers
    this measurement exists to produce, and a traceback would end the run before the frames
    that DO work were priced.
    """
    file_id: int
    arm: str
    degradation: str
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
    def output_edge(self) -> int:
        return max(self.output_size) if self.output_size else 0


# --- what kind of soft this frame is ---------------------------------------------------


def fit_edge(image: Image.Image, max_edge: int | None) -> Image.Image:
    """`image` scaled DOWN so its longer side is `max_edge`; the image itself if it fits.

    Never up: enlarging a frame to reach a ceiling would invent the pixels the number below
    is then computed over.
    """
    edge = max(image.size)
    if not max_edge or edge <= max_edge:
        return image
    scale = max_edge / edge
    return image.resize((max(1, round(image.width * scale)),
                         max(1, round(image.height * scale))))


def anisotropy(image: Image.Image) -> float | None:
    """How much of the frame's detail lies along ONE direction — 0 none, 1 all of it.

    The structure tensor of the gradients, whose two eigenvalues are the detail along and
    across the dominant direction; their normalised difference is scale-free, so it can be
    compared between a 12 Mpx frame and a 2 Mpx one without a threshold per size.

    None for a frame too small to have a gradient — nothing measured, never a 0.0 that
    would read as "perfectly isotropic".
    """
    a = np.asarray(image.convert("L"), dtype=np.float32)
    if a.ndim != 2 or a.shape[0] < 3 or a.shape[1] < 3:
        return None
    gy, gx = np.gradient(a)
    jxx, jyy, jxy = float((gx * gx).mean()), float((gy * gy).mean()), float((gx * gy).mean())
    trace = jxx + jyy
    if trace <= 0.0:                       # a perfectly flat frame has no direction at all
        return None
    spread = math.sqrt(max(0.0, (jxx - jyy) ** 2 + 4.0 * jxy * jxy))
    return spread / trace


def degradation_of(image: Image.Image, *, blur_max: float,
                   sharpness_edge: int) -> Degradation:
    """Which kind of soft this frame is: motion smear, missed focus, or general softness.

    A HEURISTIC SPLIT, and it is named as one everywhere it is printed. It is here because
    the alternative — one average over the three — is the answer to no question at all: a
    model trained on camera shake can be excellent on a third of this population and do
    nothing whatsoever for the other two thirds, and a single number would report that as
    "it sort of works".

    Direction decides first (a smear is directional and nothing else here is), and only
    then degree: deep inside the blurred window is a missed focus, near the top of it is a
    frame that is merely soft.
    """
    directional = anisotropy(centre_crop(image, ANISOTROPY_CROP))
    sharpness = junk.laplacian_variance(fit_edge(image, sharpness_edge))
    if directional is None or sharpness is None:
        return Degradation(UNKNOWN, sharpness, directional)
    if directional >= MOTION_ANISOTROPY:
        return Degradation(MOTION, sharpness, directional)
    if sharpness < blur_max * DEFOCUS_SHARE:
        return Degradation(DEFOCUS, sharpness, directional)
    return Degradation(SOFT, sharpness, directional)


# --- is this candidate the class of model the brief is asking about? -------------------


@dataclass(frozen=True)
class Probe:
    """What a candidate did to one small picture, before a single frame is spent on it."""
    scale: float
    changed: float
    error: str | None = None

    @property
    def one_to_one(self) -> bool:
        return abs(self.scale - 1.0) <= 0.01

    @property
    def usable(self) -> bool:
        return self.error is None and self.one_to_one and self.changed > PROBE_MIN_CHANGE

    @property
    def reason(self) -> str:
        """Why this candidate is not measured — empty when it is."""
        if self.error is not None:
            return f"не запустилась: {self.error}"
        if not self.one_to_one:
            return (f"это не модель один к одному: на выходе x{self.scale:.2f}. "
                    "Такую меряет F169, а не эта фича")
        if self.changed <= PROBE_MIN_CHANGE:
            return ("модель вернула кадр без изменений — это не «не навредила», "
                    "это отсутствие результата")
        return ""


# A model that moves the average pixel by less than this much (of 255) has done nothing.
# The probe picture is noise, where any real restoration model has plenty to change.
PROBE_MIN_CHANGE = 0.5
PROBE_SIZE = (256, 192)


def probe_image(size: tuple[int, int] = PROBE_SIZE, seed: int = 0) -> Image.Image:
    """A small picture with detail in every direction — what a candidate is tried on."""
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8),
                           "RGB")


def probe_one_to_one(process: RestoreFn, size: tuple[int, int] = PROBE_SIZE) -> Probe:
    """Run the candidate on one small picture and report what it IS, not how good it is.

    The F149 lesson made mechanical. Two ways to be the wrong instrument, and both are
    cheap to catch before a collection is spent on them: a model that enlarges answers the
    question F169 already answered, and a model that hands the frame back untouched
    produces blind pairs that a person will score at 50/50 and read as "no harm done".
    """
    picture = probe_image(size)
    try:
        result = process(picture)
    except Exception as exc:  # noqa: BLE001 — a candidate that will not run is a row
        return Probe(scale=0.0, changed=0.0, error=f"{type(exc).__name__}: {exc}")
    scale = max(result.size) / max(picture.size) if max(picture.size) else 0.0
    if abs(scale - 1.0) > 0.01:
        return Probe(scale=scale, changed=0.0)
    before = np.asarray(picture.convert("RGB"), dtype=np.float32)
    after = np.asarray(result.convert("RGB").resize(picture.size), dtype=np.float32)
    return Probe(scale=scale, changed=float(np.abs(after - before).mean()))


def load_restorer(model_name: str) -> RestoreFn:  # pragma: no cover — ML, needs weights
    """Load a 1:1 restoration model through transformers -> process(image) -> image.

    Lazy-import, like every model in this project (`restore.load_swin2sr`): the module
    imports without transformers installed and the failure happens where the caller is
    already turning it into a printed row. Deliberately generic — `AutoModelForImageToImage`
    rather than a class per family — because the brief names no model and this run exists
    to try several of them beside each other.
    """
    import torch
    from transformers import AutoImageProcessor, AutoModelForImageToImage

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageToImage.from_pretrained(model_name).to(device)
    model.eval()

    def process(image: Image.Image) -> Image.Image:
        inputs = processor(image, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model(**inputs)
        raw = getattr(output, "reconstruction", None)
        if raw is None:
            raw = output[0]
        array: Any = raw.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        array = np.moveaxis(array, source=0, destination=-1)
        return Image.fromarray((array * 255.0).round().astype(np.uint8))

    return process


# --- the price ------------------------------------------------------------------------


@dataclass(frozen=True)
class CostRow:
    """What one arm cost on a frame of a given size. `error` is an answer, not a failure."""
    arm: str
    megapixels: float
    input_size: tuple[int, int]
    output_size: tuple[int, int]
    seconds: float
    peak_vram_mb: float | None = None
    error: str | None = None

    @property
    def ms_per_megapixel(self) -> float:
        return (self.seconds * 1000.0) / self.megapixels if self.megapixels else 0.0

    @property
    def mb_per_megapixel(self) -> float | None:
        if self.peak_vram_mb is None or not self.megapixels:
            return None
        return self.peak_vram_mb / self.megapixels


def synthetic_frame(megapixels: float, seed: int = 0) -> Image.Image:
    """A 4:3 picture of about `megapixels`, made of noise.

    Made rather than taken from the collection on purpose: the price of a forward pass
    depends on the number of pixels and not on what is in them, and a table of costs is the
    one part of this run that needs no photograph at all.
    """
    width = max(2, round(math.sqrt(megapixels * 1_000_000 * 4 / 3)))
    height = max(2, round(width * 3 / 4))
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8), "RGB")


def cost_row(arm: Arm, megapixels: float, *, frame: Image.Image | None = None) -> CostRow:
    """One arm on one frame size — milliseconds and peak VRAM, or the reason there are none.

    Only the MODEL CALL is timed, here and in `run_frame` alike, so the two tables speak of
    the same thing. The decode is left out of both: it is a property of the file on disk,
    it differs between the arms only because one of them throws most of the frame away, and
    including it would price the reading of a JPEG as part of an instrument's cost.
    """
    picture = frame if frame is not None else synthetic_frame(megapixels)
    shown = fit_edge(picture, arm.max_edge)
    if arm.process is None:  # the original arm has no price — there is nothing to run
        return CostRow(arm=arm.name, megapixels=megapixels, input_size=shown.size,
                       output_size=shown.size, seconds=0.0)
    vram_peak_mb(reset=True)
    started = time.perf_counter()
    try:
        result = arm.process(shown)
    except Exception as exc:  # noqa: BLE001 — out of memory IS one of the answers
        return CostRow(arm=arm.name, megapixels=megapixels, input_size=shown.size,
                       output_size=(0, 0), seconds=time.perf_counter() - started,
                       peak_vram_mb=vram_peak_mb(), error=f"{type(exc).__name__}: {exc}")
    return CostRow(arm=arm.name, megapixels=megapixels, input_size=shown.size,
                   output_size=result.size, seconds=time.perf_counter() - started,
                   peak_vram_mb=vram_peak_mb())


def _growth(rows: list[CostRow], value: Callable[[CostRow], float | None]) -> float | None:
    """How much the per-megapixel `value` grew between the smallest and the largest frame.

    1.0 — the same price per megapixel at every size, which is what linear means here.
    """
    ordered = sorted(rows, key=lambda r: r.megapixels)
    first, last = value(ordered[0]), value(ordered[-1])
    if first is None or last is None or first <= 0:
        return None
    return last / first


def growth_verdict(rows: list[CostRow],
                   tolerance: float = LINEAR_TOLERANCE) -> tuple[bool, str]:
    """Does the cost grow with the PIXELS, or faster? The premise of the feature, checked.

    The whole case for a 1:1 model is that 12 megapixels in are 12 megapixels out, so the
    price follows the frame instead of the square of a multiplier. If it does not, nothing
    further in this run matters and the run says so instead of leaving the reader to
    compare three numbers by eye.
    """
    good = [r for r in rows if r.error is None]
    failed = [r for r in rows if r.error is not None]
    if failed:
        biggest = max(failed, key=lambda r: r.megapixels)
        return False, (f"на {biggest.megapixels:g} Мп не получилось: {biggest.error}. "
                       "Ради этого фича и заводилась — если 1:1 тоже не держит "
                       "полноразмерный кадр, посылка неверна, останавливаемся")
    if len(good) < 2:
        return False, "рост мерить не на чем: нужно хотя бы два размера"
    time_growth = _growth(good, lambda r: r.ms_per_megapixel)
    if time_growth is None:
        return False, "рост мерить не на чем: на маленьком кадре не измерилось время"
    memory_growth = _growth(good, lambda r: r.mb_per_megapixel)
    small = min(good, key=lambda r: r.megapixels).megapixels
    big = max(good, key=lambda r: r.megapixels).megapixels
    span = f"с {small:g} Мп на {big:g} Мп"
    if time_growth > tolerance:
        return False, (f"время растёт БЫСТРЕЕ пикселей: цена мегапикселя x{time_growth:.2f} "
                       f"{span} — посылка неверна, останавливаемся")
    if memory_growth is not None and memory_growth > tolerance:
        return False, (f"память растёт БЫСТРЕЕ пикселей: пик на мегапиксель "
                       f"x{memory_growth:.2f} {span} — посылка неверна, останавливаемся")
    memory = "" if memory_growth is None else f", память x{memory_growth:.2f}"
    return True, (f"рост линейный: цена мегапикселя x{time_growth:.2f} {span}{memory} "
                  f"(порог x{tolerance:g})")


def format_cost_table(rows: list[CostRow]) -> str:
    """Brief item 1: milliseconds and peak memory per frame size, per arm, with the verdict."""
    out = [
        "=" * 92,
        "ЦЕНА КАДРА (синтетические кадры, считается только вызов модели)",
        f"{'инструмент':>14} {'кадр':>8} {'вход':>12} {'выход':>12} {'время':>9} "
        # "мс на Мп", not "мс/Мп": the table is guarded against leaking a path,
        # and that guard rejects a slash. A unit is not worth weakening it —
        # paths in this index are POSIX, so the slash is what catches a real leak.
        f"{'мс на Мп':>10} {'пик VRAM':>10}",
    ]
    for row in sorted(rows, key=lambda r: (r.arm, r.megapixels)):
        if row.error is not None:
            out.append(f"{row.arm:>14} {row.megapixels:>7g}М не получилось: {row.error}")
            continue
        vram = f"{row.peak_vram_mb:.0f} МБ" if row.peak_vram_mb is not None else "—"
        out.append(
            f"{row.arm:>14} {row.megapixels:>7g}М {_size(row.input_size):>12} "
            f"{_size(row.output_size):>12} {row.seconds * 1000:>7.0f} мс "
            f"{row.ms_per_megapixel:>9.0f} {vram:>10}")
    for arm in sorted({row.arm for row in rows}):
        arm_rows = [r for r in rows if r.arm == arm]
        if len(arm_rows) < 2:
            continue
        linear, message = growth_verdict(arm_rows)
        out.append(f"  {arm}: {'ok' if linear else 'СТОП'} — {message}")
    out.append("=" * 92)
    return "\n".join(out)


# --- how many frames this touches -------------------------------------------------------

# The blurred slice as the PRODUCT defines it, above the ceiling this feature is about.
# `f.width`/`f.height` are what the indexer wrote, so the band is decided without decoding
# a collection to find out which frames are big.
ABOVE_CEILING = "f.width > 0 AND f.height > 0 AND MAX(f.width, f.height) > ?"

# What the count above cannot say, printed beside it every time. The filter finds 8% of
# what the owner calls soft by the strict criterion, so this is a floor and not a size.
REACH_CAVEAT = (
    "Фильтр размытых по строгому критерию владельца находит 8% — популяция известна\n"
    "плохо, и число ниже читается как оценка СНИЗУ, а не как размер среза.")


@dataclass(frozen=True)
class Reach:
    """Brief item 5: how many frames a 1:1 model would be offered at all."""
    photos: int
    above_ceiling: int
    blurred: int
    blurred_above_ceiling: int

    @property
    def share_of_blurred(self) -> float:
        return self.blurred_above_ceiling / self.blurred if self.blurred else 0.0


def _count(conn: sqlite3.Connection, frm: str, where: str, params: list[object]) -> int:
    return int(conn.execute(f"SELECT COUNT(*) {frm} WHERE {where}", params).fetchone()[0])


def slice_reach(conn: sqlite3.Connection, features: FeaturesConfig, ceiling: int) -> Reach:
    """The counts, off the product's OWN slice rules rather than a private imitation.

    `sorter.quality_slice_where('blurred', ...)` is the predicate the list, its counter and
    its album all read. Rewriting it here would let this measurement describe a population
    that no button in the interface can show.
    """
    photos_from = sorter.LOW_RESOLUTION_FROM
    photo_where = f"{sorter.QUALITY_POPULATION} AND f.width > 0 AND f.height > 0"
    where, params = sorter.quality_slice_where("blurred", features)
    blurred_from = sorter.quality_slice_from("blurred")
    return Reach(
        photos=_count(conn, photos_from, photo_where, []),
        above_ceiling=_count(conn, photos_from,
                             f"{sorter.QUALITY_POPULATION} AND {ABOVE_CEILING}", [ceiling]),
        blurred=_count(conn, blurred_from, where, list(params)),
        blurred_above_ceiling=_count(conn, blurred_from, f"{where} AND {ABOVE_CEILING}",
                                     [*params, ceiling]))


def format_reach(reach: Reach, ceiling: int) -> str:
    out = [
        "=" * 92,
        f"СКОЛЬКО КАДРОВ ЭТО КАСАЕТСЯ (потолок {ceiling} px)",
        f"  канонических фотографий:            {reach.photos}",
        f"  из них больше потолка:              {reach.above_ceiling}",
        f"  в срезе размытых:                   {reach.blurred}",
        f"  размытых И больше потолка:          {reach.blurred_above_ceiling} "
        f"({reach.share_of_blurred:.0%} среза)",
        "",
        REACH_CAVEAT,
        "Всё, что меньше потолка, остаётся за нынешней x4 навсегда (F169: на мелких",
        "модель лучше бикубика в 62% против 10%) — эта фича его не касается.",
        "=" * 92,
    ]
    return "\n".join(out)


def sample_frames(db_path: str, features: FeaturesConfig, ceiling: int, count: int,
                  seed: int) -> list[tuple[int, str]]:
    """`count` frames of the population in question: the blurred slice, above the ceiling.

    Seeded, so a second run with another candidate talks about the same frames — otherwise
    two models are compared on two collections. Sampled across the whole window rather than
    from its most blurred end: the window is what a person is shown.
    """
    where, params = sorter.quality_slice_where("blurred", features)
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            f"SELECT f.id, f.path {sorter.quality_slice_from('blurred')} "
            f"WHERE {where} AND {ABOVE_CEILING} ORDER BY f.id",
            [*params, ceiling]).fetchall()
    finally:
        conn.close()
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    picked: list[tuple[int, str]] = []
    for row in shuffled:
        if len(picked) >= count:
            break
        if Path(str(row["path"])).exists():
            picked.append((int(row["id"]), str(row["path"])))
    return picked


# --- one frame through one arm ----------------------------------------------------------


def original_run(path: Path, file_id: int, degradation: str) -> FrameRun:
    """The frame as it lies on disk — the row every other row in its table is read against."""
    size = source_size(path)
    weight = path.stat().st_size if path.exists() else 0
    return FrameRun(file_id=file_id, arm=ARM_ORIGINAL, degradation=degradation,
                    max_edge=None, source_size=size, input_size=size, output_size=size,
                    weight_bytes=int(weight), seconds=0.0)


def run_frame(arm: Arm, path: Path, file_id: int,
              degradation: str) -> tuple[FrameRun, Image.Image | None]:
    """One frame through one arm: the row, and the picture for the blind pair.

    The decode is the pipeline's own (`imaging.decode_rgb` with `apply_orientation`, what
    `restore.restore_frame` calls), so the row prices the action that would ship rather than
    a private imitation of it. The picture comes back for the sheets and is never written
    beside the original.
    """
    size = source_size(path)
    image = imaging.decode_rgb(path, arm.max_edge, apply_orientation=True)
    if image is None:
        return FrameRun(file_id=file_id, arm=arm.name, degradation=degradation,
                        max_edge=arm.max_edge, source_size=size, input_size=(0, 0),
                        output_size=(0, 0), weight_bytes=0, seconds=0.0,
                        error="кадр не читается"), None
    if arm.process is None:
        return original_run(path, file_id, degradation), image
    vram_peak_mb(reset=True)
    started = time.perf_counter()
    try:
        processed = arm.process(image)
    except Exception as exc:  # noqa: BLE001 — out of memory, broken weights: both are rows
        return FrameRun(file_id=file_id, arm=arm.name, degradation=degradation,
                        max_edge=arm.max_edge, source_size=size, input_size=image.size,
                        output_size=(0, 0), weight_bytes=0,
                        seconds=time.perf_counter() - started, peak_vram_mb=vram_peak_mb(),
                        error=f"{type(exc).__name__}: {exc}"), None
    seconds = time.perf_counter() - started
    run = FrameRun(file_id=file_id, arm=arm.name, degradation=degradation,
                   max_edge=arm.max_edge, source_size=size, input_size=image.size,
                   output_size=processed.size, weight_bytes=weigh_jpeg(processed),
                   seconds=seconds, peak_vram_mb=vram_peak_mb())
    return run, processed


def measure(arms: list[Arm], frames: list[tuple[int, str]], *, blur_max: float,
            sharpness_edge: int) -> tuple[list[FrameRun],
                                          list[tuple[FrameRun, Image.Image, Image.Image]]]:
    """Every frame through every arm, split by degradation, plus the pairs for the eyes."""
    runs: list[FrameRun] = []
    pairs: list[tuple[FrameRun, Image.Image, Image.Image]] = []
    for file_id, path in frames:
        src = Path(path)
        # Full size, not a preview: the crop half of a pair is a 1:1 window into the
        # original, and a pre-shrunk one would have nothing left to show there.
        original = imaging.decode_rgb(src, None, apply_orientation=True)
        kind = (degradation_of(original, blur_max=blur_max, sharpness_edge=sharpness_edge).kind
                if original is not None else UNKNOWN)
        runs.append(original_run(src, file_id, kind))
        for arm in arms:
            if arm.is_original:
                continue
            run, processed = run_frame(arm, src, file_id, kind)
            runs.append(run)
            if processed is not None and original is not None:
                pairs.append((run, original, processed))
    return runs, pairs


# --- the tables --------------------------------------------------------------------------


@dataclass(frozen=True)
class ArmSummary:
    """What one arm did to the frames of one degradation — medians over that band."""
    arm: str
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


def summarize(runs: list[FrameRun]) -> ArmSummary:
    """The row for one arm. A failed frame counts in `failed` and in nothing else — an
    average that includes a crash prices a run that never happened."""
    arm = runs[0].arm if runs else ""
    max_edge = runs[0].max_edge if runs else None
    good = [r for r in runs if r.error is None]
    failed = [r for r in runs if r.error is not None]
    peaks = [r.peak_vram_mb for r in runs if r.peak_vram_mb is not None]
    return ArmSummary(
        arm=arm, max_edge=max_edge, frames=len(good), failed=len(failed),
        input_size=_median_size([r.input_size for r in good]),
        output_size=_median_size([r.output_size for r in good]),
        weight_bytes=_median([float(r.weight_bytes) for r in good]),
        seconds=_median([r.seconds for r in good]),
        peak_vram_mb=max(peaks) if peaks else None,
        truth_kept=_median([truth_kept(r.source_edge, r.max_edge) for r in good]),
        error=failed[0].error if failed and not good else None)


def summaries(runs: list[FrameRun], arms: list[Arm]) -> list[ArmSummary]:
    """The original first, then one row per arm — the order the table is read in."""
    out: list[ArmSummary] = []
    for arm in arms:
        rows = [r for r in runs if r.arm == arm.name]
        if rows:
            out.append(summarize(rows))
    return out


def _size(size: tuple[int, int]) -> str:
    return f"{size[0]}x{size[1]}" if size[0] and size[1] else "—"


def _mb(value: float) -> str:
    return f"{value / (1024 * 1024):.2f} МБ" if value else "—"


def format_degradation_table(kind: str, runs: list[FrameRun], arms: list[Arm]) -> str:
    """One degradation, one table: the original first, every instrument under it.

    Separate tables and not one with a column, because these are three questions and not
    three cases of one. A model that repairs smear and does nothing for a missed focus is
    the second of the brief's three outcomes, and it is only visible here.
    """
    rows = summaries(runs, arms)
    frames = max((r.frames + r.failed for r in rows), default=0)
    out = [
        "=" * 92,
        f"ДЕГРАДАЦИЯ «{DEGRADATION_LABEL[kind]}»: кадров {frames}",
        f"{'инструмент':>14} {'вход':>12} {'выход':>12} {'вес':>10} {'время':>9} "
        f"{'пик VRAM':>10} {'правды':>8}",
    ]
    if not rows:
        out += ["  (в выборке нет кадров этого типа)", "=" * 92]
        return "\n".join(out)
    for row in rows:
        if row.error is not None:
            out.append(f"{row.arm:>14} {'—':>12} {'—':>12} не получилось: {row.error}")
            continue
        vram = f"{row.peak_vram_mb:.0f} МБ" if row.peak_vram_mb is not None else "—"
        seconds = "—" if row.arm == ARM_ORIGINAL else f"{row.seconds:.2f} с"
        out.append(
            f"{row.arm:>14} {_size(row.input_size):>12} {_size(row.output_size):>12} "
            f"{_mb(row.weight_bytes):>10} {seconds:>9} {vram:>10} {row.truth_kept:>7.0%}")
        if row.failed:
            out.append(f"{'':>14} из них не получилось: {row.failed}")
    if frames < MIN_PER_TYPE:
        out.append(f"  ВНИМАНИЕ: {frames} кадров — это не ответ про этот тип, "
                   f"это анекдот (нужно хотя бы {MIN_PER_TYPE})")
    out.append("=" * 92)
    return "\n".join(out)


def format_degradation_tables(runs: list[FrameRun], arms: list[Arm]) -> str:
    """A table per degradation. `unknown` appears only when something landed in it —
    an empty table for a bucket that exists to catch accidents is noise."""
    blocks = []
    for kind in DEGRADATIONS:
        rows = [r for r in runs if r.degradation == kind]
        if kind == UNKNOWN and not rows:
            continue
        blocks.append(format_degradation_table(kind, rows, arms))
    return "\n".join(blocks)


# --- the blind pairs ----------------------------------------------------------------------


def write_blind_pairs(pairs: list[tuple[FrameRun, Image.Image, Image.Image]],
                      out_dir: Path, seed: int) -> list[dict]:
    """Write the sheets for every (run, original, processed) and return the key.

    Numbered, never named after the frame: a basename identifies somebody's photograph in a
    folder meant to be looked through, and a name saying which model made a copy would end
    the blindness before the first sheet is open.

    THE ORDER IS SHUFFLED ACROSS ARMS, which is the one thing this needs beyond F169's
    sheets. Here two instruments run on the SAME frames, so pairs laid out in the order
    they were produced would alternate — and after three sheets a person would know which
    arm they are looking at and would be scoring a model instead of a picture.

    Two sheets where two are possible: `pair_NN.jpg` is the whole frame, which is how a
    person meets a photograph, and `pair_NN_crop.jpg` the same middle of both at native
    scale, where a difference in real detail has nowhere to hide. Whether the copy came
    back at its original size is exactly the question here, so the crop is written whenever
    it did — for both arms, since the x4 arm also returns about the size it was given.
    The flip is the SAME for both sheets of a pair.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    shuffled = list(pairs)
    rng.shuffle(shuffled)
    key: list[dict] = []
    for n, (run, original, processed) in enumerate(shuffled, start=1):
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
            "arm": run.arm,
            "degradation": run.degradation,
        })
    return key


def format_verdict_prompt(out_dir: Path) -> str:
    """The block that hands the decision back to a person, with the question stated once.

    THE QUESTION IS FIDELITY. Not "which is better", not "which is sharper" — twice in one
    day a substituted question gave the wrong answer, in both directions. A soft frame that
    a person kept is a good photograph, and detail invented into it is damage that looks
    like a repair.
    """
    return "\n".join([
        "ВЕРДИКТ ЗАПИСЫВАЕТ ЧЕЛОВЕК, А НЕ СКРИПТ.",
        f"Слепые пары лежат в {out_dir}: две картинки в одном размере, без подписей,",
        "порядок листов перемешан между инструментами.",
        "pair_NN.jpg — кадр целиком; pair_NN_crop.jpg — та же середина обеих картинок",
        "в масштабе 1:1 (на уменьшенном листе разница в настоящей детализации не видна).",
        "Посмотрите их ДО того, как откроете key.json — ключ там же.",
        "",
        "ВОПРОС ПО КАЖДОЙ ПАРЕ ОДИН: какая из двух ближе к тому, что было?",
        "Не «резче» и не «лучше»: кадр хороший, и выдуманная деталь его портит. Метрика",
        "здесь не судья — испортить кадр и получить и «резче», и высокий PSNR одинаково",
        "легко.",
        "",
        "Что означает ответ (три исхода из брифа F183):",
        "  ВЕРНЕЕ оригинала на всех типах и цена линейная -> меняем x4 на неё для",
        "     кадров больше потолка; realworld-sr-x4 остаётся на мелких, где он и выбран",
        "     замером. Два инструмента по назначению, а не один на всё;",
        "  ВЕРНЕЕ только на смазе движения -> оставляем опцией с честной подписью, что",
        "     именно она чинит. Половина популяции лучше, чем ничего, если сказано вслух;",
        "  НЕ ВЕРНЕЕ -> записываем вердикт и закрываем тему; F168 выпускает кнопку",
        "     только для мелких кадров, а для полноразмерных действие честно недоступно.",
        "",
        "Разбиение по типам деградации — эвристика (направленность градиентов плюс",
        "резкость), тип каждой пары записан в key.json: если по картинкам видно, что",
        "он назван неверно, правьте разбиение, а не вывод.",
    ])


# --- the run -------------------------------------------------------------------------


def build_arms(candidates: dict[str, RestoreFn], baseline: RestoreFn | None,
               ceiling: int) -> list[Arm]:
    """The original, then every usable candidate at native size, then the shipped x4."""
    arms = [Arm(name=ARM_ORIGINAL, max_edge=None)]
    arms += [Arm(name=name, max_edge=None, process=process)
             for name, process in candidates.items()]
    if baseline is not None:
        arms.append(Arm(name=ARM_BASELINE, max_edge=ceiling, process=baseline))
    return arms


def short_name(model_name: str) -> str:
    """`owner/weights-name` -> the part a table column has room for."""
    tail = model_name.rsplit("/", 1)[-1]
    return tail[:14]


def main() -> int:  # pragma: no cover — needs the weights and a collection
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--models", nargs="+", required=True,
                    help="candidate 1:1 restoration weights (deblur / denoise / artifacts). "
                         "The brief names NONE on purpose — name them here")
    ap.add_argument("--sample", type=int, default=12, help="frames to measure (default 12)")
    ap.add_argument("--seed", type=int, default=20260804)
    ap.add_argument("--megapixels", nargs="+", type=float, default=list(COST_MEGAPIXELS),
                    help="frame sizes for the price table")
    ap.add_argument("--no-baseline", action="store_true",
                    help="skip the current x4 model (it is the baseline — think twice)")
    ap.add_argument("--out", default="measure_deblur",
                    help="where the blind pairs and key.json go (default measure_deblur)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ceiling = int(cfg.features.restore_max_edge)

    candidates: dict[str, RestoreFn] = {}
    for model_name in args.models:
        print(f"кандидат {model_name}: грузим...")
        try:
            process = load_restorer(model_name)
        except Exception as exc:  # noqa: BLE001 — a candidate that will not load is a row
            print(f"  не загрузилась: {type(exc).__name__}: {exc}")
            continue
        probe = probe_one_to_one(process)
        if not probe.usable:
            print(f"  ОТКЛОНЕНА: {probe.reason}")
            continue
        print(f"  один к одному, кадр меняет (среднее отклонение {probe.changed:.1f}/255)")
        candidates[short_name(model_name)] = process
    if not candidates:
        raise SystemExit("ни один кандидат не подошёл — мерить нечего")

    baseline = None
    if not args.no_baseline:
        baseline = restore.shared_upscaler(cfg.features.restore_model)
    arms = build_arms(candidates, baseline, ceiling)

    print()
    print(format_cost_table([cost_row(arm, mpx) for arm in arms if not arm.is_original
                             for mpx in args.megapixels]))

    conn = sqlite3.connect(f"file:{cfg.database}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        print(format_reach(slice_reach(conn, cfg.features, ceiling), ceiling))
    finally:
        conn.close()

    frames = sample_frames(str(cfg.database), cfg.features, ceiling, args.sample, args.seed)
    if not frames:
        raise SystemExit("в срезе размытых нет кадров больше потолка — мерить нечего")
    print(f"выборка: {len(frames)} кадров размытого среза больше потолка")

    runs, pairs = measure(arms, frames, blur_max=float(cfg.features.blur_review_max),
                          sharpness_edge=int(cfg.features.sharpness_max_edge))
    print()
    print(format_degradation_tables(runs, arms))

    out_dir = Path(args.out)
    key = write_blind_pairs(pairs, out_dir, args.seed)
    (out_dir / "key.json").write_text(json.dumps(key, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    print()
    print(format_verdict_prompt(out_dir))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
