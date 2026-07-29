"""Price the frame-quality cascade (F113): what the cheap tiers see, before any threshold.

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

Usage (from the repo root, with the venv python):
    python scripts/measure_frame_quality.py --features pets
    python scripts/measure_frame_quality.py --features pets sharpness band
    python scripts/measure_frame_quality.py --sample 500 --cache frame_quality.json
    python scripts/measure_frame_quality.py --cache frame_quality.json   # replay, no CLIP

A run costs one CLIP pass over the sample (the pet prompts ride in the same call the junk
stage makes) plus a preview decode per frame for the laplacian — minutes on a few hundred
frames. `--cache` writes the per-frame aggregates out so a different grid can be tried
without paying for the model again.
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
# Percentiles are how a laplacian distribution is read: it has no natural units, so what
# matters is where the mass sits relative to itself.
PERCENTILES = (5, 10, 25, 50, 75, 90, 95)

FEATURES = ("pets", "sharpness", "band")

CACHE_VERSION = 1


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


def format_sharpness(frames: list[Frame], q: junk.QualitySettings) -> str:
    """The sharpness block: the distribution the band has to be read off."""
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
    out.append("=" * 88)
    out.append("Число зависит от разрешения (features.sharpness_max_edge) — меняя его, "
               "полосу\nнужно перемерить. Ниже полосы кадр смазан, выше — резкий; "
               "внутри решает VLM.")
    return "\n".join(out)


def format_band(frames: list[Frame], q: junk.QualitySettings) -> str:
    """The band block: how many frames the VLM would be asked about, and why.

    Split by REASON because the two conditions are independent knobs: if the whole band is
    the subject condition, moving the sharpness numbers changes nothing.
    """
    total = len(frames)
    low, high = q.sharpness_band
    by_sharp = sum(1 for f in frames
                   if f.sharpness is not None and low <= f.sharpness <= high)
    by_subject = sum(1 for f in frames if f.subject_score < q.subject_score_min)
    inside = sum(1 for f in frames if junk.uncertain_band(f.sharpness, f.subject_score, q))
    seconds = inside * 0.78  # the measured per-frame cost of the local VLM
    return "\n".join([
        "=" * 88,
        f"НЕУВЕРЕННАЯ ПОЛОСА (популяция VLM, {total} кадров)",
        f"  по резкости ({low:.1f}..{high:.1f}):       {by_sharp:>6d} "
        f"({_pct(by_sharp, total)})",
        f"  по сюжету (CLIP < {q.subject_score_min:.2f}):     {by_subject:>6d} "
        f"({_pct(by_subject, total)})",
        f"  итого в полосе (ИЛИ):          {inside:>6d} ({_pct(inside, total)})",
        f"  цена на этой выборке: ~{seconds:.0f} с (0.78 с/кадр); "
        f"в пересчёте на 20 000 кадров — "
        f"~{20000 * (inside / total if total else 0) * 0.78 / 3600:.1f} ч",
        "=" * 88,
        "Полоса сужается ещё и областью (vlm.quality_scope): по умолчанию только кадры "
        "phash-групп.\nЗдесь она не учтена — это верхняя оценка.",
    ])


def save_cache(path: Path, frames: list[Frame]) -> None:
    """Per-frame aggregates for a later replay. File ids only — never paths."""
    path.write_text(json.dumps({
        "version": CACHE_VERSION,
        "frames": [[f.file_id, f.pet_class, f.pet_score, f.sharpness, f.subject_score]
                   for f in frames],
    }), encoding="utf-8")


def load_cache(path: Path) -> list[Frame]:
    """A cache of another version is an error, not a silently wrong table."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("version") != CACHE_VERSION:
        raise SystemExit(f"{path}: кэш версии {data.get('version')}, ожидается "
                         f"{CACHE_VERSION} — перемерить с --refresh")
    return [Frame(int(fid), pet, float(score),
                  None if sharp is None else float(sharp), float(subject))
            for fid, pet, score, sharp, subject in data["frames"]]


def measure(cfg: Config, rows: list[sqlite3.Row],
            q: junk.QualitySettings) -> list[Frame]:  # pragma: no cover — ML
    """One CLIP pass (junk prompts + the pet group, as the stage makes it) + the laplacian.

    `clip_prompts(pets=True)` on purpose even when the toggle is off in the config: the
    whole point is to see what the pet group would say before deciding to switch it on.
    """
    s = naming_settings(cfg)
    classifier = clip_classifier(s)
    prompts = junk.clip_prompts(True)
    sharpness = junk.preview_sharpness_detector(q.sharpness_max_edge)

    frames: list[Frame] = []
    done = 0
    for chunk in batched(rows, s.clip_batch_size):
        probs = classifier([r["path"] for r in chunk], prompts)
        for r, p in zip(chunk, probs):
            # threshold 0.0: the winning class always comes back, and which thresholds
            # would have kept it is what the sweep is for.
            pet_class, pet_score = junk.pet_verdict(p, 0.0)
            subject = float(junk._group_probs(p, junk._JUNK_GROUP)[0])
            frames.append(
                Frame(r["id"], pet_class, pet_score, sharpness(r["path"]), subject))
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

    print()
    if "pets" in args.features:
        print(format_pets(frames, sweep_pets(frames, sorted(args.pet_thresholds)),
                          q.pet_threshold))
    if "sharpness" in args.features:
        print(format_sharpness(frames, q))
    if "band" in args.features:
        print(format_band(frames, q))


if __name__ == "__main__":
    main()
