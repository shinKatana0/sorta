"""Price the input resolution of the deep junk tier — and say what lowering it costs.

F101 removed the CPU half of the pass (fast processor + overlapped preparation). What
is left is the GPU half, and what decides the GPU half is the size of the frame the
model sees: the image is cut into tokens, and the token count grows with the area. That
number was `naming.VLM_MAX_EDGE = 896`, a constant in the code, in a project whose rule
is that thresholds live in config.yaml. F102 made it `vlm.max_edge`; this script is why
the knob was worth making.

It runs ONE load of the weights over ONE sample of the tier's own candidate frames at
896 / 672 / 448 and reports, per resolution: median and p90 ms per frame, frames/s, the
speedup against 896, peak VRAM — and the comparison that decides everything, the
verdicts label by label against the 896 baseline.

The criteria are PRE-REGISTERED below, in code, written before the first run: a number
you choose after seeing the table is not a criterion, it is a rationalization. Lowering
the default requires BOTH a speedup worth having and agreement that does not quietly
give up documents.

Privacy: nothing here identifies a frame. No path, no file id and no basename is
printed — disagreements are reported as counts per label pair, which is all that is
needed to go and look, and never a list of where somebody's documents are (the rule of
measure_ocr_gate.py, measure_streetclip.py and measure_vlm_speed.py before it).

Usage (from the repo root, with a GPU venv — `uv sync --extra gpu --extra vlm`):
    python scripts/measure_vlm_resolution.py                       # 300 frames, 896/672/448
    python scripts/measure_vlm_resolution.py --sample 500
    python scripts/measure_vlm_resolution.py --edges 896 640
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))  # the repo root — for `sorta`
sys.path.insert(0, str(_HERE))         # ...and this directory — for the sibling script

from sorta import junk, naming  # noqa: E402
from sorta.config import load_config  # noqa: E402

# The measurement script next door already solved the parts of this that are not about
# resolution: how to pick frames the tier would really classify, how to read the peak
# VRAM off the driver, how to take a p90 that is a frame and not an average of two, how
# to count disagreements without naming a file. Reusing them keeps the two reports
# comparable instead of subtly different — the vram reader is private over there and
# imported anyway, because a second copy of it would be the more expensive mistake.
from measure_vlm_speed import _vram_peak_mb as vram_peak_mb  # noqa: E402
from measure_vlm_speed import label_mismatches, percentile, sample_paths  # noqa: E402

# The baseline first: every comparison below is against it, and it is the value that
# shipped, so "no change" has to be a possible outcome of this run.
DEFAULT_EDGES = (896, 672, 448)

# --- Pre-registered acceptance criteria (F102) -------------------------------
#
# 1. Agreement with the 896 baseline >= 98% on at least 300 frames, AND no systematic
#    loss of the `document` class: at most 2% of the baseline's documents may turn into
#    `photo`. Documents are the frames made of small text, resolution is exactly what
#    reads small text, so that is where a cut hits first and where an average agreement
#    number would hide it.
# 2. A speedup of at least 40%. Less is not worth the risk: F101 already gave ~20% for
#    free, and any disagreement eats into the 10.7% of changed verdicts that are the
#    entire reason the deep tier is switched on.
MIN_SAMPLE = 300
MIN_AGREEMENT = 0.98
MAX_DOCUMENT_LOSS = 0.02
MIN_SPEEDUP = 1.40

# The label the deep tier answers with for a document, and the one a lost document
# turns into (junk._VLM_LABEL_TO_VERDICT maps them to the `document`/`photo` verdicts).
DOCUMENT_LABEL = "document"
PHOTO_LABEL = "personal_photo"


@dataclass(frozen=True)
class EdgeResult:
    """What one pass at one resolution cost, and what it decided."""
    max_edge: int
    labels: tuple[str, ...]
    frame_ms: tuple[float, ...]
    wall_sec: float
    peak_vram_mb: float | None

    @property
    def median_ms(self) -> float:
        return statistics.median(self.frame_ms) if self.frame_ms else 0.0

    @property
    def p90_ms(self) -> float:
        return percentile(self.frame_ms, 0.9)

    @property
    def frames_per_sec(self) -> float:
        return len(self.labels) / self.wall_sec if self.wall_sec else 0.0


def agreement(base: EdgeResult, other: EdgeResult) -> float:
    """The share of frames both passes labelled the same way (1.0 — every one of them).

    A frame missing from either pass counts against agreement: fewer answers is not the
    same as agreeing, and pretending otherwise would reward a resolution that crashed.
    """
    total = max(len(base.labels), len(other.labels))
    if not total:
        return 0.0
    same = sum(1 for a, b in zip(base.labels, other.labels) if a == b)
    return same / total


def document_loss(base: EdgeResult, other: EdgeResult) -> tuple[int, int]:
    """(documents of the baseline that became `personal_photo`, documents in total).

    Directional on purpose. A document that the smaller frame turns into a photo is the
    failure this tier exists to prevent — it lands in somebody's family album. The
    reverse (a photo read as a document) is a different, milder mistake and is already
    counted by `agreement`.
    """
    documents = sum(1 for label in base.labels if label == DOCUMENT_LABEL)
    lost = sum(1 for a, b in zip(base.labels, other.labels)
               if a == DOCUMENT_LABEL and b == PHOTO_LABEL)
    return lost, documents


def speedup(base: EdgeResult, other: EdgeResult) -> float:
    """How many times faster `other` is than the baseline (0.0 — nothing to divide by)."""
    return other.frames_per_sec / base.frames_per_sec if base.frames_per_sec else 0.0


@dataclass(frozen=True)
class Assessment:
    """One candidate resolution measured against the pre-registered criteria."""
    max_edge: int
    speedup: float
    agreement: float
    documents_lost: int
    documents_total: int

    @property
    def document_loss_frac(self) -> float:
        return self.documents_lost / self.documents_total if self.documents_total else 0.0

    @property
    def fast_enough(self) -> bool:
        return self.speedup >= MIN_SPEEDUP

    @property
    def accurate_enough(self) -> bool:
        return (self.agreement >= MIN_AGREEMENT
                and self.document_loss_frac <= MAX_DOCUMENT_LOSS)


def assess(base: EdgeResult, other: EdgeResult) -> Assessment:
    lost, documents = document_loss(base, other)
    return Assessment(max_edge=other.max_edge, speedup=speedup(base, other),
                      agreement=agreement(base, other),
                      documents_lost=lost, documents_total=documents)


def outcome(results: list[EdgeResult]) -> tuple[str, str]:
    """(A | B | C, one line saying why) — the verdict of the brief, decided by the data.

    A — both criteria met by some resolution: lower the `vlm.max_edge` default to the
        fastest such one and put the numbers in the CHANGELOG.
    B — the speedup is there but the agreement is not (or the sample was too small to
        claim it): the default stays, and the knob remains for whoever wants speed more
        than accuracy, with an honest warning in the config comment.
    C — no resolution reaches the speedup: the question of resolution is closed, with
        numbers.
    """
    base = results[0]
    if len(results) < 2:
        return "C", "сравнивать не с чем — запрошено одно разрешение"
    sample_ok = len(base.labels) >= MIN_SAMPLE
    checks = [assess(base, r) for r in results[1:]]
    fast = [c for c in checks if c.fast_enough]
    if not fast:
        best = max(checks, key=lambda c: c.speedup)
        return "C", (f"ускорение нигде не дотянуло до {MIN_SPEEDUP:.2f}x "
                     f"(лучшее — x{best.speedup:.2f} на {best.max_edge}px): "
                     f"вопрос разрешения закрыт, дефолт {base.max_edge} остаётся")
    if not sample_ok:
        return "B", (f"выборка {len(base.labels)} кадров меньше "
                     f"пред-зарегистрированных {MIN_SAMPLE} — согласие вердиктов на ней "
                     f"не считается доказанным, дефолт {base.max_edge} остаётся")
    good = [c for c in fast if c.accurate_enough]
    if not good:
        worst = min(fast, key=lambda c: c.agreement)
        return "B", (f"ускорение есть, но согласие ниже порога "
                     f"(на {worst.max_edge}px — {worst.agreement:.1%} при "
                     f"{MIN_AGREEMENT:.0%}, документов потеряно "
                     f"{worst.document_loss_frac:.1%} при {MAX_DOCUMENT_LOSS:.0%}): "
                     f"дефолт {base.max_edge} не трогаем, ручка остаётся")
    pick = max(good, key=lambda c: c.speedup)
    return "A", (f"оба критерия взяты на {pick.max_edge}px "
                 f"(x{pick.speedup:.2f}, согласие {pick.agreement:.1%}, "
                 f"документов потеряно {pick.document_loss_frac:.1%}): "
                 f"меняем дефолт vlm.max_edge {base.max_edge} -> {pick.max_edge}")


def run_edge(classifier: junk.VlmClassifyFn, paths: list[str], max_edge: int,
             workers: int) -> EdgeResult:
    """One pass over `paths` through the pipeline's own `_vlm_labels`.

    The stage's function, not a copy of it: what is being measured has to be what runs,
    or the table prices something nobody executes (the lesson measure_ocr_gate.py starts
    with).
    """
    vram_peak_mb(reset=True)
    frame_ms: list[float] = []
    labels: list[str] = []
    wall0 = time.perf_counter()
    previous = wall0
    for item in junk._vlm_labels(classifier, paths, workers):
        now = time.perf_counter()
        frame_ms.append((now - previous) * 1000.0)
        previous = now
        labels.append("ERROR" if isinstance(item, BaseException) else item)
    wall = time.perf_counter() - wall0
    return EdgeResult(max_edge=max_edge, labels=tuple(labels), frame_ms=tuple(frame_ms),
                      wall_sec=wall, peak_vram_mb=vram_peak_mb())


def format_table(results: list[EdgeResult]) -> str:
    """The speed table: one row per resolution, everything relative to the first."""
    base = results[0]
    out = [
        "=" * 84,
        f"РАЗРЕШЕНИЕ VLM: {len(base.labels)} кадров, база — {base.max_edge}px",
        f"{'max_edge':>10} {'медиана':>9} {'p90':>9} {'кадр/с':>8} {'ускорение':>10} "
        f"{'пик VRAM':>12}",
    ]
    for r in results:
        gain = f"x{speedup(base, r):.2f}" if base.frames_per_sec else "—"
        vram = f"{r.peak_vram_mb:.0f} МБ" if r.peak_vram_mb is not None else "—"
        out.append(f"{r.max_edge:>10d} {r.median_ms:>8.0f}м {r.p90_ms:>8.0f}м "
                   f"{r.frames_per_sec:>8.2f} {gain:>10} {vram:>12}")
    out.append("=" * 84)
    return "\n".join(out)


def format_verdicts(results: list[EdgeResult]) -> str:
    """The verdict comparison: agreement, documents lost, and the pairs that moved."""
    base = results[0]
    lines = [f"ВЕРДИКТЫ (сверка с {base.max_edge}px, кадров {len(base.labels)}, "
             f"из них документов {sum(1 for lb in base.labels if lb == DOCUMENT_LABEL)}):"]
    for r in results[1:]:
        check = assess(base, r)
        lines.append(
            f"  {r.max_edge:>4}px: согласие {check.agreement:>6.1%} "
            f"(порог {MIN_AGREEMENT:.0%}), документов -> photo "
            f"{check.documents_lost} из {check.documents_total} "
            f"({check.document_loss_frac:.1%}, порог {MAX_DOCUMENT_LOSS:.0%}), "
            f"ускорение x{check.speedup:.2f} (порог x{MIN_SPEEDUP:.2f})")
        for (was, now), count in sorted(label_mismatches(base, r).items(),
                                        key=lambda kv: -kv[1]):
            lines.append(f"{'':>12}{was} -> {now}: {count}")
    if len(results) == 1:
        lines.append("  (сравнивать не с чем — запрошено одно разрешение)")
    return "\n".join(lines)


def format_outcome(results: list[EdgeResult]) -> str:
    letter, why = outcome(results)
    return f"ИСХОД {letter}: {why}"


def measure(model_name: str, paths: list[str], edges: list[int],
            workers: int) -> list[EdgeResult]:  # pragma: no cover — ML
    """Every requested resolution on ONE load of the weights (20.5 GB — a second does
    not fit).

    Only the classifier is rebuilt between passes, and it is rebuilt out of the same
    runtime: the resolution lives in the decode step (junk.vlm_classifier_from), not in
    the model, so nothing about the weights, the prompt or the decode changes between
    the rows of the table.
    """
    model, processor, device = naming.load_qwen(model_name, use_fast=True)
    runtime = naming.qwen_runtime(
        model, processor, device,
        lambda: naming.qwen_processor(model_name, use_fast=True))
    results: list[EdgeResult] = []
    for max_edge in edges:
        print(f"разрешение {max_edge}px: {len(paths)} кадров, потоков {workers}...")
        classifier = junk.vlm_classifier_from(runtime, max_edge=max_edge)
        results.append(run_edge(classifier, paths, max_edge, workers))
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--sample", type=int, default=MIN_SAMPLE,
                    help=f"frames per resolution (default {MIN_SAMPLE} — the "
                         f"pre-registered minimum)")
    ap.add_argument("--seed", type=int, default=20260728)
    ap.add_argument("--edges", nargs="+", type=int, default=list(DEFAULT_EDGES),
                    help="resolutions to compare, the baseline first")
    ap.add_argument("--workers", type=int,
                    help="preparation threads (default: vlm.workers)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    paths, origin = sample_paths(str(cfg.database), args.sample, args.seed)
    if not paths:
        raise SystemExit("нет подходящих кадров в индексе — нечего мерить")
    workers = args.workers or cfg.vlm.workers
    print(f"выборка: {len(paths)} кадров — {origin}")
    print(f"модель: {cfg.vlm.model}, потоков подготовки: {workers}")
    if len(paths) < MIN_SAMPLE:
        print(f"ВНИМАНИЕ: кадров меньше {MIN_SAMPLE} — согласие вердиктов на такой "
              f"выборке не считается доказанным (исход A недоступен)")

    results = measure(cfg.vlm.model, paths, args.edges, workers)
    print()
    print(format_table(results))
    print(format_verdicts(results))
    print()
    print(format_outcome(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
