"""Price F206 on a sample: the animal check and the rescue, serial against pipelined.

The regression this measures came out of the run of 2026-08-05, where the two questions of
the junk stage's back half cost 0.42 frames/s against the deep tier's pipelined 1.4 over
the same model — 116 minutes a run over 4 281 frames — because `_frame_question` was the
plain serial path while the tier next to it overlapped its halves (F101).

THE ACCEPTANCE IS ON A SAMPLE AND NOT ON A COLLECTION. The difference between 0.42 and 1.4
frames/s is visible over a few hundred frames (12 minutes against 3.5 at 300), so this asks
for 200-400 and nothing more; the number that goes into the documentation comes from the
owner's next full run and is not a condition of the feature.

Three columns decide it, and the third is the one that outranks the other two:

    frames/s     the rate of the arm — `serial` is the pre-F206 path (one worker, both
                 halves on this thread), `pipeline` is `vlm.workers` preparation threads
    peak VRAM    printed for BOTH arms, because the pipeline holds more prepared frames at
                 once and a mode that wants noticeably more memory is a separate
                 conversation. The prepared tensors are CPU tensors (naming.qwen_runtime),
                 so the expectation is that this column does not move
    answers       how many frames answered differently between the two arms. A change of
                 SCHEDULE may not move a verdict, so anything but zero here is a stop and a
                 look, not a price paid for speed

Nothing is reimplemented: the askers are the stage's own (`junk.vlm_pet_asker`,
`junk.vlm_junk_rescue_asker` over `naming.shared_vlm`) and both arms run through
`junk._vlm_labels`, which is the function the pipeline lives in. A private copy would
price a pass nobody runs.

Privacy: nothing printed identifies a frame — no path, no basename, no file id. The
candidates are read out of the same columns the stage gates on, and the database is opened
`mode=ro`: a measurement writes nothing.

Usage (from the repo root, with the venv python, on the machine whose card is free):
    python scripts/measure_asker_pipeline.py --sample 300
    python scripts/measure_asker_pipeline.py --question rescue --workers 4
"""
from __future__ import annotations

import argparse
import random
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sorta import junk  # noqa: E402
from sorta.config import Config, load_config  # noqa: E402
from sorta.naming import shared_vlm  # noqa: E402

# What the pipeline has to buy before the change is worth its reading. Well below the ~x3
# the live numbers project, because the point of the threshold is to separate a working
# overlap from noise on somebody else's machine, not to re-measure the card.
MIN_SPEEDUP = 1.5

# The two questions of the back half, with the column each of them gates on. Both are
# `frame_quality` columns written by the run whose candidates this samples, which is what
# makes the sample the REAL population rather than an arbitrary list of photographs.
QUESTIONS = {
    "pets": ("pet_score", "pet_candidate_threshold", 0.3),
    "rescue": ("junk_score", "junk_rescue_threshold", 0.02),
}


@dataclass(frozen=True)
class ArmRow:
    """One arm over the sample: what it cost and what it answered."""
    name: str
    workers: int
    frames: int
    seconds: float
    answers: list[str]
    errors: int
    peak_vram_mb: float | None

    @property
    def rate(self) -> float:
        return self.frames / self.seconds if self.seconds else 0.0


def sample_paths(db_path: str, question: str, threshold: float, n: int,
                 seed: int) -> list[str]:
    """`n` paths of the real candidate population of `question`, shuffled by `seed`.

    The gate replayed here is the stage's own — a candidate is a frame whose score REACHES
    the threshold, inside the population that has a `frame_quality` row at all (F120: the
    personal photographs) — so the sample cannot be a population the stage never asks about.
    """
    column = QUESTIONS[question][0]
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            f"""SELECT f.path FROM files f
                JOIN frame_quality fq ON fq.file_id = f.id
                WHERE f.dup_of IS NULL AND f.error IS NULL AND f.media_type = 'photo'
                  AND fq.{column} IS NOT NULL AND fq.{column} >= ?
                ORDER BY f.id""", (threshold,)).fetchall()
    finally:
        conn.close()
    paths = [r["path"] for r in rows]
    random.Random(seed).shuffle(paths)
    return [p for p in paths if Path(p).exists()][:n]


def _vram_peak_mb(reset: bool = False) -> float | None:  # pragma: no cover — needs CUDA
    """Peak VRAM reserved by torch since the last reset, MB (None — no CUDA).

    Printed for both arms because that is what the brief asks of this change: the pipeline
    keeps more frames in flight, and if the memory that costs is not what F101 argued (the
    prepared tensors stay on the CPU, so the peak is one frame's inputs), the mode needs
    discussing rather than shipping.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        if reset:
            torch.cuda.reset_peak_memory_stats()
            return None
        return float(torch.cuda.max_memory_reserved()) / (1024 * 1024)
    except Exception:  # noqa: BLE001 — a measurement of a missing card is not an error
        return None


def build_asker(cfg: Config, question: str):  # pragma: no cover — ML
    """The stage's own asker over the shared runtime — the halves included when it has them."""
    runtime = shared_vlm(cfg.vlm.model)
    if question == "pets":
        return junk.vlm_pet_asker(runtime, max_edge=cfg.vlm.max_edge)
    return junk.vlm_junk_rescue_asker(runtime, max_edge=cfg.vlm.max_edge)


def run_arm(asker, paths: list[str], workers: int,
            name: str) -> ArmRow:  # pragma: no cover — ML
    """One pass over the sample through `junk._vlm_labels` — the stage's own loop.

    `workers=1` is the serial arm and it is not a simulation of the old code: that is
    literally the path `_vlm_labels` takes when the runtime has no halves or the pool is
    one thread, i.e. what every one of these questions did before F206.
    """
    _vram_peak_mb(reset=True)
    answers: list[str] = []
    errors = 0
    started = time.perf_counter()
    for item in junk._vlm_labels(asker, paths, workers):
        if isinstance(item, BaseException):
            errors += 1
            answers.append("")   # what a frame the model failed on has always contributed
        else:
            answers.append(item)
    seconds = time.perf_counter() - started
    return ArmRow(name=name, workers=workers, frames=len(paths), seconds=seconds,
                  answers=answers, errors=errors, peak_vram_mb=_vram_peak_mb())


def disagreements(before: ArmRow, after: ArmRow, question: str) -> int:
    """How many frames the two arms PARSED differently — the criterion that outranks speed.

    Parsed and not raw: what reaches the database is the parser's answer, and two runs of a
    generative model may word one answer differently while meaning the same thing. A
    difference here is a difference in a stored label.
    """
    parse = (junk.parse_pet_answer if question == "pets"
             else junk.parse_junk_rescue_answer)
    return sum(1 for a, b in zip(before.answers, after.answers)
               if parse(a) != parse(b))


def format_table(rows: list[ArmRow], moved: int) -> str:
    """The table the acceptance is read off — rates, VRAM before and after, disagreements."""
    base = rows[0]
    out = [
        "=" * 88,
        f"КОНВЕЙЕР ВОПРОСОВ: {base.frames} кадров на строку, база — "
        f"{base.workers} поток(а/ов)",
        f"{'арм':>10} {'потоков':>8} {'секунд':>9} {'кадров/с':>10} {'ускорение':>11} "
        f"{'ошибок':>8} {'пик VRAM':>11}",
    ]
    for r in rows:
        gain = f"x{r.rate / base.rate:.2f}" if base.rate and r.rate else "—"
        vram = f"{r.peak_vram_mb:.0f} МБ" if r.peak_vram_mb is not None else "—"
        out.append(f"{r.name:>10} {r.workers:>8d} {r.seconds:>9.1f} {r.rate:>10.2f} "
                   f"{gain:>11} {r.errors:>8d} {vram:>11}")
    out.append("=" * 88)
    out.append(f"вердикты разошлись: {moved} из {base.frames}")
    return "\n".join(out)


def outcome(rows: list[ArmRow], moved: int) -> str:
    """The pre-registered answer: identical answers first, then the rate."""
    before, after = rows[0], rows[-1]
    if moved:
        return (f"СТОП: {moved} кадров ответили по-разному. Расписание работы не меняет "
                "вопрос — расхождение надо разобрать, а не усреднить")
    gain = after.rate / before.rate if before.rate else 0.0
    if gain >= MIN_SPEEDUP:
        return (f"принято: x{gain:.2f} ({before.rate:.2f} -> {after.rate:.2f} кадров/с), "
                f"вердикты совпали кадр в кадр")
    return (f"не принято: x{gain:.2f} при пороге x{MIN_SPEEDUP:.2f} — перекрытия не видно, "
            f"смотреть, отдаёт ли рантайм половины (naming.SplitVlm)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--question", choices=sorted(QUESTIONS), default="pets")
    ap.add_argument("--sample", type=int, default=300,
                    help="frames per arm (the brief asks for 200-400)")
    ap.add_argument("--workers", type=int, default=0,
                    help="preparation threads of the pipelined arm (default: vlm.workers)")
    ap.add_argument("--seed", type=int, default=20260805)
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    q = junk.quality_settings(cfg)
    _column, key, fallback = QUESTIONS[args.question]
    threshold = float(getattr(q, key, fallback))
    paths = sample_paths(str(cfg.database), args.question, threshold,
                         args.sample, args.seed)
    if not paths:
        print(f"нет кандидатов: ни одного кадра с {_column} >= {threshold:.3f} — "
              f"сначала прогон стадии junk, потом это измерение")
        return 1

    workers = args.workers or cfg.vlm.workers
    asker = build_asker(cfg, args.question)
    rows = [run_arm(asker, paths, 1, "serial"),
            run_arm(asker, paths, max(2, workers), "pipeline")]
    moved = disagreements(rows[0], rows[1], args.question)
    print(format_table(rows, moved))
    print(outcome(rows, moved))
    return 0


if __name__ == "__main__":  # pragma: no cover — CLI
    raise SystemExit(main())
