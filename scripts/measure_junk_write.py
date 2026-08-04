"""Price the `junk_write` phase: what the 19.4 ms per frame is actually spent on.

F147 printed the phases of the junk stage for the first time, and one line of that
table asked for this script (2026-08-03, 38 485 files):

    junk_write   470,3 s   24 196 frames   19,4 ms/frame   the whole collection

19.4 ms per ROW of SQLite would be extraordinary — CLIP on the GPU costs 11.7 ms per
frame in the same stage — and the usual cause of a number like that is a commit per
row. So the first thing this measures is exactly that: the same `media_class` upsert
the stage runs, under the three commit strategies (one transaction for the whole pass,
one per chunk, one per row), on a real database file.

But `junk_write` is not only the upsert. The phase is entered before the per-frame loop
that ALSO measures the laplacian (`_QualityPass.measure`) and stores the CLIP vector
(`_EmbeddingPass.store`) — junk.py says so where the phase is entered — so the second
half of this script prices those on real frames of the index. A lever is only worth
pulling if the seconds are behind it, and only a breakdown can say which one is.

Nothing here writes to the collection's database: the SQLite half runs against a
throwaway file (`--db-dir`, default the system temp), and the frame half opens the
index READ-ONLY and only reads paths out of it.

Privacy: nothing here identifies a frame. No path, no file id and no basename is
printed — the same rule measure_ocr_gate.py and measure_vlm_speed.py follow.

Usage (from the repo root, with the venv python):
    python scripts/measure_junk_write.py                    # 24 196 rows, no frames
    python scripts/measure_junk_write.py --frames 200       # + the laplacian, from config.yaml
    python scripts/measure_junk_write.py --rows 5000 --chunk 500
"""
from __future__ import annotations

import argparse
import random
import sqlite3
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sorta import junk  # noqa: E402
from sorta.config import load_config  # noqa: E402
from sorta.db import connect  # noqa: E402

# The population of the measurement that ordered this script: the frames `junk_write`
# walked on the live run of 2026-08-03.
DEFAULT_ROWS = 24196
# The chunk of the stage is `naming.clip.batch_size` (16 by default) — the size a
# "commit per batch" would really have, so it is the one priced here.
DEFAULT_CHUNK = 16
# ViT-L-14's width — the vector `_EmbeddingPass.store` packs and writes per frame.
EMBEDDING_DIM = 768


@dataclass(frozen=True)
class WriteRow:
    """One commit strategy over the same rows: total seconds and the per-row cost."""
    name: str
    rows: int
    seconds: float
    commits: int

    @property
    def ms_per_row(self) -> float:
        return 1000.0 * self.seconds / self.rows if self.rows else 0.0


@dataclass(frozen=True)
class FrameRow:
    """One per-frame cost of the phase, measured on real files."""
    name: str
    frames: int
    seconds: float

    @property
    def ms_per_frame(self) -> float:
        return 1000.0 * self.seconds / self.frames if self.frames else 0.0


def _prepare_db(path: Path, rows: int) -> sqlite3.Connection:
    """A throwaway index with `rows` file rows — media_class references files(id)."""
    conn = connect(path)
    now = junk.utcnow_iso()
    with conn:
        conn.executemany(
            "INSERT INTO files (id, path, size, mtime, ext, media_type, indexed_at)"
            " VALUES (?, ?, 1, 1.0, '.jpg', 'photo', ?)",
            [(i, f"/measure/{i}.jpg", now) for i in range(1, rows + 1)])
    return conn


def measure_upsert(conn: sqlite3.Connection, rows: int, chunk: int,
                   mode: str) -> WriteRow:
    """The stage's own `media_class` upsert over `rows`, committed the way `mode` says.

    `stage` is what junk.classify does today — ONE transaction around the whole pass
    (`with conn:` outside the chunk loop, sqlite3's implicit BEGIN on the first write).
    `chunk` commits every `chunk` rows, which is what "write in batches" would mean
    here. `row` is the hypothesis this script was written to test: a commit per row.
    """
    now = junk.utcnow_iso()
    params = [(i, "photo", "clip", 0.9, now, "clip") for i in range(1, rows + 1)]
    commits = 0
    started = time.perf_counter()
    if mode == "stage":
        with conn:
            for p in params:
                conn.execute(junk._MEDIA_CLASS_UPSERT, p)
        commits = 1
    elif mode == "chunk":
        for start in range(0, len(params), chunk):
            with conn:
                for p in params[start:start + chunk]:
                    conn.execute(junk._MEDIA_CLASS_UPSERT, p)
            commits += 1
    else:
        for p in params:
            with conn:
                conn.execute(junk._MEDIA_CLASS_UPSERT, p)
            commits += 1
    seconds = time.perf_counter() - started
    return WriteRow(name=mode, rows=rows, seconds=seconds, commits=commits)


def measure_embedding_write(conn: sqlite3.Connection, rows: int) -> WriteRow:
    """`_EmbeddingPass.store`: pack a vector and upsert it, inside the stage transaction.

    The second write of the phase, and the one that carries a payload — 768 float32 per
    frame — so it is priced next to the verdict rather than assumed to be free.
    """
    rng = np.random.default_rng(20260803)
    vecs = [rng.standard_normal(EMBEDDING_DIM).astype(np.float32) for _ in range(rows)]
    now = junk.utcnow_iso()
    started = time.perf_counter()
    with conn:
        for i, vec in enumerate(vecs, start=1):
            conn.execute(junk._EMBEDDING_UPSERT,
                         (i, "measure", EMBEDDING_DIM, junk.pack_embedding(vec), now))
    seconds = time.perf_counter() - started
    return WriteRow(name="clip_embeddings", rows=rows, seconds=seconds, commits=1)


def sample_paths(db_path: str, n: int, seed: int) -> list[str]:
    """`n` canonical photographs that still exist on disk — read-only, paths only."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = [r[0] for r in conn.execute(
            "SELECT path FROM files WHERE dup_of IS NULL AND error IS NULL"
            " AND media_type = 'photo' ORDER BY id")]
    finally:
        conn.close()
    random.Random(seed).shuffle(rows)
    return [p for p in rows if Path(p).exists()][:n]


def measure_sharpness(paths: list[str], max_edge: int) -> FrameRow:
    """`_QualityPass.measure`'s own cost: one decode of the preview plus two laplacians.

    Through the stage's detector, not a copy of it — the preview cache is part of the
    price and a private re-implementation would time something nobody runs.
    """
    detector = junk.preview_sharpness_detector(max_edge)
    started = time.perf_counter()
    for path in paths:
        detector(path)
    return FrameRow(name="sharpness (laplacian)", frames=len(paths),
                    seconds=time.perf_counter() - started)


def format_write_table(rows: list[WriteRow], chunk: int) -> str:
    """The before/after table the brief asks for: commit strategy -> ms per row."""
    base = next((r for r in rows if r.name == "stage"), rows[0])
    out = [
        "=" * 88,
        f"ЗАПИСЬ media_class: {base.rows} строк, тот же upsert, что в стадии",
        f"{'стратегия':>28} {'коммитов':>10} {'секунд':>10} {'мс/строку':>12} "
        f"{'против stage':>14}",
    ]
    names = {
        "stage": "одна транзакция (сейчас)",
        "chunk": f"коммит на {chunk} строк",
        "row": "коммит на строку",
    }
    for r in rows:
        ratio = f"x{r.ms_per_row / base.ms_per_row:.1f}" if base.ms_per_row else "—"
        out.append(f"{names.get(r.name, r.name):>28} {r.commits:>10d} "
                   f"{r.seconds:>10.2f} {r.ms_per_row:>12.3f} {ratio:>14}")
    out.append("=" * 88)
    return "\n".join(out)


def format_frame_table(rows: list[FrameRow], phase_ms: float) -> str:
    """What one frame of the phase costs, item by item, against the measured 19.4 ms."""
    out = [
        "=" * 88,
        f"ФАЗА junk_write ПО СЛАГАЕМЫМ (замер стадии 2026-08-03: {phase_ms:.1f} мс/кадр)",
        f"{'слагаемое':>28} {'кадров':>10} {'секунд':>10} {'мс/кадр':>12} "
        f"{'доля фазы':>14}",
    ]
    for r in rows:
        share = f"{100.0 * r.ms_per_frame / phase_ms:.0f}%" if phase_ms else "—"
        out.append(f"{r.name:>28} {r.frames:>10d} {r.seconds:>10.2f} "
                   f"{r.ms_per_frame:>12.3f} {share:>14}")
    out.append("=" * 88)
    return "\n".join(out)


def verdict(rows: list[WriteRow]) -> str:
    """The one sentence the brief wants recorded next to the numbers.

    Batching is worth doing only if the rows say the writes are NOT batched today. The
    stage already wraps its whole pass in one transaction, so the expected outcome is
    "nothing to batch" — and then the reason has to be written down, so the next person
    does not spend a day on the same idea.
    """
    stage = next((r for r in rows if r.name == "stage"), None)
    per_row = next((r for r in rows if r.name == "row"), None)
    if stage is None or per_row is None:
        return "сравнивать не с чем — запрошена одна стратегия"
    if stage.ms_per_row >= 1.0:
        return (f"запись действительно дорогая: {stage.ms_per_row:.2f} мс/строку даже "
                f"в одной транзакции — пачки тут ни при чём, искать причину дальше")
    return (f"вставка стоит {stage.ms_per_row:.3f} мс/строку в одной транзакции "
            f"(коммит на строку — {per_row.ms_per_row:.3f} мс, x"
            f"{per_row.ms_per_row / stage.ms_per_row:.0f}); стадия УЖЕ пишет одной "
            f"транзакцией, так что 19,4 мс/кадр фазы junk_write — это не SQLite")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml",
                    help="only needed with --frames (the index and sharpness_max_edge)")
    ap.add_argument("--rows", type=int, default=DEFAULT_ROWS,
                    help=f"rows per commit strategy (default {DEFAULT_ROWS})")
    ap.add_argument("--chunk", type=int, default=DEFAULT_CHUNK,
                    help=f"rows per commit in the `chunk` strategy (default {DEFAULT_CHUNK})")
    ap.add_argument("--frames", type=int, default=0, metavar="N",
                    help="also price the laplacian on N real frames of the index")
    ap.add_argument("--seed", type=int, default=20260803)
    ap.add_argument("--db-dir", help="where the throwaway database goes (default: temp)")
    ap.add_argument("--phase-ms", type=float, default=19.4,
                    help="the measured cost of the phase the breakdown is compared with")
    args = ap.parse_args()

    if args.rows < 1:
        raise SystemExit("--rows: строк не меньше одной")

    with tempfile.TemporaryDirectory(dir=args.db_dir) as tmp:
        db = Path(tmp) / "measure_junk_write.db"
        print(f"временная база: {db.name} ({args.rows} строк files)")
        conn = _prepare_db(db, args.rows)
        try:
            rows = [measure_upsert(conn, args.rows, args.chunk, mode)
                    for mode in ("stage", "chunk", "row")]
            embeddings = measure_embedding_write(conn, args.rows)
        finally:
            conn.close()

    print()
    print(format_write_table(rows, args.chunk))
    print(f"ИТОГ: {verdict(rows)}")

    frame_rows = [FrameRow(name="media_class (upsert)", frames=args.rows,
                           seconds=rows[0].seconds),
                  FrameRow(name="clip_embeddings (768f)", frames=embeddings.rows,
                           seconds=embeddings.seconds)]
    if args.frames:
        cfg = load_config(args.config)
        paths = sample_paths(str(cfg.database), args.frames, args.seed)
        if not paths:
            print("кадров в индексе нет — слагаемое резкости не измерено")
        else:
            q = junk.quality_settings(cfg)
            print(f"резкость: {len(paths)} кадров, max_edge {q.sharpness_max_edge}...")
            frame_rows.append(measure_sharpness(paths, q.sharpness_max_edge))
    print()
    print(format_frame_table(frame_rows, args.phase_ms))
    total = sum(r.ms_per_frame for r in frame_rows)
    print(f"сумма слагаемых: {total:.2f} мс/кадр из {args.phase_ms:.1f} мс/кадр фазы")
    return 0


if __name__ == "__main__":
    sys.exit(main())
