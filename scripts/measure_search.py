"""How good is search by words on THIS collection? A measurement, not a feature (F129).

The engine ranks photographs against a query by cosine over the vectors of the search
index (F141: `search_embeddings`, filled by `sorta junk` with `features.search_index: true`
— NOT the classification vectors of `clip_embeddings`, which are a different model). What
that ranking is worth has to be measured rather than assumed, and F121/F122 is the reason:
the animal class looked like it worked until 320 hand-labelled frames showed that only
half of the question was right. This script is what measured the 22%-against-98% gap
between the two models on Russian queries in the first place.

So this script prints the numbers a person needs BEFORE anybody names an accuracy:

* the top-N of each query with its scores — the list a human reads down until it stops
  being about their words;
* the distribution of the whole collection over score bands for that query, which is what
  says whether the top is a peak or just the top of a flat pile;
* with a filled-in worksheet (`--write-labels` then `--labels`), the precision at several
  depths and per score band — the numbers themselves.

Nothing is reimplemented: `search.encode_query`, `search.search` and `search.search_text`
are the same functions `sorta search` and the query album run, driven off the same config.
A private copy of the arithmetic would measure this script instead of the feature.

Privacy: file ids and scores by default — no path, no basename (the rule of
measure_frame_quality.py and measure_ocr_gate.py). `--paths` prints them anyway, for the
one job that cannot be done without seeing the frames: labelling by eye. The population is
personal photographs only, so a document never enters this table in the first place.

The database is opened through the project's own connection; nothing is written to it.

Usage (from the repo root, with a GPU venv — `uv sync --extra gpu --extra dev`):
    python scripts/measure_search.py --queries "торт" "снег" "море"
    python scripts/measure_search.py --queries "торт" --top 40 --paths
    python scripts/measure_search.py --queries-file queries.txt --write-labels marks.json
    python scripts/measure_search.py --queries-file queries.txt --labels marks.json
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sorta import search  # noqa: E402
from sorta.config import Config, load_config  # noqa: E402
from sorta.db import connect  # noqa: E402
from sorta.junk import search_index_model, search_index_settings  # noqa: E402
from sorta.naming import naming_settings  # noqa: E402

# The bands the collection is spread over. Uneven on purpose: a CLIP cosine between a short
# query and a photograph lives in a narrow strip (roughly 0.15-0.35 for this model), so
# bands of equal width would put the entire collection into one of them and say nothing.
# The fine steps are where the answer changes.
SCORE_BANDS = (0.0, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.32, 0.35, 1.0001)

# The depths the precision is reported at. A search is read from the top, and where it
# stops being useful is the question — one number over the whole list would hide it.
DEPTHS = (5, 10, 20, 50)

DEFAULT_TOP = 20
# Below this many marks a per-query precision is a number with no interval worth printing.
MIN_LABELS = 10


@dataclass(frozen=True)
class Result:
    """One query, ranked: the top the human reads and every score for the distribution."""
    query: str
    hits: list[tuple[int, float]]   # (file_id, score), best first — the top only
    scores: list[float]             # the score of every frame in the collection


def band_of(score: float, bands: tuple[float, ...] = SCORE_BANDS) -> tuple[float, float]:
    """The [low, high) band a score falls into; the lowest band swallows everything below."""
    for low, high in zip(bands, bands[1:]):
        if low <= score < high:
            return low, high
    return bands[0], bands[1]


def band_counts(scores: list[float],
                bands: tuple[float, ...] = SCORE_BANDS) -> list[tuple[float, float, int]]:
    """(low, high, how many frames) per band, in the order the bands are declared."""
    counts = {(low, high): 0 for low, high in zip(bands, bands[1:])}
    for score in scores:
        counts[band_of(score, bands)] += 1
    return [(low, high, counts[(low, high)]) for low, high in zip(bands, bands[1:])]


def precision_at(hits: list[tuple[int, float]], labels: dict[int, bool],
                 depths: tuple[int, ...] = DEPTHS) -> list[tuple[int, int, int, float]]:
    """(depth, labelled, correct, precision) — over the LABELLED frames of each prefix.

    Unlabelled frames are dropped rather than counted as wrong: a partially filled
    worksheet has to be usable, and treating an unanswered frame as a miss would invent the
    very marks the worksheet exists to collect. A depth with nothing labelled in it is
    reported with a precision of 0.0 and a count of 0, so the emptiness is visible instead
    of being rounded into an answer.
    """
    rows: list[tuple[int, int, int, float]] = []
    for depth in depths:
        marked = [labels[fid] for fid, _score in hits[:depth] if fid in labels]
        correct = sum(1 for value in marked if value)
        rows.append((depth, len(marked), correct,
                     correct / len(marked) if marked else 0.0))
    return rows


def precision_by_band(hits: list[tuple[int, float]], labels: dict[int, bool],
                      bands: tuple[float, ...] = SCORE_BANDS,
                      ) -> list[tuple[float, float, int, int, float]]:
    """The same numbers per score band: (low, high, labelled, correct, precision).

    This is the table that answers "where does the ranking stop being about the query",
    which no single accuracy figure can. Bands with nothing labelled are left out — an
    empty row here would be a claim about a band nobody looked at.
    """
    marked: dict[tuple[float, float], list[bool]] = {}
    for file_id, score in hits:
        if file_id in labels:
            marked.setdefault(band_of(score, bands), []).append(labels[file_id])
    rows: list[tuple[float, float, int, int, float]] = []
    for low, high in zip(bands, bands[1:]):
        values = marked.get((low, high))
        if not values:
            continue
        correct = sum(1 for value in values if value)
        rows.append((low, high, len(values), correct, correct / len(values)))
    return rows


def format_top(result: Result, paths: dict[int, str] | None) -> str:
    """The list itself. Ids only unless the caller asked for paths (see the privacy note)."""
    out = [f"ЗАПРОС «{result.query}» — top {len(result.hits)} из {len(result.scores)}",
           f"  {'#':>4} {'оценка':>7}  {'путь' if paths else 'file_id'}"]
    for rank, (file_id, score) in enumerate(result.hits, 1):
        tail = paths.get(file_id, "?") if paths else str(file_id)
        out.append(f"  {rank:>4} {score:>7.3f}  {tail}")
    return "\n".join(out)


def format_bands(result: Result) -> str:
    """Where the collection sits for this query — the shape the top-N came out of."""
    total = len(result.scores) or 1
    out = [f"ПОЛОСЫ БЛИЗОСТИ «{result.query}»",
           f"  {'полоса':>12} {'кадров':>8} {'доля':>7} {'из них в top':>13}"]
    top_by_band = {band: 0 for band in zip(SCORE_BANDS, SCORE_BANDS[1:])}
    for _file_id, score in result.hits:
        top_by_band[band_of(score)] += 1
    for low, high, count in reversed(band_counts(result.scores)):
        out.append(f"  {low:>5.2f}-{high:<6.2f} {count:>8} {count / total:>7.1%} "
                   f"{top_by_band[(low, high)]:>13}")
    return "\n".join(out)


def format_precision(result: Result, labels: dict[int, bool]) -> str:
    """The numbers — and the warning that they are few, when they are.

    Printed only where marks exist: an accuracy computed over three frames is exactly the
    kind of number this script was written to prevent.
    """
    marked = [fid for fid, _score in result.hits if fid in labels]
    out = [f"ТОЧНОСТЬ «{result.query}» (размечено {len(marked)} из {len(result.hits)})"]
    if len(marked) < MIN_LABELS:
        out.append(f"  ВНИМАНИЕ: меньше {MIN_LABELS} отметок — по такой выборке точность "
                   f"не называют")
    out.append(f"  {'глубина':>8} {'размечено':>10} {'верных':>8} {'точность':>9}")
    for depth, labelled, correct, precision in precision_at(result.hits, labels):
        out.append(f"  {depth:>8} {labelled:>10} {correct:>8} {precision:>9.1%}")
    rows = precision_by_band(result.hits, labels)
    if rows:
        out.append(f"  {'полоса':>12} {'размечено':>10} {'верных':>8} {'точность':>9}")
        for low, high, labelled, correct, precision in reversed(rows):
            out.append(f"  {low:>5.2f}-{high:<6.2f} {labelled:>10} {correct:>8} "
                       f"{precision:>9.1%}")
    return "\n".join(out)


def write_label_template(path: Path, results: list[Result]) -> int:
    """A worksheet to fill in: `{query: {file_id: null}}` — file ids and nothing else.

    Whoever fills it in opens those frames (`--paths`, or the web app by id) and replaces
    each null with true — the frame answers the query — or false. Returns how many frames
    were written.
    """
    sheet = {r.query: {str(fid): None for fid, _score in r.hits} for r in results}
    path.write_text(json.dumps(sheet, ensure_ascii=False, indent=1), encoding="utf-8")
    return sum(len(r.hits) for r in results)


def load_labels(path: Path) -> dict[str, dict[int, bool]]:
    """The filled-in worksheet -> {query: {file_id: does this frame answer it}}.

    Frames still holding `null` are simply not labelled yet and are dropped.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    return {query: {int(fid): bool(value) for fid, value in marks.items()
                    if value is not None}
            for query, marks in data.items()}


def read_queries(args: argparse.Namespace) -> list[str]:
    """The queries, from the command line or from a file (one per line, `#` a comment)."""
    queries = list(args.queries or [])
    if args.queries_file:
        for line in Path(args.queries_file).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                queries.append(line)
    return list(dict.fromkeys(queries))  # a repeated query would be measured twice


def rank_all(cfg: Config, conn, queries: list[str],
             top: int) -> list[Result]:  # pragma: no cover — ML
    """One CLIP text pass per query, then the same ranking `sorta search` runs.

    The whole collection is ranked, not just the top: the band table is about the
    distribution the top was drawn from, and a top-N alone cannot show it.
    """
    model = search_index_model(cfg)
    encoder = search.text_encoder(search_index_settings(naming_settings(cfg), model))
    total = int(conn.execute("SELECT COUNT(*) FROM search_embeddings").fetchone()[0])
    results: list[Result] = []
    for query in queries:
        ranked = search.search(conn, search.encode_query(query, encoder), model,
                               max(total, 1))
        results.append(Result(query=query, hits=ranked[:top],
                              scores=[score for _fid, score in ranked]))
        print(f"  ранжирование: «{query}» — {len(ranked)} кадров", flush=True)
    return results


def main() -> None:  # pragma: no cover — I/O and the model
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--queries", nargs="+", help="the queries to measure")
    ap.add_argument("--queries-file", help="a file with one query per line (# — comment)")
    ap.add_argument("--top", type=int, default=DEFAULT_TOP,
                    help=f"how many frames per query to print (default {DEFAULT_TOP})")
    ap.add_argument("--paths", action="store_true",
                    help="print file paths instead of ids — needed to label by eye")
    ap.add_argument("--write-labels", help="write a worksheet of the top-N (file ids "
                                           "only) to fill in, and exit")
    ap.add_argument("--labels", help="JSON {query: {file_id: true|false}} — enables the "
                                     "precision block")
    args = ap.parse_args()

    queries = read_queries(args)
    if not queries:
        raise SystemExit("нечего мерить: задайте --queries или --queries-file")

    cfg = load_config(args.config)
    conn = connect(cfg.database)
    try:
        try:
            results = rank_all(cfg, conn, queries, args.top)
        except search.EmbeddingsMissing as exc:
            raise SystemExit(
                f"нечего ранжировать ({exc.reason}): эмбеддингов этой модели в поисковом "
                f"индексе нет — запустите `sorta junk` (features.search_index: true)"
            ) from None

        if args.write_labels:
            written = write_label_template(Path(args.write_labels), results)
            print(f"размечать: {written} кадров записано в {args.write_labels} "
                  f"(только file_id; замените null на true/false)")
            return

        labels = load_labels(Path(args.labels)) if args.labels else {}
        for result in results:
            paths = (search.file_paths(conn, [fid for fid, _s in result.hits])
                     if args.paths else None)
            print()
            print(format_top(result, paths))
            print(format_bands(result))
            if result.query in labels:
                print(format_precision(result, labels[result.query]))
        if args.labels and not labels:
            print("\nразметки нет: в файле только null — точность не считается")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
