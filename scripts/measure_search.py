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
  depths and per score band — the numbers themselves;
* with `--fusion`, the same numbers for FOUR variants side by side: each index alone and
  each way of merging them (F153).

F153 and the half nobody measured
---------------------------------
Every accuracy quoted for this search so far is a precision at the top, and both models
have nearly the same one (88/96/98% at ranks 1/3/5) while returning DIFFERENT frames. So
precision cannot decide whether merging the two indexes is worth anything — RECALL can,
and it has never been computed for either of them. `--fusion` is that measurement: it
ranks both indexes, merges them with `search.fuse` (the feature's own function, not a copy
of it), and prints precision AND recall at each depth for L14, XLM, `rank` and `union`.

The recall it prints is a POOLED recall and says so on every table: the denominator is the
frames a person marked `true` for that query, which is all a hand-labelled sample can
offer. That makes it a comparison between variants and not an absolute — and it is biased
towards whatever produced the frames that got labelled, which is why the tables also print
how much of each variant's top is UNLABELLED. A variant whose top is half unmarked has not
been measured yet, it has been sampled; the honest move then is to label those frames
(`--write-labels` covers the merged tops too) rather than to read a number off it.

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
    python scripts/measure_search.py --queries-file queries.txt --labels marks.json --fusion
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sorta import search  # noqa: E402
from sorta.config import (  # noqa: E402
    SEARCH_FUSION_RANK,
    SEARCH_FUSION_UNION,
    Config,
    load_config,
)
from sorta.db import connect  # noqa: E402
from sorta.junk import (  # noqa: E402
    embedding_model,
    search_index_model,
    search_index_settings,
)
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

# F153: the four variants `--fusion` compares. The two indexes are named by the model
# everybody calls them by rather than by their table, because that is how the earlier
# measurements are written down; the two merges are named by the config value that turns
# them on, so a number printed here maps onto `features.search_fusion` with no translation.
VARIANT_SEARCH = "XLM"    # the search index, `search_embeddings` (F141)
VARIANT_CLASS = "L14"     # the classification index, `clip_embeddings` (F128)
FUSION_VARIANTS = (SEARCH_FUSION_RANK, SEARCH_FUSION_UNION)
# How deep the merged lists are built. Deeper than the deepest measured depth on purpose:
# a rank fusion at depth 50 must be able to pull a frame up from below 50 in one of the
# lists, which is exactly the case the whole feature is a bet on.
FUSION_DEPTH = 200


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


def recall_at(hits: list[tuple[int, float]], labels: dict[int, bool],
              depths: tuple[int, ...] = DEPTHS) -> list[tuple[int, int, int, float]]:
    """(depth, found, pool, recall) — POOLED recall, and the pool is printed with it.

    The denominator is every frame marked `true` for this query, wherever it was found:
    that is the only set of relevant frames a hand-labelled sample can offer, since nobody
    has looked at all 19 753 photographs. A frame in the prefix that carries no mark counts
    as not relevant HERE, which is the opposite of what `precision_at` does with it — and
    deliberately so: precision asks "of what a person checked, how much was right", recall
    asks "of what a person found, how much did this variant surface". Mixing the two rules
    would let a variant improve its recall by returning frames nobody has judged.

    A query with no positive marks at all reports 0.0 over a pool of 0, so the emptiness
    is visible rather than rounded into an answer.
    """
    pool = sum(1 for value in labels.values() if value)
    rows: list[tuple[int, int, int, float]] = []
    for depth in depths:
        found = sum(1 for file_id, _score in hits[:depth] if labels.get(file_id))
        rows.append((depth, found, pool, found / pool if pool else 0.0))
    return rows


@dataclass(frozen=True)
class Score:
    """One (variant, depth) cell of the comparison table — F153's whole output.

    `unlabelled` is not decoration: it is how many frames of the prefix nobody has judged,
    and a precision computed next to a large one of those is a number about a sample, not
    about the variant.
    """
    variant: str
    depth: int
    labelled: int
    correct: int
    precision: float
    found: int
    pool: int
    recall: float
    unlabelled: int


def compare(variants: dict[str, list[tuple[int, float]]], labels: dict[int, bool],
            depths: tuple[int, ...] = DEPTHS) -> list[Score]:
    """Every variant at every depth, from the SAME marks — the table the brief asks for.

    One label set for all four variants is the point: a frame is judged once whatever
    produced it (the rule the 217 judgements of 2026-08-02 were collected under), so the
    numbers of two variants are about the same photographs and can be put in one column.
    """
    scores: list[Score] = []
    for variant, hits in variants.items():
        precisions = {row[0]: row[1:] for row in precision_at(hits, labels, depths)}
        recalls = {row[0]: row[1:] for row in recall_at(hits, labels, depths)}
        for depth in depths:
            labelled, correct, precision = precisions[depth]
            found, pool, recall = recalls[depth]
            scores.append(Score(
                variant=variant, depth=depth, labelled=labelled, correct=correct,
                precision=precision, found=found, pool=pool, recall=recall,
                unlabelled=len(hits[:depth]) - labelled))
    return scores


def format_comparison(query: str, scores: list[Score]) -> str:
    """The four variants side by side, and the warnings that keep them readable.

    Sorted by depth first and then by the order the variants were computed in, so the
    question the table answers — "at this depth, which variant found more" — is read across
    consecutive lines rather than by hunting up and down the page.
    """
    pool = max((s.pool for s in scores), default=0)
    out = [f"СЛИЯНИЕ «{query}» (положительных отметок в пуле: {pool})"]
    if pool < MIN_LABELS:
        out.append(f"  ВНИМАНИЕ: меньше {MIN_LABELS} положительных отметок — по такой "
                   f"выборке ни точность, ни полноту не называют")
    out.append("  полнота здесь ОТНОСИТЕЛЬНАЯ: знаменатель — размеченные кадры, а не все "
               "подходящие в коллекции")
    out.append(f"  {'глубина':>8} {'вариант':>8} {'размечено':>10} {'точность':>9} "
               f"{'найдено':>8} {'полнота':>8} {'без отметки':>12}")
    for score in sorted(scores, key=lambda s: s.depth):
        out.append(f"  {score.depth:>8} {score.variant:>8} {score.labelled:>10} "
                   f"{score.precision:>9.1%} {score.found:>8} {score.recall:>8.1%} "
                   f"{score.unlabelled:>12}")
    if any(s.unlabelled > s.labelled for s in scores):
        out.append("  ВНИМАНИЕ: у части вариантов неразмеченных кадров больше, чем "
                   "размеченных — разметки для вывода не хватает, доразметьте выдачу")
    return "\n".join(out)


def with_fusions(rankings: dict[str, list[tuple[int, float]]],
                 depth: int = FUSION_DEPTH) -> dict[str, list[tuple[int, float]]]:
    """The index rankings plus one per fusion mode — the four variants, in order.

    `search.fuse` does the merging, not a copy of it living here: a private
    reimplementation would measure this script instead of the feature, which is the rule
    the whole module already follows for `encode_query` and `search`. It is handed file ids
    and no scores, so nothing in this measurement can add up numbers of two spaces either.
    """
    out = dict(rankings)
    lists = [[file_id for file_id, _score in hits] for hits in rankings.values() if hits]
    for mode in FUSION_VARIANTS:
        out[mode] = search.fuse(lists, mode, depth) if lists else []
    return out


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


def format_recall(result: Result, labels: dict[int, bool]) -> str:
    """The other half of the numbers — how much of what a person marked this list reached.

    Printed next to the precision table and never instead of it: a precision of 98% at
    top-5 says nothing about how many of the cakes in the archive were left behind, and
    that is the question F153 exists to answer. The denominator is stated on the line
    above the table, because a relative recall read as an absolute one is worse than no
    recall at all.
    """
    pool = sum(1 for value in labels.values() if value)
    out = [f"ПОЛНОТА «{result.query}» (знаменатель — {pool} размеченных подходящих "
           f"кадров, не вся коллекция)"]
    if pool < MIN_LABELS:
        out.append(f"  ВНИМАНИЕ: меньше {MIN_LABELS} положительных отметок — по такой "
                   f"выборке полноту не называют")
    out.append(f"  {'глубина':>8} {'найдено':>8} {'из':>6} {'полнота':>9}")
    for depth, found, total, recall in recall_at(result.hits, labels):
        out.append(f"  {depth:>8} {found:>8} {total:>6} {recall:>9.1%}")
    return "\n".join(out)


def top_ids(results: list[Result]) -> dict[str, list[int]]:
    """{query: the file ids of its top}, the ordinary worksheet of one variant."""
    return {r.query: [file_id for file_id, _score in r.hits] for r in results}


def merged_ids(variants: dict[str, list[tuple[int, float]]], top: int) -> list[int]:
    """F153: the top-N of EVERY variant, each frame once, in the order first met.

    What a fusion worksheet has to cover: a frame the merge pulled up and neither index
    showed on its own is unlabelled by construction, and `recall_at` counts an unlabelled
    frame as not relevant. Labelling only one variant's top therefore measures the merge
    against a sample chosen by its competitor.
    """
    out: dict[int, None] = {}
    for hits in variants.values():
        for file_id, _score in hits[:top]:
            out.setdefault(file_id, None)
    return list(out)


def write_label_template(path: Path, ids_by_query: dict[str, Sequence[int]]) -> int:
    """A worksheet to fill in: `{query: {file_id: null}}` — file ids and nothing else.

    Whoever fills it in opens those frames (`--paths`, or the web app by id) and replaces
    each null with true — the frame answers the query — or false. Returns how many frames
    were written.
    """
    sheet = {query: {str(file_id): None for file_id in ids}
             for query, ids in ids_by_query.items()}
    path.write_text(json.dumps(sheet, ensure_ascii=False, indent=1), encoding="utf-8")
    return sum(len(marks) for marks in sheet.values())


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


def rank_variants(cfg: Config, conn, queries: list[str], depth: int = FUSION_DEPTH,
                  ) -> dict[str, dict[str, list[tuple[int, float]]]]:  # pragma: no cover
    """F153: {query: {variant: ranking}} for BOTH indexes — the input of `with_fusions`.

    One tower at a time and not two side by side: the second model is not needed until the
    first has answered every query, and two CLIP text towers in memory at once is a peak
    this machine has no reason to pay. Each index is asked with the query encoded by ITS
    OWN tower — that is the entire reason the two live apart — and ranked `depth` deep
    rather than over the whole collection, because a merge and the depths measured here
    both fit inside that.

    An index that cannot rank at all (no vectors, or all of another model) drops out with a
    printed reason and the run continues on what is left: that is the state the feature
    itself handles the same way, and a measurement that died there would tell nobody why.
    """
    s = naming_settings(cfg)
    index_model = search_index_model(cfg)
    plan = (
        (VARIANT_SEARCH, index_model, search_index_settings(s, index_model), search.search),
        (VARIANT_CLASS, embedding_model(s), s, search.search_classification),
    )
    out: dict[str, dict[str, list[tuple[int, float]]]] = {query: {} for query in queries}
    for variant, model, settings, rank_with in plan:
        encoder = search.text_encoder(settings)
        for query in queries:
            try:
                ranked = rank_with(conn, search.encode_query(query, encoder), model, depth)
            except search.EmbeddingsMissing as exc:
                print(f"  {variant} ({model}): ранжировать нечем ({exc.reason}) — "
                      f"вариант не участвует", flush=True)
                break
            out[query][variant] = ranked
            print(f"  ранжирование {variant}: «{query}» — {len(ranked)} кадров", flush=True)
    return out


def report_fusion(cfg: Config, conn, queries: list[str], args: argparse.Namespace,
                  ) -> None:  # pragma: no cover — I/O and the model
    """`--fusion`: the four variants of F153, measured on one set of marks.

    The band tables are deliberately absent here: a band is a table of COSINES, and two of
    the four variants are ranked by a weight computed from positions that belongs to no
    vector space. Printing one for them would invent a comparison the feature spends its
    whole design refusing to make.
    """
    ranked = rank_variants(cfg, conn, queries)
    variants = {query: with_fusions(per_index) for query, per_index in ranked.items()}
    if args.write_labels:
        written = write_label_template(
            Path(args.write_labels),
            {query: merged_ids(per_variant, args.top)
             for query, per_variant in variants.items()})
        print(f"размечать: {written} кадров записано в {args.write_labels} — это ОБЪЕДИНЁННАЯ "
              f"верхушка всех вариантов (только file_id; замените null на true/false)")
        return
    labels = load_labels(Path(args.labels)) if args.labels else {}
    if not labels:
        raise SystemExit("для сравнения вариантов нужна разметка: --labels marks.json "
                         "(сначала --fusion --write-labels)")
    for query, per_variant in variants.items():
        if query not in labels:
            continue
        print()
        print(format_comparison(query, compare(per_variant, labels[query])))


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
    ap.add_argument("--fusion", action="store_true",
                    help="F153: rank both indexes and print precision and recall at each "
                         "depth for L14, XLM and each fusion mode (needs --labels)")
    args = ap.parse_args()

    queries = read_queries(args)
    if not queries:
        raise SystemExit("нечего мерить: задайте --queries или --queries-file")

    cfg = load_config(args.config)
    conn = connect(cfg.database)
    try:
        if args.fusion:
            report_fusion(cfg, conn, queries, args)
            return
        try:
            results = rank_all(cfg, conn, queries, args.top)
        except search.EmbeddingsMissing as exc:
            raise SystemExit(
                f"нечего ранжировать ({exc.reason}): эмбеддингов этой модели в поисковом "
                f"индексе нет — запустите `sorta junk` (features.search_index: true)"
            ) from None

        if args.write_labels:
            written = write_label_template(Path(args.write_labels), top_ids(results))
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
                print(format_recall(result, labels[result.query]))
        if args.labels and not labels:
            print("\nразметки нет: в файле только null — точность не считается")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
