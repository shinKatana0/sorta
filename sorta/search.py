"""F129: search by words over the CLIP vectors the junk stage keeps — the engine, no interface.

Every slice of the collection is written code today: a tab for animals, a tab for people,
a tab for products. A search turns a slice into a QUERY — someone types a word and gets
one — so "food", "snow", "the sea" stop being a programmer's work. What that stands on is
already on disk: `search_embeddings` holds an L2-normalized CLIP vector per canonical
photograph together with the name of the model that produced it.

F141 is why that table is not `clip_embeddings`. A search has to answer the language the
user types, and the classification model does not: `ViT-L-14` scores 22% precision at
top-5 on Russian queries against 98% for `xlm-roberta-base-ViT-B-32`, with four of eight
labelled concepts (cake, food, mountains, children) returning nothing whatsoever. Swapping
the pipeline's model would fix that for free and would also invalidate the landmark (F75),
animal (F122) and cascade (F130) thresholds calibrated on its numbers — so search got a
SECOND vector, from a model of its own (`features.search_model`), written by a second pass
behind `features.search_index`. With that toggle off this engine has nothing to rank and
says so; the classification vectors are deliberately not a fallback, because a ranking
produced by the wrong model looks exactly like a good one.

Three properties are the whole feature, and each of them is a decision rather than an
implementation detail:

* **The model filter is not optional.** A vector computed by another model is not
  comparable with this query, and mixing the two produces a plausible ranking that nothing
  in the output marks as wrong. Rows of another model never enter the ranking; if that
  leaves nothing to rank, the caller is told so (`EmbeddingsMissing`) instead of being
  handed a short list. The filter itself lives where F128 put it — inside
  `junk.read_search_embeddings`, the one function that reads the table.
* **An empty table is a reason, not an empty result.** "Nothing was found" and "nothing was
  ever computed" read identically in a list of zero lines, and only one of them is fixed by
  running `sorta junk`.
* **This ranks, it does not classify.** There is no "this really is a cake" threshold and
  there will not be one, for the same reason sharpness has none: the score orders frames
  against each other and says nothing in absolute terms. `features.search_page` is
  therefore a SAMPLE SIZE, not a cutoff — and since F173 it is not even the end of the
  sample: `rank` returns a WINDOW of the ranking together with the length of the whole of
  it, so a caller can walk down the list instead of being cut off at the first page. Depth
  is the one lever of completeness the measurements found (the query «дети» goes from 61%
  to 89% when the list is doubled), so it is the last thing this engine may take away.

Known limits, measured elsewhere and repeated here so a caller does not have to guess:
compound queries ("a cake with candles on a table by the window") are weak — CLIP takes a
sentence as one whole and single subjects are what it does well. The population is
personal photographs only (F120): a screenshot's vector is noise in a search over a family
archive, and it is not in the table to begin with — which also means a document (a
passport, a medical form) can never surface here, because the stage stores no row for one.

The accuracy of this search HAS been measured on a real collection, with
`scripts/measure_search.py` and 217 hand-labelled judgements over 8 concepts, each
concept-frame pair judged once whatever produced it. With the search index on
(`xlm-roberta-base-ViT-B-32`) it is 98% precision at top-5 in Russian and 95% in English;
the classification model it replaced for this purpose gave 22% in Russian and 98% in
English. F121/F122 is why that measurement was not optional: the animal class looked like
it worked until 320 hand-labelled frames showed that only half of the question was right.
The one concept still weak for both models is "a city at night" (80%) — a compound query,
which is the class the limits above already name.

F153 puts the OTHER index next to this one, behind `features.search_fusion`. The two
models score 88/96/98% at ranks 1/3/5 apiece and return DIFFERENT frames for the same
word — the user's own words while looking at both lists: "it disagrees with xlm english on
which photos, but both are good, even though they differ" — and two models that are wrong
in different places are the one case where merging beats either half. Two ways of merging
are offered and both work on POSITIONS: `rank` (reciprocal rank fusion — agreement between
the models wins) and `union` (the two lists as sets, each frame keeping its best place).
Neither one adds the scores up, and `fuse` cannot: it is handed file ids and no numbers,
because a cosine of ViT-L-14 and a cosine of xlm-roberta-base-ViT-B-32 belong to different
spaces and look comparable anyway. Which mode is worth defaulting to is a question for
`scripts/measure_search.py --fusion`, which prints precision AND RECALL — the half nobody
has measured, and the half a merge is expected to move. A merged ranking is paged like any
other (F173): the window is cut after the merge and the total is the length of the merged
list, so «показано N из M» stays a fact about the list the reader is actually looking at.

The real CLIP is loaded exactly once, in `text_encoder`; everything else takes an encoder
as an argument, which is what lets the tests run the whole engine without a model.
"""
from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .config import (
    SEARCH_FUSION_OFF,
    SEARCH_FUSION_RANK,
    SEARCH_FUSION_UNION,
    Config,
    FeaturesConfig,
)
from .junk import (
    embedding_model,
    read_clip_embeddings,
    read_search_embeddings,
    search_index_model,
    search_index_settings,
)
from .landmarks import batched
from .naming import NamingSettings, naming_settings

_log = logging.getLogger(__name__)

# Query strings -> their text features, one row per string, in the same order. Replaced in
# tests; the real one is `text_encoder` below — the project's own open_clip, never a second
# model, because a query has to land in the space the stored vectors live in.
TextEncoder = Callable[[Sequence[str]], np.ndarray]

# The population of a search (F120), and the reason it is spelled out here rather than left
# to the table: `search_embeddings` is written by the junk stage and only ever holds personal
# photographs, but a file can become a duplicate or go unreadable AFTER its vector was
# stored, and neither of those belongs in a result list.
_CANDIDATES_SQL = """SELECT id FROM files
    WHERE dup_of IS NULL AND error IS NULL AND media_type = 'photo'"""

# Why there is nothing to rank. Two states, and they need two different sentences: one is
# fixed by running the junk stage, the other by running it AGAIN after a model change.
REASON_EMPTY = "empty"              # the table holds no vectors at all
REASON_OTHER_MODEL = "other_model"  # it holds vectors, all of them of another model

# The two tables a ranking can come out of, named so `_nothing_to_rank` can count the one
# that was actually read. They are never mixed in a single ranking — that is the whole
# point of the model column — and a fusion merges two RANKINGS, not two tables.
_SEARCH_TABLE = "search_embeddings"  # F141: the multilingual index, search's own
_CLASS_TABLE = "clip_embeddings"     # F128: the classification vectors, ViT-L-14

# F153: the smoothing constant of reciprocal rank fusion, at the value the method was
# published with. It is not calibrated on anything here and nothing asks it to be: any
# positive K leaves a single list in its own order and keeps the property the mode exists
# for — a frame both models put first outranks a frame only one of them did. What K does
# choose is how far down a second list may still rescue a frame, and 60 is the usual
# answer to that.
RRF_K = 60


class EmbeddingsMissing(RuntimeError):
    """No usable vector for this query — raised instead of returning an empty list.

    Carries the reason as a code rather than as a message: the sentence a user reads is
    interface text and belongs in the i18n catalog with its three languages, while the
    engine only knows WHICH of the two states it is in. `stored` counts the rows of the
    query's own model, `total` every row in the table — the pair is what makes "you have
    19 757 vectors, all of another model" sayable.
    """

    def __init__(self, reason: str, model: str, total: int, stored: int) -> None:
        super().__init__(f"{reason}: model={model!r}, stored={stored}, total={total}")
        self.reason = reason
        self.model = model
        self.total = total
        self.stored = stored


def text_encoder(s: NamingSettings) -> TextEncoder:  # pragma: no cover — ML, smoke test
    """The real open_clip text tower — the same model and weights the images went through.

    Loaded through `create_model_and_transforms` like `landmarks.clip_classifier`, and
    deliberately so: the pair (architecture, checkpoint) has to be resolved by the same call
    the image side used, otherwise "the same model" is a claim rather than a fact. The
    returned vectors are L2-normalized here as well, so a dot product is a cosine.

    F141: WHICH settings those are is the caller's to get right, and for a search they are
    the search side's — `junk.search_index_settings(naming_settings(cfg), model)`, which is
    what `search_text` passes. Handed `naming_settings(cfg)` unchanged this builds the
    classification tower, whose queries land in a space no row of the search index lives
    in; the width check in `search` catches that and leaves nothing to rank, which is a
    visible failure rather than a quiet one.
    """
    import open_clip
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, _ = open_clip.create_model_and_transforms(
        s.clip_model, pretrained=s.clip_pretrained, device=device)
    tokenizer = open_clip.get_tokenizer(s.clip_model)
    model.eval()

    def encode(texts: Sequence[str]) -> np.ndarray:
        with torch.no_grad():
            feats = model.encode_text(tokenizer(list(texts)).to(device))
            feats /= feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().astype(np.float32)

    return encode


# F153: the classification text tower, built at most once per process. The callers that
# bring their own encoder bring the SEARCH one (the web app holds one per server, the album
# one per run), so a fusion has to find the second half somewhere — and building it per
# query would mean loading ViT-L-14 on every word somebody types, which is precisely what
# those callers cache to avoid. Keyed by the model name because that pair (architecture,
# weights) is what a tower IS.
_class_encoders: dict[str, TextEncoder] = {}


def _classification_encoder(s: NamingSettings) -> TextEncoder:  # pragma: no cover — ML
    """The text tower of the classification model, from the cache above."""
    key = embedding_model(s)
    if key not in _class_encoders:
        _class_encoders[key] = text_encoder(s)
    return _class_encoders[key]


def encode_query(text: str, encoder: TextEncoder) -> np.ndarray:
    """A query -> a unit vector in the space the stored embeddings live in.

    Normalized here even though the encoder already normalizes, for the reason
    `junk.pack_embedding` normalizes a vector the image tower had normalized: with unit
    vectors on both sides a search is one matmul and no per-row arithmetic, and that
    guarantee is worth more than the trust it replaces. A zero vector (an encoder that
    answered with nothing) is left as it is rather than divided by zero — it ranks
    everything equally, which is the honest outcome of a query that carries no direction.
    """
    if not text.strip():
        raise ValueError("encode_query: the query is empty")
    vec = np.asarray(encoder([text.strip()]), dtype=np.float32).ravel()
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else vec


@dataclass(frozen=True)
class Page:
    """One window of a ranking, and how long the ranking it came out of is.

    F173: `total` is the point. A list of exactly `limit` frames says nothing about whether
    the ranking ended there or was cut there, and those are different facts — for a query
    the second is almost always the true one. Everything a caller needs to draw «показано
    N из M» and to decide whether a "show more" button belongs on the screen is here, so
    that neither of the two can be recomputed slightly differently somewhere else.
    """

    hits: list[tuple[int, float]]
    total: int
    offset: int
    limit: int

    @property
    def has_more(self) -> bool:
        """Whether anything is left below this window — the button's whole condition."""
        return self.offset + len(self.hits) < self.total


def rank(conn: sqlite3.Connection, query: np.ndarray, model: str, *,
         limit: int, offset: int = 0) -> Page:
    """The ranking: (file_id, score), best first, windowed to [offset, offset+limit).

    The score is a dot product — both sides are unit vectors, so it IS the cosine, and no
    normalization happens per row. Ties are broken by file_id (the ids are sorted before the
    stable argsort), which is what makes a repeated search return the same list rather than
    whatever order the dict happened to have: a ranking that reshuffles between runs cannot
    be measured, and measuring it is a condition of this feature.

    That same determinism is what makes PAGING honest here (F173). The whole list is scored
    and ordered on every call, and a window is taken out of it afterwards, so the second
    page is the continuation of the first and not a second opinion about the collection: no
    frame is shown twice and none is skipped between the pages. Only the window is turned
    into Python objects — the argsort ran over everything either way, and materializing
    300 000 tuples to hand back 200 of them would be the one part of this that scales badly.

    Vectors of another model are absent by construction — `read_search_embeddings` filters
    on `model` — and a row whose width does not match the query is dropped as well: a
    truncated blob is a broken row, not a reason for the whole search to fail. That width
    check is also the last guard against a query encoded by the classification tower (768
    numbers) reaching the search index (512): it cannot rank, so it ranks nothing.

    F153: this ranks the SEARCH index and only ever that one. `rank_classification` is the
    other table's twin, and the two are separate functions rather than one with an argument
    for the same reason `junk` has two readers — see there.
    """
    q = np.asarray(query, dtype=np.float32).ravel()
    candidates = [int(r["id"]) for r in conn.execute(_CANDIDATES_SQL)]
    return _rank(conn, read_search_embeddings(conn, model, candidates), q, model,
                 limit=limit, offset=offset, table=_SEARCH_TABLE)


def rank_classification(conn: sqlite3.Connection, query: np.ndarray, model: str, *,
                        limit: int, offset: int = 0) -> Page:
    """F153: the same ranking over the CLASSIFICATION index (`clip_embeddings`).

    A function of its own rather than a table argument on `rank`, for the reason
    `junk.read_search_embeddings` is a function of its own: the model filter is the safety
    property of both, and a parameter choosing which table to apply it to is one call site
    away from being handed the wrong one. Here the two stay apart all the way down — the
    reader, the model name and the table counted in the refusal are picked together.

    The query must be encoded by the CLASSIFICATION tower; a search-model query is 512
    numbers against these 768 and the width check below drops every row rather than
    ranking it, which is a visible failure and not a plausible list. This is only ever
    reached through `features.search_fusion`: on its own the classification index is NOT a
    fallback for search (F141), because a ranking quietly produced by the wrong model looks
    exactly like a good one.
    """
    q = np.asarray(query, dtype=np.float32).ravel()
    candidates = [int(r["id"]) for r in conn.execute(_CANDIDATES_SQL)]
    return _rank(conn, read_clip_embeddings(conn, model, candidates), q, model,
                 limit=limit, offset=offset, table=_CLASS_TABLE)


def _rank(conn: sqlite3.Connection, vectors: dict[int, np.ndarray], q: np.ndarray,
          model: str, *, limit: int, offset: int, table: str) -> Page:
    """The arithmetic both indexes share: unit vectors in, one window of a ranking out."""
    ids = sorted(fid for fid, vec in vectors.items() if vec.size == q.size)
    if not ids:
        raise _nothing_to_rank(conn, model, table)
    matrix = np.stack([vectors[fid] for fid in ids])
    scores = matrix @ q
    start = max(0, offset)
    order = np.argsort(-scores, kind="stable")[start:start + max(0, limit)]
    return Page(hits=[(ids[i], float(scores[i])) for i in order],
                total=len(ids), offset=start, limit=max(0, limit))


def search(conn: sqlite3.Connection, query: np.ndarray, model: str,
           limit: int, offset: int = 0) -> list[tuple[int, float]]:
    """`rank` for a caller that wants the frames and not the length of the list.

    What `sorta search` and the measurement script both do with a ranking: print it. Kept
    as its own name so that the CLI is not made to unpack a page it has no use for.
    """
    return rank(conn, query, model, limit=limit, offset=offset).hits


def search_classification(conn: sqlite3.Connection, query: np.ndarray, model: str,
                          limit: int, offset: int = 0) -> list[tuple[int, float]]:
    """`rank_classification` for a caller that wants the frames alone (F153).

    The measurement script is that caller: it compares whole lists of file ids variant by
    variant, and the length of the classification index is not a fact it has any use for.
    """
    return rank_classification(conn, query, model, limit=limit, offset=offset).hits


# --- F153: two indexes, one answer -----------------------------------------------------

# One index, ranked: `rank` over the search index and `rank_classification` over the
# classification one. The two are picked together with the model name and never apart from
# it, which is why the fusion below carries the pair around rather than a table flag.
_Ranker = Callable[..., Page]

# How deep each index is ranked before the merge: all of it. A merge of two TRUNCATED lists
# cannot be windowed correctly — a frame just below the cut in both lists outranks a frame
# inside the cut in one of them, which is the whole property `rank` mode exists for — and
# it cannot state a total either, which is what F173's paging needs to be true. So the
# opt-in mode pays for two full lists of Python tuples per query. That is query time, and
# no pass over any image: a run does not get slower by a millisecond.
_WHOLE_RANKING = 1 << 30


@dataclass(frozen=True)
class Fusion:
    """One answer of the search, plus which indexes are behind it (F153).

    `page` is what every caller already handles — F173's window of the ranking with the
    length of the whole of it. The other two fields exist because a merge can be a merge of
    one: an index that has nothing to rank must not turn into a quietly shorter answer, so
    the models that ranked are named in `used` and the ones that did not are in `missing`
    with the engine's own reason code (`REASON_EMPTY` / `REASON_OTHER_MODEL`) next to them.

    What a `score` on that page MEANS depends on `mode`, and that is not a wart: with `off`
    it is the cosine of the search model, with a fusion it is a weight computed from
    POSITIONS in the two lists and belongs to no vector space at all. Both are ordering
    numbers with no absolute meaning (see the module docstring), which is why nothing in
    the project reads a threshold off one.
    """
    mode: str
    page: Page
    used: tuple[str, ...]
    missing: dict[str, str]

    @property
    def hits(self) -> list[tuple[int, float]]:
        """The window itself — for a caller that pages through nothing."""
        return self.page.hits


def fusion_mode(cfg: Config) -> str:
    """`features.search_fusion` — `off` | `rank` | `union` (default: `off`)."""
    features = getattr(cfg, "features", None) or FeaturesConfig()
    return str(getattr(features, "search_fusion", FeaturesConfig.search_fusion))


def fuse(rankings: Sequence[Sequence[int]], mode: str,
         limit: int) -> list[tuple[int, float]]:
    """Several rankings -> one, by POSITION alone: (file_id, weight), best first.

    File ids and nothing else is what this takes, and that is the main invariant of the
    feature rather than a convenience: the scores of two models are numbers of two
    different spaces, they print alike, and a function that never receives them cannot add
    them, average them or compare them. The order of `rankings` carries the only
    information used — the place a frame holds in each list.

    * `rank` (reciprocal rank fusion) sums `1 / (RRF_K + place)` over the lists a frame
      appears in, so a frame both models rank first beats a frame only one of them does;
    * `union` takes the BEST place instead of the sum, which is the set merge the brief
      asks for: the two top-Ns collapsed into one list, a frame found by a single model
      keeping the place that model gave it.

    Ties are broken by file_id — with `union` they are the normal case (two frames each
    ranked first by one model), and a search that reshuffles between runs cannot be
    measured. An unknown mode raises: silently ranking by something else is the failure
    this whole feature is written to avoid.
    """
    if mode not in (SEARCH_FUSION_RANK, SEARCH_FUSION_UNION):
        raise ValueError(f"fuse: unknown fusion mode {mode!r}, expected one of "
                         f"{SEARCH_FUSION_RANK!r}/{SEARCH_FUSION_UNION!r}")
    weights: dict[int, float] = {}
    for ranking in rankings:
        for place, file_id in enumerate(ranking, 1):
            weight = 1.0 / (RRF_K + place)
            if mode == SEARCH_FUSION_RANK:
                weights[file_id] = weights.get(file_id, 0.0) + weight
            else:
                weights[file_id] = max(weights.get(file_id, 0.0), weight)
    order = sorted(weights, key=lambda file_id: (-weights[file_id], file_id))
    return [(file_id, weights[file_id]) for file_id in order[:max(0, limit)]]


def rank_text(cfg: Config, conn: sqlite3.Connection, text: str, *,
              limit: int | None = None, offset: int = 0,
              encoder: TextEncoder | None = None,
              class_encoder: TextEncoder | None = None) -> Page:
    """`encode_query` + `rank` with everything the config already knows.

    The one entry point the CLI, the album, the interface and the measurement share, so
    that "which model are we comparing against" is answered in one place
    (`junk.search_index_model` — the architecture AND the weights) instead of four.
    `limit=None` means `features.search_page` — a PAGE since F173, and the config comment
    is where the difference between that and a ceiling is written down. The encoder is
    loaded only when the caller does not bring one, which is what keeps the CLIP import out
    of every module that merely imports this one.

    F141: that model is `features.search_model` and NOT `naming.clip.*`. The two are
    different on purpose — the classification model is the one every threshold in the
    pipeline is calibrated on, and the search model is the one that answers a Russian query
    — so the text tower is built from the search side's settings as well. A caller who
    brings its own encoder brings the responsibility with it; a mismatch cannot corrupt a
    ranking (the model filter and the width check in `search` see to that), it can only
    leave nothing to rank.

    F153: with `features.search_fusion` on, the same call ALSO ranks the classification
    index and merges the two lists — the call site does not change, which is why the CLI,
    the album and the web app got the feature without a line of their own. The page this
    returns is `search_fusion(...).page`; a caller that needs to know WHICH indexes
    answered calls that instead.
    """
    return search_fusion(cfg, conn, text, limit=limit, offset=offset, encoder=encoder,
                         class_encoder=class_encoder).page


def search_fusion(cfg: Config, conn: sqlite3.Connection, text: str, *,
                  limit: int | None = None, offset: int = 0,
                  encoder: TextEncoder | None = None,
                  class_encoder: TextEncoder | None = None) -> Fusion:
    """F153: `rank_text` with the merge visible — which indexes ranked, and which did not.

    With `features.search_fusion: off` this is today's search exactly: `rank` over the
    search index, the same window and the same total, and the classification index not
    read, not counted and not encoded for — so the mode is also the switch for the second
    CLIP text pass a query would otherwise pay for.

    With `rank` or `union` both indexes are ranked and `fuse` merges the positions. Each
    side is encoded by ITS OWN tower — the search model's for `search_embeddings`, the
    classification model's for `clip_embeddings` — because a query has to land in the space
    the stored vectors live in; `class_encoder` is that second tower and is built here only
    when the caller did not bring one.

    The window is taken AFTER the merge and the total is the size of the merged ranking, so
    F173's «показано N из M» stays true with a fusion on: N and M are then about a list
    neither index has on its own. That is also why both are ranked in full — see
    `_WHOLE_RANKING`.

    An index with nothing to rank does not sink the query: the other one answers, and the
    fact travels out loud — a warning in the log and the reason in `missing` — because a
    silently halved fusion is a ranking nobody can tell from a whole one. If NEITHER index
    can rank, the search index's own refusal is raised, which is the sentence the interface
    already knows how to say.
    """
    mode = fusion_mode(cfg)
    model = search_index_model(cfg)
    size = int(limit if limit is not None else cfg.features.search_page)
    start = max(0, offset)
    if encoder is None:  # pragma: no cover — ML, smoke test
        encoder = text_encoder(search_index_settings(naming_settings(cfg), model))
    vector = encode_query(text, encoder)
    if mode == SEARCH_FUSION_OFF:
        return Fusion(mode, rank(conn, vector, model, limit=size, offset=start),
                      (model,), {})

    class_model = embedding_model(naming_settings(cfg))
    if class_encoder is None:  # pragma: no cover — ML, smoke test
        class_encoder = _classification_encoder(naming_settings(cfg))
    sides: tuple[tuple[str, np.ndarray, _Ranker], ...] = (
        (model, vector, rank),
        (class_model, encode_query(text, class_encoder), rank_classification),
    )
    ranked: dict[str, list[int]] = {}
    missing: dict[str, str] = {}
    for name, query, rank_with in sides:
        try:
            ranked[name] = [file_id for file_id, _score in
                            rank_with(conn, query, name, limit=_WHOLE_RANKING).hits]
        except EmbeddingsMissing as exc:
            missing[name] = exc.reason
    if not ranked:
        raise _nothing_to_rank(conn, model, _SEARCH_TABLE)
    for name, reason in missing.items():
        _log.warning("поиск: индекс модели %r не участвует в слиянии (%s) — "
                     "ранжирует только %s", name, reason, ", ".join(ranked))
    merged = fuse(list(ranked.values()), mode, start + size)
    return Fusion(
        mode,
        Page(hits=merged[start:], offset=start, limit=size,
             total=len({file_id for ranking in ranked.values() for file_id in ranking})),
        tuple(ranked), missing)


def search_text(cfg: Config, conn: sqlite3.Connection, text: str, *,
                limit: int | None = None, offset: int = 0,
                encoder: TextEncoder | None = None,
                class_encoder: TextEncoder | None = None) -> list[tuple[int, float]]:
    """`rank_text` for a caller that wants the frames alone — the CLI and the album.

    Neither of those two draws a "show more" button, so neither has anything to do with
    the length of the ranking; they get the list they asked for and nothing to unpack.
    """
    return rank_text(cfg, conn, text, limit=limit, offset=offset, encoder=encoder,
                     class_encoder=class_encoder).hits


def file_paths(conn: sqlite3.Connection, file_ids: Sequence[int]) -> dict[int, str]:
    """file_id -> path for a result list — the printing side of a search.

    Chunked for the reason `read_clip_embeddings` gives: `features.search_page` is a
    user-set number and SQLite has a ceiling on bound parameters.
    """
    out: dict[int, str] = {}
    for part in batched(list(file_ids), 500):
        marks = ",".join("?" * len(part))
        out.update({int(r["id"]): str(r["path"]) for r in conn.execute(
            f"SELECT id, path FROM files WHERE id IN ({marks})", tuple(part))})
    return out


def _nothing_to_rank(conn: sqlite3.Connection, model: str,
                     table: str = _SEARCH_TABLE) -> EmbeddingsMissing:
    """Which of the two empty states this is — counts only, no vector is read.

    A count over the table rather than a second read of it: this runs on the path where the
    answer is already known to be empty, and what is missing is the reason, not the data.

    F141: the table counted is the SEARCH index. A collection with a full
    `clip_embeddings` and no search index is `empty` here, and correctly so — those
    vectors cannot answer this query, and saying "you have 19 757 of them" about a table
    this search will never read would be an answer to a question nobody asked.

    F153: which is why `table` is an argument now and defaults to the search index. A
    fusion asks the same question of `clip_embeddings`, and the counts have to be about the
    table that came up empty — the numbers end up in a sentence a person reads.
    """
    total = int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
    stored = int(conn.execute(
        f"SELECT COUNT(*) FROM {table} WHERE model = ?", (model,)).fetchone()[0])
    reason = REASON_OTHER_MODEL if total and not stored else REASON_EMPTY
    return EmbeddingsMissing(reason, model, total, stored)
