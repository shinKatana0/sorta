"""F129: search by words over the CLIP vectors the junk stage keeps — the engine, no interface.

F141 is why search reads `search_embeddings` and not `clip_embeddings`. Over 217
hand-labelled judgements on 8 concepts (`scripts/measure_search.py`),
`xlm-roberta-base-ViT-B-32` gives 98% precision at top-5 in Russian and 95% in English;
the classification model `ViT-L-14` gives 22% and 98%, four of eight concepts (cake,
food, mountains, children) returning nothing at all in Russian. Swapping the pipeline's
model would have invalidated the landmark (F75), animal (F122) and cascade (F130)
thresholds calibrated on its numbers, so search got a SECOND index of its own
(`features.search_model`, behind `features.search_index`). The classification vectors are
deliberately not a fallback: a ranking produced by the wrong model looks like a good one.

Three properties, each a decision. Rows of another model never enter a ranking (the
filter lives in `junk.read_search_embeddings`, F128). An empty table raises
`EmbeddingsMissing` with a reason, because "nothing was found" and "nothing was ever
computed" read identically and only one is fixed by running `sorta junk`. And this ranks
without classifying: no threshold, now or later, `features.search_page` being a SAMPLE
SIZE and `rank` returning a WINDOW plus the length of the whole ranking (F173) — depth is
the one measured lever of completeness, «дети» going from 61% to 89% when the list is
doubled. Compound queries stay the weak class: "a city at night" is 80% for both models.

The population is personal photographs only (F120), so a screenshot, or a document, has
no row here to surface. The real CLIP is loaded once, in `text_encoder`; everything else
takes an encoder as an argument, which lets the tests run the engine without a model.
"""
from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from . import accel
from .config import (
    SEARCH_FUSION_OFF,
    SEARCH_FUSION_RANK,
    SEARCH_FUSION_UNION,
    Config,
    FeaturesConfig,
)
from .faults import Fault
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

# Query strings -> their text features, one row per string, in order. Replaced in tests.
TextEncoder = Callable[[Sequence[str]], np.ndarray]

# The population of a search (F120): a file can become a duplicate or go unreadable
# AFTER its vector was stored, and neither belongs in a result list.
_CANDIDATES_SQL = """SELECT id FROM files
    WHERE dup_of IS NULL AND error IS NULL AND media_type = 'photo'"""

# Two states needing two different sentences: one is fixed by running the junk stage,
# the other by running it AGAIN after a model change.
REASON_EMPTY = "empty"              # the table holds no vectors at all
REASON_OTHER_MODEL = "other_model"  # it holds vectors, all of them of another model

_SEARCH_TABLE = "search_embeddings"  # F141: the multilingual index, search's own
_CLASS_TABLE = "clip_embeddings"     # F128: the classification vectors, ViT-L-14

# F153: the smoothing constant of reciprocal rank fusion, at the value the method was
# published with. Nothing calibrates it — any positive K keeps a frame both models put
# first ahead of one only a single model did; K only chooses how far down a second list
# may still rescue a frame.
RRF_K = 60


class EmbeddingsMissing(Fault, RuntimeError):
    """No usable vector for this query — raised instead of returning an empty list.

    The reason is a code, not a message: the sentence a user reads belongs in the i18n
    catalog with its three languages. `stored` counts the rows of the query's own model
    and `total` every row, which makes "19 757 vectors, all of another model" sayable.
    """

    codes = ("search_embeddings_empty", "search_embeddings_other_model")

    def __init__(self, reason: str, model: str, total: int, stored: int) -> None:
        super().__init__(f"{reason}: model={model!r}, stored={stored}, total={total}",
                         f"search_embeddings_{reason}",
                         reason=reason, model=repr(model), stored=stored, total=total)
        self.reason = reason
        self.model = model
        self.total = total
        self.stored = stored


def text_encoder(s: NamingSettings) -> TextEncoder:  # pragma: no cover — ML, smoke test
    """The real open_clip text tower — the same model and weights the images went through.

    Loaded through `create_model_and_transforms` like `landmarks.clip_classifier`, so that
    (architecture, checkpoint) is resolved by the same call the image side used, and
    L2-normalized here too so a dot product is a cosine. F141: for a search those settings
    are the search side's, `junk.search_index_settings(naming_settings(cfg), model)` —
    handed `naming_settings` unchanged this builds the classification tower instead, which
    the width check in `_rank` turns into nothing to rank rather than a quiet answer.
    """
    import open_clip
    import torch

    device = accel.torch_device(torch)  # F220: CUDA -> MPS -> CPU, chosen in one place
    model, _, _ = open_clip.create_model_and_transforms(
        s.clip_model, pretrained=s.clip_pretrained, device=device)
    tokenizer = open_clip.get_tokenizer(s.clip_model)
    model.eval()

    # F220: the cheapest of the retreats — a text tower over one typed phrase is
    # milliseconds on a processor, and an encoder that raises leaves every ranking
    # downstream with nothing to rank. Never fires on CUDA (accel.CpuFallback).
    fallback = accel.CpuFallback(device, lambda dev: model.to(dev),
                                 what="search: the clip text tower")

    def encode(texts: Sequence[str]) -> np.ndarray:
        def run(on_device: str) -> np.ndarray:
            with torch.no_grad():
                feats = model.encode_text(tokenizer(list(texts)).to(on_device))
                feats /= feats.norm(dim=-1, keepdim=True)
            return feats.cpu().numpy().astype(np.float32)

        return fallback.run(run)

    return encode


# F153: the classification text tower, built at most once per process — callers that
# bring their own encoder bring the SEARCH one, and building this per query would load
# ViT-L-14 on every typed word.
_class_encoders: dict[str, TextEncoder] = {}


def _classification_encoder(s: NamingSettings) -> TextEncoder:  # pragma: no cover — ML
    """The text tower of the classification model, from the cache above."""
    key = embedding_model(s)
    if key not in _class_encoders:
        _class_encoders[key] = text_encoder(s)
    return _class_encoders[key]


def encode_query(text: str, encoder: TextEncoder) -> np.ndarray:
    """A query -> a unit vector in the space the stored embeddings live in.

    Normalized even though the encoder already normalizes, for the reason
    `junk.pack_embedding` does the same: with unit vectors on both sides a search is one
    matmul and no per-row arithmetic. A zero vector is left alone, not divided by zero.
    """
    if not text.strip():
        raise ValueError("encode_query: the query is empty")
    vec = np.asarray(encoder([text.strip()]), dtype=np.float32).ravel()
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else vec


def encode_queries(texts: Sequence[str], encoder: TextEncoder) -> np.ndarray:
    """Several phrases -> ONE unit vector: the ensemble a pinned slice is ranked by (F151).

    Each phrase is normalized before the mean and the mean again after, so a phrase the
    tower answered with a longer vector does not weigh more for that alone. Blank phrases
    are dropped; a slice with nothing left raises rather than ranking by an arbitrary
    direction.

    What the ensemble does NOT do is improve accuracy — measured on 200 labelled frames,
    one phrase, three and six are within the noise of each other. The reason for the list
    is that a slice can be retuned in `config.yaml` (`config.DEFAULT_SAVED_SLICES`).
    """
    wanted = [t.strip() for t in texts if t and t.strip()]
    if not wanted:
        raise ValueError("encode_queries: the slice carries no query")
    # A COPY (`np.array`, not `np.asarray`): the rows are normalized in place below, and
    # an encoder answering out of a buffer of its own would have it rewritten underneath.
    matrix = np.array(encoder(wanted), dtype=np.float32).reshape(len(wanted), -1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = np.divide(matrix, norms, out=matrix, where=norms > 0)  # zero rows: left alone
    mean = matrix.mean(axis=0)
    norm = float(np.linalg.norm(mean))
    return mean / norm if norm > 0 else mean


@dataclass(frozen=True)
class Page:
    """One window of a ranking, and how long the ranking it came out of is.

    F173: `total` is the point — a list of exactly `limit` frames says nothing about
    whether the ranking ended there or was cut there. Everything «показано N из M» and the
    "show more" button need is here, so neither can be recomputed differently elsewhere.
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

    The score is a dot product of two unit vectors, so it IS the cosine. Ties break by
    file_id (the ids are sorted before the stable argsort), and that determinism is what
    makes paging honest (F173): the whole list is scored and ordered on every call and the
    window taken afterwards, so the second page continues the first. Only the window
    becomes Python objects — materializing 300 000 tuples to hand back 200 is the one part
    that scales badly. A row whose width does not match is dropped, which is also the last
    guard against a query from the classification tower (768) reaching the index (512).
    """
    q = np.asarray(query, dtype=np.float32).ravel()
    candidates = [int(r["id"]) for r in conn.execute(_CANDIDATES_SQL)]
    return _rank(conn, read_search_embeddings(conn, model, candidates), q, model,
                 limit=limit, offset=offset, table=_SEARCH_TABLE)


def rank_classification(conn: sqlite3.Connection, query: np.ndarray, model: str, *,
                        limit: int, offset: int = 0) -> Page:
    """F153: the same ranking over the CLASSIFICATION index (`clip_embeddings`).

    A function of its own rather than a table argument on `rank`: a parameter choosing
    which table the model filter applies to is one call site away from the wrong one. The
    query must come from the CLASSIFICATION tower — a search-model query is 512 numbers
    against these 768. Only ever reached through `features.search_fusion`; on its own this
    index is NOT a fallback for search (F141).
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
    """`rank` for a caller that wants the frames and not the length of the list."""
    return rank(conn, query, model, limit=limit, offset=offset).hits


def search_classification(conn: sqlite3.Connection, query: np.ndarray, model: str,
                          limit: int, offset: int = 0) -> list[tuple[int, float]]:
    """`rank_classification` for a caller that wants the frames alone (F153)."""
    return rank_classification(conn, query, model, limit=limit, offset=offset).hits


# --- F153: two indexes, one answer -----------------------------------------------------
# Both models score 88/96/98% at ranks 1/3/5 apiece and return DIFFERENT frames for the
# same word, which is the case where merging beats either half. Whether it should be the
# default is a question for `measure_search.py --fusion`, which prints RECALL.

_Ranker = Callable[..., Page]

# How deep each index is ranked before the merge: all of it. A merge of two TRUNCATED
# lists cannot be windowed correctly — a frame just below the cut in both outranks a frame
# inside the cut in one — nor state a total. Two full lists of tuples per query, no image.
_WHOLE_RANKING = 1 << 30


@dataclass(frozen=True)
class Fusion:
    """One answer of the search, plus which indexes are behind it (F153).

    `used`/`missing` exist because a merge can be a merge of one: an index with nothing to
    rank must not turn into a quietly shorter answer, so the models that ranked are named
    and the ones that did not carry a reason code (`REASON_EMPTY`/`REASON_OTHER_MODEL`).
    What a `score` MEANS depends on `mode`: with `off` the cosine of the search model,
    with a fusion a weight computed from POSITIONS that belongs to no vector space at all.
    """
    mode: str
    page: Page
    used: tuple[str, ...]
    missing: dict[str, str]

    @property
    def hits(self) -> list[tuple[int, float]]:
        """The window itself, for a caller that does not page."""
        return self.page.hits


def fusion_mode(cfg: Config) -> str:
    """`features.search_fusion` — `off` | `rank` | `union` (default: `off`)."""
    features = getattr(cfg, "features", None) or FeaturesConfig()
    return str(getattr(features, "search_fusion", FeaturesConfig.search_fusion))


def fuse(rankings: Sequence[Sequence[int]], mode: str,
         limit: int) -> list[tuple[int, float]]:
    """Several rankings -> one, by POSITION alone: (file_id, weight), best first.

    File ids and nothing else, which is the invariant: a cosine of ViT-L-14 and one of
    xlm-roberta-base-ViT-B-32 belong to different spaces and print alike, so a function
    that never receives them cannot add or compare them. `rank` (reciprocal rank fusion)
    sums `1 / (RRF_K + place)` over the lists a frame appears in, so a frame both models
    rank first beats a frame only one of them does; `union` takes the BEST place instead
    of the sum. Ties break by file_id; an unknown mode raises.
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
    "which model are we comparing against" is answered once (`junk.search_index_model`).
    `limit=None` means `features.search_page`. The encoder is loaded only when the caller
    does not bring one, which keeps the CLIP import out of every module importing this.
    F141: that model is `features.search_model` and NOT `naming.clip.*`, so the text tower
    is built from the search side's settings too. A caller that needs to know WHICH
    indexes answered calls `search_fusion` instead.
    """
    return search_fusion(cfg, conn, text, limit=limit, offset=offset, encoder=encoder,
                         class_encoder=class_encoder).page


def search_fusion(cfg: Config, conn: sqlite3.Connection, text: str, *,
                  limit: int | None = None, offset: int = 0,
                  encoder: TextEncoder | None = None,
                  class_encoder: TextEncoder | None = None) -> Fusion:
    """F153: `rank_text` with the merge visible — which indexes ranked, and which did not.

    With `off` the classification index is not read, not counted and not encoded for, so
    the mode is also the switch for the second CLIP text pass. With `rank` or `union` both
    are ranked and `fuse` merges the positions, each side encoded by ITS OWN tower.

    The window is taken AFTER the merge and the total is the size of the merged ranking,
    so «показано N из M» stays true with a fusion on — which is why both are ranked in
    full (see `_WHOLE_RANKING`). An index with nothing to rank does not sink the query:
    the other answers and the fact travels out loud, because a silently halved fusion
    cannot be told from a whole one. If NEITHER can rank, the search index's refusal is
    raised.
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
        _log.warning("search: the index of model %r is left out of the fusion (%s) — "
                     "only %s ranks", name, reason, ", ".join(ranked))
    merged = fuse(list(ranked.values()), mode, start + size)
    return Fusion(
        mode,
        Page(hits=merged[start:], offset=start, limit=size,
             total=len({file_id for ranking in ranked.values() for file_id in ranking})),
        tuple(ranked), missing)


def rank_queries(cfg: Config, conn: sqlite3.Connection, texts: Sequence[str], *,
                 limit: int | None = None, offset: int = 0,
                 encoder: TextEncoder | None = None) -> Page:
    """`rank_text` for a PINNED slice — the same ranking, asked by several phrases (F151).

    A saved slice is a saved query, so it goes down this path and not a second engine, and
    without a threshold of its own — which each of the six hand-written filters it
    replaces had. On the sample that decided it, asking the vectors beats the filter that
    was there (animals 60% recall against 33%) and creates the two slices that were not
    there at all (children 61%, products 65%).
    """
    model = search_index_model(cfg)
    if encoder is None:  # pragma: no cover — ML, smoke test
        encoder = text_encoder(search_index_settings(naming_settings(cfg), model))
    vector = encode_queries(texts, encoder)
    return rank(conn, vector, model, offset=offset,
                limit=int(limit if limit is not None else cfg.features.search_page))


def search_text(cfg: Config, conn: sqlite3.Connection, text: str, *,
                limit: int | None = None, offset: int = 0,
                encoder: TextEncoder | None = None,
                class_encoder: TextEncoder | None = None) -> list[tuple[int, float]]:
    """`rank_text` for a caller that wants the frames alone — the CLI and the album."""
    return rank_text(cfg, conn, text, limit=limit, offset=offset, encoder=encoder,
                     class_encoder=class_encoder).hits


# --- F189: a name is a selection, not a query ------------------------------------------
# A query over the embeddings is a RANKING; a person's name is an EXACT SELECTION. The two
# are never merged into one list, because an exact selection presented like the top of a
# ranking is read as one. So a name gives a LIST: no threshold, no depth, no "show more by
# relevance" — paging by count is another matter.

# `sorter._CTE` is the source of truth for what a person's frames ARE: it resolves the
# `merged_into` chains and takes the label off the ROOT (F31). The condition below is the
# album's own subject condition for `kind='person'`, so a search by a name and an album of
# that name cannot disagree by a frame — a test compares the two SETS.
_PERSON_FILES_SQL = """SELECT f.id FROM files f
    WHERE f.dup_of IS NULL AND f.error IS NULL
      AND f.id IN (SELECT file_id FROM _person_files
                   WHERE casefold(label) = casefold(?))
    ORDER BY f.id"""

# Whether the typed string IS a name. Only ROOT clusters: a cluster that was merged away
# keeps its own row, and a name left behind on a swallowed cluster names nobody.
# `casefold` is the project's UDF — SQLite's NOCASE is ASCII-only and these names are
# usually Cyrillic — which with the `strip()` in `match_person` is the whole of "«ирина »
# and «Ирина» are one name".
_PERSON_LABEL_SQL = """SELECT label FROM face_clusters
    WHERE merged_into IS NULL AND label IS NOT NULL AND casefold(label) = casefold(?)
    ORDER BY id LIMIT 1"""

# A `Page` carries (file_id, score) pairs because a ranking has scores; a selection has
# none. Callers are expected to leave this off the screen: a "closeness 0.000" under an
# exact answer would be a measurement nobody made.
PERSON_NO_SCORE = 0.0


def _person_selection(conn: sqlite3.Connection) -> str:
    """The album's person CTE, with the `casefold` UDF registered on this connection.

    Imported inside the function: `sorter` imports THIS module, so a top-level import
    would be a cycle.
    """
    from .sorter import _CTE, _sql_casefold
    conn.create_function("casefold", 1, _sql_casefold, deterministic=True)
    return _CTE


def match_person(conn: sqlite3.Connection, text: str) -> str | None:
    """The typed string as a person's NAME — the label as stored, or None.

    None is the ordinary case: a string that names nobody goes on to the ranking untouched
    rather than producing an empty "no frames of this person" screen. The match is exact
    apart from case and surrounding blanks, nothing fuzzy — «Ира» does not find «Ирина»,
    because a near-miss puts somebody else's frames under your child's name. Ambiguity
    (two root clusters named alike) resolves to the lowest id, and both select the same
    frames anyway: the album's condition matches by label, not by cluster.
    """
    name = text.strip()
    if not name:
        return None
    _person_selection(conn)  # for the `casefold` UDF alone
    row = conn.execute(_PERSON_LABEL_SQL, (name,)).fetchone()
    return str(row["label"]) if row is not None else None


def person_files(conn: sqlite3.Connection, label: str) -> list[int]:
    """Every canonical frame of this person, in index order — the album's own selection.

    Ordered by id: there is no ranking here and paging needs a stable order.
    """
    rows = conn.execute(_person_selection(conn) + _PERSON_FILES_SQL, (label,)).fetchall()
    return [int(r["id"]) for r in rows]


def person_page(conn: sqlite3.Connection, label: str, *, limit: int,
                offset: int = 0) -> Page:
    """One window of that selection, in the shape every caller already pages through.

    A `Page` and not a list of its own, so «показано N из M» and the "show more" button
    are drawn by the same code whatever produced the frames. The window is cut in Python —
    a person's frames are thousands at the very most, not the collection.
    """
    ids = person_files(conn, label)
    start = max(0, offset)
    size = max(0, limit)
    return Page(hits=[(file_id, PERSON_NO_SCORE) for file_id in ids[start:start + size]],
                total=len(ids), offset=start, limit=size)


def file_paths(conn: sqlite3.Connection, file_ids: Sequence[int]) -> dict[int, str]:
    """file_id -> path for a result list.

    Chunked: `features.search_page` is a user-set number and SQLite has a ceiling on
    bound parameters.
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

    F141: the table defaults to the SEARCH index, so a collection with a full
    `clip_embeddings` and no search index reads as `empty` — those vectors cannot answer
    this query. F153 made `table` an argument for the fusion's sake.
    """
    total = int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
    stored = int(conn.execute(
        f"SELECT COUNT(*) FROM {table} WHERE model = ?", (model,)).fetchone()[0])
    reason = REASON_OTHER_MODEL if total and not stored else REASON_EMPTY
    return EmbeddingsMissing(reason, model, total, stored)
