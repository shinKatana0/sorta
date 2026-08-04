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

F151 adds a caller and not a mechanism. A PINNED slice — «дети», «товары» — is a saved
query: a list of English phrases out of `features.saved_slices`, averaged into one
direction by `encode_queries` and ranked by `rank_queries` down this very path. It gets
the model filter, the reason instead of an empty list, the deterministic order and the
window because it is a query; what it does not get, and what the six hand-written filters
it replaces each had, is a threshold of its own. On the sample that decided it, asking the
vectors beats the filter that was there (animals 60% recall against 33%) and creates the
two slices that were not there at all (children 61%, products 65%).

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

The real CLIP is loaded exactly once, in `text_encoder`; everything else takes an encoder
as an argument, which is what lets the tests run the whole engine without a model.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .config import Config
from .junk import read_search_embeddings, search_index_model, search_index_settings
from .landmarks import batched
from .naming import NamingSettings, naming_settings

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


def encode_queries(texts: Sequence[str], encoder: TextEncoder) -> np.ndarray:
    """Several phrases -> ONE unit vector: the ensemble a pinned slice is ranked by (F151).

    Each phrase is brought to a norm of 1 before the mean, and the mean is normalized
    again. Both steps matter for the same reason `encode_query` normalizes: the result has
    to be a unit vector so that a dot product against the stored rows is a cosine, and a
    phrase the tower answered with a longer vector must not weigh more than its neighbours
    merely for that.

    The averaging is what makes the list a list: three phrases give a direction none of
    them has on its own, which is the difference between a slice and a query somebody
    typed. What the ensemble does NOT do is improve accuracy — measured on 200 labelled
    frames, one phrase, three and six are within the noise of each other — so the reason
    for the list is that a slice can be retuned in `config.yaml` (see
    `config.DEFAULT_SAVED_SLICES`).

    The whole ensemble goes to the tower in ONE call: the phrases of a slice are known
    together, and a call per phrase is a load of the same model N times over in the CLI.
    Blank phrases are dropped; a slice with nothing left raises, because a pinned query
    with no words would rank the collection by an arbitrary direction and look like an
    answer.
    """
    wanted = [t.strip() for t in texts if t and t.strip()]
    if not wanted:
        raise ValueError("encode_queries: the slice carries no query")
    matrix = np.asarray(encoder(wanted), dtype=np.float32).reshape(len(wanted), -1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # A zero row is left alone rather than divided by zero — it contributes no direction,
    # which is the honest outcome of a phrase the tower answered with nothing.
    matrix = np.divide(matrix, norms, out=matrix, where=norms > 0)
    mean = matrix.mean(axis=0)
    norm = float(np.linalg.norm(mean))
    return mean / norm if norm > 0 else mean


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
    """
    q = np.asarray(query, dtype=np.float32).ravel()
    candidates = [int(r["id"]) for r in conn.execute(_CANDIDATES_SQL)]
    vectors = read_search_embeddings(conn, model, candidates)
    ids = sorted(fid for fid, vec in vectors.items() if vec.size == q.size)
    if not ids:
        raise _nothing_to_rank(conn, model)
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


def rank_text(cfg: Config, conn: sqlite3.Connection, text: str, *,
              limit: int | None = None, offset: int = 0,
              encoder: TextEncoder | None = None) -> Page:
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
    """
    model = search_index_model(cfg)
    if encoder is None:  # pragma: no cover — ML, smoke test
        encoder = text_encoder(search_index_settings(naming_settings(cfg), model))
    vector = encode_query(text, encoder)
    return rank(conn, vector, model, offset=offset,
                limit=int(limit if limit is not None else cfg.features.search_page))


def rank_queries(cfg: Config, conn: sqlite3.Connection, texts: Sequence[str], *,
                 limit: int | None = None, offset: int = 0,
                 encoder: TextEncoder | None = None) -> Page:
    """`rank_text` for a PINNED slice — the same ranking, asked by several phrases (F151).

    Deliberately the same path and not a second engine: a saved slice is a saved query, so
    it gets the model filter, the width check, the reason instead of an empty list, the
    deterministic order and the window — everything a typed query gets — and the only
    difference between the two callers is how the vector was built. A slice that ranked by
    rules of its own would be the sixth filter this feature exists to remove.
    """
    model = search_index_model(cfg)
    if encoder is None:  # pragma: no cover — ML, smoke test
        encoder = text_encoder(search_index_settings(naming_settings(cfg), model))
    vector = encode_queries(texts, encoder)
    return rank(conn, vector, model, offset=offset,
                limit=int(limit if limit is not None else cfg.features.search_page))


def search_text(cfg: Config, conn: sqlite3.Connection, text: str, *,
                limit: int | None = None, offset: int = 0,
                encoder: TextEncoder | None = None) -> list[tuple[int, float]]:
    """`rank_text` for a caller that wants the frames alone — the CLI and the album.

    Neither of those two draws a "show more" button, so neither has anything to do with
    the length of the ranking; they get the list they asked for and nothing to unpack.
    """
    return rank_text(cfg, conn, text, limit=limit, offset=offset, encoder=encoder).hits


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


def _nothing_to_rank(conn: sqlite3.Connection, model: str) -> EmbeddingsMissing:
    """Which of the two empty states this is — counts only, no vector is read.

    A count over the table rather than a second read of it: this runs on the path where the
    answer is already known to be empty, and what is missing is the reason, not the data.

    F141: the table counted is the SEARCH index. A collection with a full
    `clip_embeddings` and no search index is `empty` here, and correctly so — those
    vectors cannot answer this query, and saying "you have 19 757 of them" about a table
    this search will never read would be an answer to a question nobody asked.
    """
    total = int(conn.execute("SELECT COUNT(*) FROM search_embeddings").fetchone()[0])
    stored = int(conn.execute(
        "SELECT COUNT(*) FROM search_embeddings WHERE model = ?", (model,)).fetchone()[0])
    reason = REASON_OTHER_MODEL if total and not stored else REASON_EMPTY
    return EmbeddingsMissing(reason, model, total, stored)
