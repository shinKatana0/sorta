"""F129: search by words over the CLIP vectors F128 kept — the engine, without an interface.

Every slice of the collection is written code today: a tab for animals, a tab for people,
a tab for products. A search turns a slice into a QUERY — someone types a word and gets
one — so "food", "snow", "the sea" stop being a programmer's work. What that stands on is
already on disk: `clip_embeddings` holds an L2-normalized CLIP vector per canonical
photograph together with the name of the model that produced it (F128).

Three properties are the whole feature, and each of them is a decision rather than an
implementation detail:

* **The model filter is not optional.** A vector computed by another model is not
  comparable with this query, and mixing the two produces a plausible ranking that nothing
  in the output marks as wrong. Rows of another model never enter the ranking; if that
  leaves nothing to rank, the caller is told so (`EmbeddingsMissing`) instead of being
  handed a short list. The filter itself lives where F128 put it — inside
  `junk.read_clip_embeddings`, the one function that reads the table.
* **An empty table is a reason, not an empty result.** "Nothing was found" and "nothing was
  ever computed" read identically in a list of zero lines, and only one of them is fixed by
  running `sorta junk`.
* **This ranks, it does not classify.** There is no "this really is a cake" threshold and
  there will not be one, for the same reason sharpness has none: the score orders frames
  against each other and says nothing in absolute terms. `features.search_limit` is
  therefore a SAMPLE SIZE, not a cutoff.

Known limits, measured elsewhere and repeated here so a caller does not have to guess:
compound queries ("a cake with candles on a table by the window") are weak — CLIP takes a
sentence as one whole and single subjects are what it does well. The population is
personal photographs only (F120): a screenshot's vector is noise in a search over a family
archive, and it is not in the table to begin with — which also means a document (a
passport, a medical form) can never surface here, because F128 stores no row for one.

The accuracy of this search on a real collection has NOT been measured. That is what
`scripts/measure_search.py` is for, and F121/F122 is why it is not optional: the animal
class looked like it worked until 320 hand-labelled frames showed that only half of the
question was right.

The real CLIP is loaded exactly once, in `text_encoder`; everything else takes an encoder
as an argument, which is what lets the tests run the whole engine without a model.
"""
from __future__ import annotations

import sqlite3
from typing import Callable, Sequence

import numpy as np

from .config import Config
from .junk import embedding_model, read_clip_embeddings
from .naming import NamingSettings, naming_settings

# Query strings -> their text features, one row per string, in the same order. Replaced in
# tests; the real one is `text_encoder` below — the project's own open_clip, never a second
# model, because a query has to land in the space the stored vectors live in.
TextEncoder = Callable[[Sequence[str]], np.ndarray]

# The population of a search (F120), and the reason it is spelled out here rather than left
# to the table: `clip_embeddings` is written by the junk stage and only ever holds personal
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


def search(conn: sqlite3.Connection, query: np.ndarray, model: str,
           limit: int) -> list[tuple[int, float]]:
    """The ranking: (file_id, score) for the `limit` nearest photographs, best first.

    The score is a dot product — both sides are unit vectors, so it IS the cosine, and no
    normalization happens per row. Ties are broken by file_id (the ids are sorted before the
    stable argsort), which is what makes a repeated search return the same list rather than
    whatever order the dict happened to have: a ranking that reshuffles between runs cannot
    be measured, and measuring it is a condition of this feature.

    Vectors of another model are absent by construction — `read_clip_embeddings` filters on
    `model` — and a row whose width does not match the query is dropped as well: a truncated
    blob is a broken row, not a reason for the whole search to fail.
    """
    q = np.asarray(query, dtype=np.float32).ravel()
    candidates = [int(r["id"]) for r in conn.execute(_CANDIDATES_SQL)]
    vectors = read_clip_embeddings(conn, model, candidates)
    ids = sorted(fid for fid, vec in vectors.items() if vec.size == q.size)
    if not ids:
        raise _nothing_to_rank(conn, model)
    matrix = np.stack([vectors[fid] for fid in ids])
    scores = matrix @ q
    order = np.argsort(-scores, kind="stable")[:max(0, limit)]
    return [(ids[i], float(scores[i])) for i in order]


def search_text(cfg: Config, conn: sqlite3.Connection, text: str, *,
                limit: int | None = None,
                encoder: TextEncoder | None = None) -> list[tuple[int, float]]:
    """`encode_query` + `search` with everything the config already knows.

    The one entry point the CLI, the album and the measurement share, so that "which model
    are we comparing against" is answered in one place (`junk.embedding_model` — the
    architecture AND the weights) instead of three. `limit=None` means
    `features.search_limit`; the encoder is built on demand, so a caller that never reaches
    the model — an empty table, a bad query — does not pay for loading it.
    """
    if encoder is None:  # pragma: no cover — ML, smoke test
        encoder = text_encoder(naming_settings(cfg))
    vector = encode_query(text, encoder)
    model = embedding_model(naming_settings(cfg))
    return search(conn, vector, model,
                  int(limit if limit is not None else cfg.features.search_limit))


def file_paths(conn: sqlite3.Connection, file_ids: Sequence[int]) -> dict[int, str]:
    """file_id -> path for a result list, in one query — the printing side of a search."""
    if not file_ids:
        return {}
    marks = ",".join("?" * len(file_ids))
    return {int(r["id"]): str(r["path"]) for r in conn.execute(
        f"SELECT id, path FROM files WHERE id IN ({marks})", tuple(file_ids))}


def _nothing_to_rank(conn: sqlite3.Connection, model: str) -> EmbeddingsMissing:
    """Which of the two empty states this is — counts only, no vector is read.

    A count over the table rather than a second read of it: this runs on the path where the
    answer is already known to be empty, and what is missing is the reason, not the data.
    """
    total = int(conn.execute("SELECT COUNT(*) FROM clip_embeddings").fetchone()[0])
    stored = int(conn.execute(
        "SELECT COUNT(*) FROM clip_embeddings WHERE model = ?", (model,)).fetchone()[0])
    reason = REASON_OTHER_MODEL if total and not stored else REASON_EMPTY
    return EmbeddingsMissing(reason, model, total, stored)
