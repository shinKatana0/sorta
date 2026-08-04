"""F184: the products the model SAW and called photographs — does a query find them?

The hand-labelled measurement of 2026-08-03 (999 frames, three layers, shuffled) put
numbers on where the completeness of the product slice is lost:

    products in the archive   ~2 036   (9.2% of it)
    found                      1 649   RECALL 81%
    missed by the MODEL          288   <- this script
    missed by the GATE           100   closed: 3.6 hours for a hundred finds

288 against 100 — the loss on the ANSWER is three times the loss on the selection. Those
frames reached the model, it looked at them and said "a photograph", so the lever here is
not the threshold: a threshold decides who gets asked, and these were asked already.

The hypothesis this measurement is built on is B of the brief — a second opinion from the
vectors, which COSTS NOTHING. `clip_embeddings` (F128) are computed by the fast tier for
the whole collection it keeps vectors for (the photograph population — see the note on
privacy below), so asking them a product query is arithmetic over ready vectors: not a
single model call, not a single pass over an image. The recall measurement of 2026-08-02
says it is the likely answer as well: products by query came out at 88%, above the 81% of
the whole current path. A query and a VLM are wrong in different places, and that is exactly
the case where a second opinion is cheaper than improving the first.

What is printed, in the order the brief asks for it:

1. How many of the missed products the query reaches AT SEVERAL DEPTHS. Depth is the one
   confirmed lever of completeness (doubling the list gives +25 points), so the grid
   doubles and is a parameter, never a literal.
2. The PRICE: how many frames a person has to look at to collect those finds. Recall
   without a price is half an answer.
3. A breakdown of the missed frames themselves BY REASON (hypothesis A of the brief) over
   a sample somebody can go through by eye — see `--write-reasons`.
4. A BASELINE — a random draw of the same depth from the same population. Without it
   "the query finds N" has nothing to be compared against.

Privacy, and here it is stricter than usual. Products sit next to the `document` class:
passports, certificates, medical forms. The eye list excludes `verdict = 'document'` IN
THE QUERY (see `EYE_SQL`) rather than in the marking — the F133 rule, a hidden line is not
a rule. Everything else printed is counts; paths appear only with `--paths`, which is the
one job that cannot be done without seeing the frames.

The index population is a second, independent reason the same thing holds, and it is worth
knowing before the price column is read: both vector tables are written for personal
photographs only (F120), so on the live collection `clip_embeddings` holds 19 212 rows and
every one of them is a frame with `verdict = 'photo'`. A document cannot be ranked at all,
a product the pipeline already filed is not in the list to inflate its price, and the 288
are all rankable by construction — they ARE frames the pipeline calls photographs. The
`shown` / `hidden` columns keep counting anyway, because that is a property of today's
population and not of the report.

Nothing here is a private copy of the feature: the phrases come from
`features.saved_slices` (F151), the vector from `search.encode_queries` and the ranking
from `search.search_classification` / `search.search`. A copy would measure this script
instead of the engine.

Nothing is written to the database.

Usage (from the repo root; a CPU venv is enough — `uv sync --extra cpu --extra dev` — since
only the text tower of CLIP runs and no image is ever opened):
    python scripts/measure_product_recall.py --labels marks.json
    python scripts/measure_product_recall.py --labels marks.json --depths 200,400,800
    python scripts/measure_product_recall.py --labels marks.json --write-reasons why.json
    python scripts/measure_product_recall.py --labels marks.json --reasons why.json
"""
from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sorta import search  # noqa: E402
from sorta.config import DEFAULT_SAVED_SLICES, Config, load_config  # noqa: E402
from sorta.db import connect  # noqa: E402
from sorta.junk import (  # noqa: E402
    embedding_model,
    search_index_model,
    search_index_settings,
)
from sorta.landmarks import batched  # noqa: E402
from sorta.naming import naming_settings  # noqa: E402

PRODUCT_VERDICT = "product"
DOCUMENT_VERDICT = "document"

# `media_class.source` of a frame the model answered about. Everything else in that column
# means the fast tier decided alone, i.e. the gate never let the frame through.
VLM_SOURCE = "vlm"

# The slice whose phrases this query is asked with — `features.saved_slices` (F151).
PRODUCT_SLICE = "products"

# The two indexes a ranking can come out of. `class` is the default and the reason the
# whole feature is cheap: `clip_embeddings` (F128) are computed for the entire collection
# by the fast tier, whatever the settings say. `search` is the multilingual index of F141
# and exists here because the earlier query measurements were made on it — it is filled
# only with `features.search_index: true`, so it cannot be the default of a measurement
# that advertises itself as free.
INDEX_CLASS = "class"
INDEX_SEARCH = "search"

# The depth grid, doubling. Doubling is not decoration: it is the only lever of
# completeness the measurements confirmed (the query «дети» goes 61% -> 89% when the list
# is doubled), so a grid that grows any other way would hide the effect it is here to show.
DEFAULT_DEPTHS = (200, 400, 800, 1600, 3200)

# Fixed so that a rerun draws the same baseline and the same eye sample as the run being
# argued about. The date the 999 frames were labelled on.
DEFAULT_SEED = 20260803

# How many of the missed frames go into the eye list by default. Enough to see which of the
# three reasons dominates, few enough that a person actually goes through all of them.
DEFAULT_EYE_SAMPLE = 60

# The reasons of hypothesis A, as the brief separates them. The codes are what a person
# writes into the worksheet; the sentences are what the report prints next to the counts.
# A reason outside this vocabulary is counted under its own name rather than dropped — the
# person looking at the frames may see something the brief did not predict, and losing that
# is worse than an unplanned line in a table.
REASONS: dict[str, str] = {
    "borderline": "пограничный по существу: вещь на столе дома — товар или быт",
    "narrow": "вопрос узок: витрина на скриншоте, вещь в руках, коллаж из ракурсов",
    "feature_missing": "нужен признак, которого в кадре нет: ценник, белый фон",
    "other": "ни одна из трёх причин",
}

# The whole ranking, not a page of it: the depth grid, the price and the random baseline
# are all read off ONE ranked list, and the baseline is only honest if it is drawn from the
# same population the list came out of. `search.rank` materializes just the window, so the
# number is a window that covers everything rather than a promise about the collection.
WHOLE_RANKING = 1 << 30

# --- Pre-registered acceptance criteria (F184, phase 0) ----------------------
#
# Written down before the first run, so the table cannot talk anybody into an outcome
# afterwards. They are about the RANKING, not about the price: what a person is willing to
# pay per find is their call and the price column is printed for it, but whether the
# vectors know anything about these 288 frames at all is a question with a number.
#
# A — the query is a second opinion: at the deepest measured list it reaches at least half
#     of the missed products AND finds at least twice what a random draw of the same depth
#     does. The next feature is a review list at that depth.
# B — the signal is real but the list is too short: the lift holds, the recall does not.
#     Depth is the confirmed lever, so the answer is a deeper list, not another question.
# C — no lift: the query fails on the very frames the model fails on. Then the answer is in
#     the reason breakdown of hypothesis A, and no amount of depth will help.
RECALL_MIN = 0.5
LIFT_MIN = 2.0

# The eye list is built with `verdict = 'document'` excluded HERE, in the query, and not by
# a filter somebody has to remember to apply afterwards (F133). A frame with no
# classification row at all stays in: it is unclassified, not a document.
EYE_SQL = """SELECT f.id AS id, f.path AS path
             FROM files f LEFT JOIN media_class mc ON mc.file_id = f.id
             WHERE f.id IN ({marks})
               AND (mc.verdict IS NULL OR mc.verdict != '{document}')"""


@dataclass(frozen=True)
class Frame:
    """One hand-labelled frame, next to what the pipeline decided about it.

    `is_product` is the human's mark — the truth this measurement is against. `verdict` is
    what the index holds now (None: the frame has no classification row). `asked` is the
    fact that separates the two kinds of loss: the model answered about this frame, so a
    product it did not recognize is a miss of the ANSWER and not of the gate.
    """
    file_id: int
    verdict: str | None
    asked: bool
    is_product: bool

    @property
    def found(self) -> bool:
        """The current path got it: the frame is filed as a product today."""
        return self.is_product and self.verdict == PRODUCT_VERDICT

    @property
    def missed_by_model(self) -> bool:
        """THE 288: shown to the model, answered about, and not a product afterwards."""
        return self.is_product and self.asked and self.verdict != PRODUCT_VERDICT

    @property
    def missed_by_gate(self) -> bool:
        """The other loss: the model was never asked. Closed by the brief, counted here
        only so the two numbers stay next to each other."""
        return self.is_product and not self.asked and self.verdict != PRODUCT_VERDICT


@dataclass(frozen=True)
class Row:
    """One depth of the list: what it reaches, what it costs and what chance would give.

    `shown` is the price in frames a person has to judge: everything in the prefix that is
    not already filed as a product (nothing to review there) and is not a document (never
    shown — see the privacy note). `hidden` is how many of those documents there were, as a
    count, so that `shown` being smaller than the depth is explained rather than mysterious.
    """
    depth: int
    found: int
    shown: int
    hidden: int
    random_found: int
    misses: int

    @property
    def recall(self) -> float:
        return self.found / self.misses if self.misses else 0.0

    @property
    def random_recall(self) -> float:
        return self.random_found / self.misses if self.misses else 0.0

    @property
    def price(self) -> float | None:
        """Frames a person looks at per product recovered (None — nothing recovered)."""
        return self.shown / self.found if self.found else None

    @property
    def lift(self) -> float | None:
        """How many times the ranking beats chance at this depth (None — chance found
        nothing, and a ratio over zero is not a number worth printing)."""
        return self.found / self.random_found if self.random_found else None


def _as_mark(value: object) -> bool:
    """One cell of the sheet -> is this frame a product.

    Words are accepted next to booleans because the sheet is filled in by a person and by
    whatever tool the layer was collected with: `true`, `1`, `product` all mean the same
    thing. Anything unrecognized stops the run instead of quietly becoming `false` — a
    typo that reads as "not a product" would shrink the very set being measured.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in ("true", "1", "yes", "y", "product", "товар"):
        return True
    if text in ("false", "0", "no", "n", "photo", "фото"):
        return False
    raise SystemExit(f"разметка: непонятная отметка {value!r} — ожидается true/false "
                     f"(или product/photo)")


def _collect_labels(data: object, out: dict[int, bool]) -> None:
    """One level of the sheet, into `out`. Cells still holding `null` are not labelled."""
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, (dict, list)):  # one layer of the three-layer sample
                _collect_labels(value, out)
            elif value is not None:
                out[int(key)] = _as_mark(value)
    elif isinstance(data, list):
        for row in data:
            if isinstance(row, dict):
                mark = row.get("product", row.get("label"))
                if mark is not None:
                    out[int(row["file_id"])] = _as_mark(mark)
            elif row[1] is not None:
                out[int(row[0])] = _as_mark(row[1])
    else:
        raise SystemExit("разметка: не похоже на лист отметок — ожидается отображение "
                         "«file_id: true/false», список строк или слои из таких листов")


def load_labels(path: Path) -> dict[int, bool]:
    """The hand labels -> {file_id: is this frame really a product}.

    The sheet is produced outside this repo (the 999 frames of 2026-08-03 were labelled in
    three layers), so three shapes are accepted: a flat `{file_id: mark}`, a list of rows,
    and a mapping of LAYER NAME to either of those — the layers of that sample kept apart,
    which is also the shape `measure_search.py --write-labels` writes. They are merged:
    the layer a frame came from decided who got asked, and this script reads that from the
    index instead.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "rows" in data:
        data = data["rows"]
    out: dict[int, bool] = {}
    _collect_labels(data, out)
    if not out:
        raise SystemExit(f"{path}: в листе нет ни одной отметки — измерять нечего")
    return out


def labelled_frames(conn: sqlite3.Connection, labels: dict[int, bool]) -> list[Frame]:
    """The labelled sample joined with the index: the verdict now and who decided it.

    Chunked for the reason `search.file_paths` gives — the sheet is a user-sized list and
    SQLite has a ceiling on bound parameters. Duplicates and unreadable files drop out:
    they are outside every population in this project, and a mark on one of them would be
    counted against a frame the product never shows anybody.
    """
    frames: list[Frame] = []
    for part in batched(sorted(labels), 500):
        marks = ",".join("?" * len(part))
        for r in conn.execute(
            f"""SELECT f.id AS id, mc.verdict AS verdict, mc.source AS source
                FROM files f LEFT JOIN media_class mc ON mc.file_id = f.id
                WHERE f.id IN ({marks}) AND f.dup_of IS NULL AND f.error IS NULL""",
            tuple(part),
        ):
            frames.append(Frame(
                file_id=int(r["id"]),
                verdict=None if r["verdict"] is None else str(r["verdict"]),
                asked=str(r["source"]) == VLM_SOURCE,
                is_product=labels[int(r["id"])]))
    return frames


def all_verdicts(conn: sqlite3.Connection) -> dict[int, str]:
    """{file_id: verdict} for the whole index — what the price of a list is computed from.

    A frame with no row here carries no verdict, and `Row.shown` counts it as one a person
    still has to look at: an unclassified frame is not a frame already filed as a product.
    """
    return {int(r["file_id"]): str(r["verdict"])
            for r in conn.execute("SELECT file_id, verdict FROM media_class")}


def eye_candidates(conn: sqlite3.Connection, file_ids: Sequence[int]) -> dict[int, str]:
    """{file_id: path} of the frames that may be LOOKED AT — documents excluded in SQL.

    The exclusion lives in the query (`EYE_SQL`) rather than in a filter over the result,
    because the rule is "a document never reaches the eye list" and a rule that depends on
    a caller remembering to apply it is not a rule (F133). Every path in this project's
    eye lists comes out of here, so there is one place to check.
    """
    out: dict[int, str] = {}
    for part in batched(list(file_ids), 500):
        marks = ",".join("?" * len(part))
        sql = EYE_SQL.format(marks=marks, document=DOCUMENT_VERDICT)
        out.update({int(r["id"]): str(r["path"])
                    for r in conn.execute(sql, tuple(part))})
    return out


def eye_sample(conn: sqlite3.Connection, missed: Sequence[Frame], size: int,
               seed: int) -> dict[int, str]:
    """`size` of the missed frames to go through by eye — random, seeded, no documents.

    Random and not the first N: the index is ordered by the time the files were indexed, so
    a prefix of the misses would be one trip and the reason breakdown would describe that
    trip. The draw happens AFTER the document exclusion, so hiding papers cannot make the
    sample shorter than it was asked to be while there are frames left to put in it.
    """
    allowed = eye_candidates(conn, [f.file_id for f in missed])
    picked = sorted(allowed)
    random.Random(seed).shuffle(picked)
    return {file_id: allowed[file_id] for file_id in picked[:max(0, size)]}


def random_draw(ranked: Sequence[int], depth: int, rng: random.Random) -> list[int]:
    """`depth` frames drawn from the SAME population the ranking came out of.

    The baseline of the brief. Drawn from `ranked` itself and not from the collection:
    the ranking covers exactly the frames that have a vector of this model, and a baseline
    over a wider set would be answering an easier question than the one being compared.
    """
    return rng.sample(list(ranked), min(max(0, depth), len(ranked)))


def sweep(ranked: Sequence[int], verdicts: dict[int, str], missed: Sequence[int],
          depths: Sequence[int], seed: int = DEFAULT_SEED) -> list[Row]:
    """The table: one row per depth, over one ranked list and one set of misses.

    Every depth is a PREFIX of the same ranking, so the rows are comparable by
    construction — a deeper list can only find more, and the price column is what says
    whether the extra finds are worth having.
    """
    wanted = set(missed)
    rng = random.Random(seed)
    rows: list[Row] = []
    for depth in sorted({int(d) for d in depths}):
        prefix = list(ranked[:depth])
        drawn = random_draw(ranked, depth, rng)
        rows.append(Row(
            depth=depth,
            found=sum(1 for file_id in prefix if file_id in wanted),
            shown=sum(1 for file_id in prefix
                      if verdicts.get(file_id) not in (PRODUCT_VERDICT, DOCUMENT_VERDICT)),
            hidden=sum(1 for file_id in prefix
                       if verdicts.get(file_id) == DOCUMENT_VERDICT),
            random_found=sum(1 for file_id in drawn if file_id in wanted),
            misses=len(wanted)))
    return rows


def load_reasons(path: Path) -> dict[int, str]:
    """The filled-in worksheet -> {file_id: reason}. Cells still holding `null` are not
    answered yet and are dropped, so a half-filled sheet is usable."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return {int(file_id): str(value).strip()
            for file_id, value in data.items() if value is not None}


def count_reasons(reasons: dict[int, str]) -> Counter[str]:
    """How many frames per reason. Unknown codes keep their own name — see REASONS."""
    return Counter(reasons.values())


def write_reason_template(path: Path, file_ids: Sequence[int]) -> int:
    """A worksheet to fill in: `{file_id: null}` — file ids and nothing else.

    Whoever fills it in opens those frames (`--paths`, or the web app by id) and replaces
    each null with one of the reason codes. Returns how many frames were written.
    """
    sheet = {str(file_id): None for file_id in file_ids}
    path.write_text(json.dumps(sheet, ensure_ascii=False, indent=1), encoding="utf-8")
    return len(sheet)


def decide(rows: Sequence[Row]) -> tuple[str, str]:
    """(the outcome letter, one line saying why) — by the pre-registered criteria above.

    Read off the DEEPEST row: the prefixes grow, so that row is where the recall of the
    query is highest, and taking the best of several depths instead would be picking the
    number after seeing it.
    """
    if not rows:
        raise SystemExit("нечего решать: в таблице нет ни одной глубины")
    deep = max(rows, key=lambda r: r.depth)
    if not deep.misses:
        return "C", ("модель не пропустила ни одного размеченного товара — второе мнение "
                     "не о чем спрашивать")
    price = (f"{deep.price:.0f} кадров на находку" if deep.price is not None
             else "цены нет: ноль находок")
    lift = (f"{deep.lift:.1f}x" if deep.lift is not None
            else "случайный отбор не нашёл ничего")
    if deep.lift is not None and deep.lift < LIFT_MIN:
        return "C", (
            f"на глубине {deep.depth} запрос берёт {deep.recall:.1%} пропусков против "
            f"{deep.random_recall:.1%} у случайного отбора ({lift} при пороге "
            f"{LIFT_MIN:.1f}x): подъёма нет — запрос ошибается на тех же кадрах, что и "
            f"модель, и ответ надо искать в разборе причин, а не в глубине списка")
    if deep.recall < RECALL_MIN:
        return "B", (
            f"подъём есть ({lift} при пороге {LIFT_MIN:.1f}x), но на глубине {deep.depth} "
            f"запрос берёт {deep.recall:.1%} пропусков при пороге {RECALL_MIN:.0%}, цена "
            f"{price}: сигнал настоящий, список короткий — мерить глубже, единственный "
            f"подтверждённый рычаг полноты это глубина")
    return "A", (
        f"на глубине {deep.depth} запрос берёт {deep.recall:.1%} пропусков при пороге "
        f"{RECALL_MIN:.0%}, подъём над случайным отбором {lift} при пороге "
        f"{LIFT_MIN:.1f}x, цена {price}: второе мнение запросом работает — следующей "
        f"фичей список на пересмотр этой глубины")


def _pct(part: int, whole: int) -> str:
    return f"{100.0 * part / whole:.1f}%" if whole else "—"


def format_population(frames: Sequence[Frame]) -> str:
    """The head: the labelled sample, split the way the two losses split it.

    The three layers of the sample are not reconstructed here — the index says who was
    asked, and that is the fact the split is made on rather than the layer a frame was
    drawn from.
    """
    products = [f for f in frames if f.is_product]
    found = sum(1 for f in products if f.found)
    by_model = sum(1 for f in products if f.missed_by_model)
    by_gate = sum(1 for f in products if f.missed_by_gate)
    return "\n".join([
        "=" * 96,
        f"ТОВАРЫ, КОТОРЫЕ МОДЕЛЬ ВИДЕЛА И НЕ УЗНАЛА: {len(frames)} размеченных кадров, "
        f"из них товаров {len(products)}",
        "=" * 96,
        f"нашли (вердикт `{PRODUCT_VERDICT}` сегодня): {found} "
        f"({_pct(found, len(products))} размеченных товаров)",
        f"пропустила МОДЕЛЬ (спрошена, ответила иначе): {by_model} "
        f"({_pct(by_model, len(products))}) — эта фича",
        f"пропустил ГЕЙТ (модель не спрашивали): {by_gate} "
        f"({_pct(by_gate, len(products))}) — закрыто замером, здесь только для сравнения",
    ])


def format_depths(rows: Sequence[Row], population: int) -> str:
    """The table the brief asks for: recall, price and the baseline, per depth."""
    out = [
        f"ГЛУБИНА СПИСКА -> СКОЛЬКО ПРОПУСКОВ ЗАБРАЛИ (ранжируется {population} кадров)",
        f"  {'глубина':>8} {'найдено':>8} {'полнота':>8} {'показать':>9} "
        f"{'цена':>14} {'случайно':>9} {'подъём':>8}",
    ]
    for r in sorted(rows, key=lambda x: x.depth):
        price = f"{r.price:.0f} кадр/шт" if r.price is not None else "—"
        lift = f"{r.lift:.1f}x" if r.lift is not None else "—"
        out.append(f"  {r.depth:>8} {r.found:>8} {r.recall:>8.1%} {r.shown:>9} "
                   f"{price:>14} {r.random_found:>9} {lift:>8}")
    out.append("«показать» — кадры списка, которые человеку придётся смотреть: уже "
               "разложенные\n  товары и документы из него исключены; «случайно» — тот же "
               "объём, взятый наугад\n  из того же множества. Обе колонки обычно равны "
               "глубине: индекс векторов держит\n  только кадры с вердиктом `photo` "
               "(F120), а там ни разложенных товаров, ни\n  документов нет по построению.")
    return "\n".join(out)


def format_documents(rows: Sequence[Row], unreachable: int, misses: int) -> str:
    """What the privacy rule costs, in frames, and what it hides on the way.

    Printed whatever the outcome: a product that the index calls a document can never be
    put in front of a person by this path, so a recall computed as if it could would be a
    promise the feature is not allowed to keep.
    """
    hidden = max((r.hidden for r in rows), default=0)
    out = [f"ДОКУМЕНТЫ: из списка скрыто до {hidden} кадров с вердиктом "
           f"`{DOCUMENT_VERDICT}` — в лист для глаз они не попадают ни при каком запросе"]
    if unreachable:
        out.append(f"  из самих пропусков под `{DOCUMENT_VERDICT}` лежат {unreachable} "
                   f"из {misses} ({_pct(unreachable, misses)}) — этим путём их не "
                   f"забрать, и полнота выше {_pct(misses - unreachable, misses)} "
                   f"по нему недостижима")
    else:
        out.append(f"  ни один из {misses} пропусков не лежит под "
                   f"`{DOCUMENT_VERDICT}` — правило приватности здесь ничего не стоит "
                   f"в полноте")
    return "\n".join(out)


def format_reasons(counts: Counter[str], marked: int, sample: int, misses: int) -> str:
    """Hypothesis A: why the model called these frames photographs, by count.

    The share is over the MARKED frames and the sample size is printed next to it: a
    breakdown of sixty frames is a description of sixty frames, and the line above the
    table is what keeps it from being read as a description of all 288.
    """
    out = [f"РАЗБОР ПРИЧИН: размечено {marked} из {sample} кадров выборки "
           f"(вся выборка — {misses} пропусков)"]
    if not marked:
        out.append("  (ни одной причины не проставлено: в листе только null)")
        return "\n".join(out)
    for reason, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        note = REASONS.get(reason, "код вне словаря причин")
        out.append(f"  {reason:>16}: {count:>4} ({_pct(count, marked)}) — {note}")
    unknown = sorted(set(counts) - set(REASONS))
    if unknown:
        out.append(f"  ВНИМАНИЕ: коды вне словаря — {', '.join(unknown)}; "
                   f"считаются отдельными строками, а не отбрасываются")
    return "\n".join(out)


def format_outcome(rows: Sequence[Row]) -> str:
    letter, why = decide(rows)
    return f"ИСХОД {letter}: {why}"


def product_queries(cfg: Config, override: Sequence[str] | None) -> list[str]:
    """The phrases the query is asked with — `features.saved_slices['products']` (F151).

    The slice the product already ships is what a second opinion has to be measured with:
    a private list of phrases here would measure a query nobody can turn on. `--queries`
    overrides it for the one case that needs it — trying a wording before it is saved.
    """
    if override:
        return list(override)
    slices = tuple(getattr(getattr(cfg, "features", None), "saved_slices", ())
                   or DEFAULT_SAVED_SLICES)
    for saved in (*slices, *DEFAULT_SAVED_SLICES):
        if saved.name == PRODUCT_SLICE:
            return list(saved.queries)
    raise SystemExit(f"в `features.saved_slices` нет среза `{PRODUCT_SLICE}` — задайте "
                     f"формулировки явно: --queries \"a photo of a product\"")


def ranker(cfg: Config, conn: sqlite3.Connection,
           index: str) -> Callable[[Sequence[str], int], list[int]]:  # pragma: no cover
    """(phrases, depth) -> ranked file ids, over the index the caller named.

    One text pass and no image pass: the vectors are on disk already, which is the entire
    cost argument of this feature. Each index is asked with the query encoded by ITS OWN
    tower — the reason the two live apart at all — and the ranking functions are the
    engine's, so what is measured here is what `sorta search` would return.
    """
    s = naming_settings(cfg)
    if index == INDEX_CLASS:
        model = embedding_model(s)
        settings, rank_with = s, search.search_classification
    else:
        model = search_index_model(cfg)
        settings, rank_with = search_index_settings(s, model), search.search
    encoder = search.text_encoder(settings)

    def rank(queries: Sequence[str], depth: int) -> list[int]:
        vector = search.encode_queries(queries, encoder)
        return [file_id for file_id, _score in rank_with(conn, vector, model, depth)]

    return rank


def parse_depths(text: str) -> list[int]:
    """"200,400,800" -> [200, 400, 800]. Sorted, deduplicated, never empty.

    Depth is a parameter and not a literal anywhere in this script — it is the one lever
    the measurements confirmed, so a grid that could not be moved would make the report
    unable to answer the question it exists for.
    """
    values = sorted({int(part) for part in text.replace(",", " ").split()})
    values = [v for v in values if v > 0]
    if not values:
        raise SystemExit("--depths: пустая сетка глубин")
    return values


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--labels", required=True,
                    help="JSON {file_id: true|false} — the hand labels of the sample")
    ap.add_argument("--queries", nargs="+",
                    help=f"phrases to rank by (default: `features.saved_slices` "
                         f"`{PRODUCT_SLICE}`)")
    ap.add_argument("--index", choices=(INDEX_CLASS, INDEX_SEARCH), default=INDEX_CLASS,
                    help=f"which vectors to rank ({INDEX_CLASS} — clip_embeddings, "
                         f"computed for the whole collection; {INDEX_SEARCH} — "
                         f"search_embeddings, needs features.search_index)")
    ap.add_argument("--depths", default=",".join(str(d) for d in DEFAULT_DEPTHS),
                    help=f"the depth grid (default {DEFAULT_DEPTHS[0]}..."
                         f"{DEFAULT_DEPTHS[-1]}, doubling)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED,
                    help=f"the seed of the baseline and the eye sample "
                         f"(default {DEFAULT_SEED})")
    ap.add_argument("--eye-sample", type=int, default=DEFAULT_EYE_SAMPLE,
                    help=f"how many missed frames to go through by eye "
                         f"(default {DEFAULT_EYE_SAMPLE})")
    ap.add_argument("--write-reasons", help="write a worksheet of the eye sample (file "
                                            "ids only) to fill in, and exit")
    ap.add_argument("--reasons", help="JSON {file_id: reason} — enables the breakdown")
    ap.add_argument("--paths", action="store_true",
                    help="print the paths of the eye sample — needed to look at the "
                         "frames; documents are never in that list")
    args = ap.parse_args()

    depths = parse_depths(args.depths)
    cfg = load_config(args.config)
    labels = load_labels(Path(args.labels))
    conn = connect(cfg.database)
    try:
        frames = labelled_frames(conn, labels)
        if not frames:
            raise SystemExit("ни один размеченный кадр не найден в индексе: разметка не "
                             "от этой коллекции или файлы выпали как дубликаты")
        missed = [f for f in frames if f.missed_by_model]
        print(format_population(frames))
        if not missed:
            raise SystemExit("модель не пропустила ни одного размеченного товара — "
                             "мерить нечего")

        sample = eye_sample(conn, missed, args.eye_sample, args.seed)
        if args.write_reasons:
            written = write_reason_template(Path(args.write_reasons), list(sample))
            print(f"разбор причин: {written} кадров записано в {args.write_reasons} "
                  f"(только file_id; замените null на код причины: "
                  f"{', '.join(REASONS)})")
            if args.paths:
                for file_id, path in sample.items():
                    print(f"  {file_id:>8}  {path}")
            return 0

        queries = product_queries(cfg, args.queries)
        print(f"запрос: {', '.join(queries)} (индекс {args.index}, "
              f"глубины {', '.join(str(d) for d in depths)})")
        try:
            ranked = ranker(cfg, conn, args.index)(queries, WHOLE_RANKING)
        except search.EmbeddingsMissing as exc:
            raise SystemExit(
                f"ранжировать нечем ({exc.reason}): векторов модели {exc.model!r} в "
                f"индексе нет — запустите `sorta junk`") from None

        verdicts = all_verdicts(conn)
        rows = sweep(ranked, verdicts, [f.file_id for f in missed], depths, args.seed)
        unreachable = sum(1 for f in missed if f.verdict == DOCUMENT_VERDICT)

        print("-" * 96)
        print(format_depths(rows, len(ranked)))
        print("-" * 96)
        print(format_documents(rows, unreachable, len(missed)))
        print("-" * 96)
        if args.reasons:
            reasons = load_reasons(Path(args.reasons))
            print(format_reasons(count_reasons(reasons), len(reasons), len(sample),
                                 len(missed)))
        else:
            print(f"РАЗБОР ПРИЧИН: разметки нет — соберите лист для глаз "
                  f"(--write-reasons why.json --paths) и заполните его кодами: "
                  f"{', '.join(REASONS)}")
        print("=" * 96)
        print(format_outcome(rows))
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
