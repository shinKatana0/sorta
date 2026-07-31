"""F96 (discovery): can a VLM READ the place off the frame?

21.8% of the collection (~5 000 files) ends up with no place at all, and that is the
biggest product hole: the tool promises folders by city. The CLIP path is exhausted —
landmarks.py reaches 154 files out of 26 135 — and StreetCLIP was measured separately
(F85b) and rejected: 36% accuracy, no usable threshold, CC BY-NC weights.

The F96 hypothesis is a different one. StreetCLIP MATCHES a frame against a list of
places and therefore always guesses something. A VLM READS: a pharmacy sign, a menu in
Thai, a road plate, a number plate, a hotel name on a towel. That is text recognition
plus world knowledge, not landscape recognition, and Qwen2.5-VL can do it.

Plausible, but unmeasured — nobody knows how many of those 5 000 frames carry readable
geographic text at all. Hence this script: the measurement comes first, the feature (if
any) after it, as a separate brief.

Method — the hidden check of F85a/F85b: take the files whose place is known EXACTLY
from EXIF GPS, show the model the pixels and nothing else, compare its answer with the
truth at two levels, country and city. No labelling, no human looks at an image.

The sample is RANDOM over the whole GPS-truth population, not the first N rows: the
collection is grouped into trips, so the first 300 files would all be one city and the
headline number would be a fact about that city.

The prompt must ALLOW "unknown" and demand it when the frame carries no evidence. A
model that always answers is useless here: we are filling a field that is currently
empty, and a wrong city is WORSE than an empty one — the file silently moves into a
foreign folder and is never found again. Abstention is the main requirement of the
method, not a weakness of it.

Pre-registered criteria (written down BEFORE the run, see `verdict` — they live in code
so a disappointing table cannot be met by quietly lowering a bar). Counted over the
answers where the model did NOT abstain:

    city accuracy    >= 85%   below it we would confidently file every seventh file
                              into the wrong city
    country accuracy >= 95%   the country is what actually decides the layout here,
                              and an error in it is the most visible one
    answer rate      >= 15%   below it the game is not worth the candle: over an hour
                              of GPU for a few hundred files out of 5 000

    outcome A — all three cleared -> write the feature brief
    outcome B — country cleared, city not -> write the country only (it still feeds
                trip inheritance, F85a, and the UI hint)
    outcome C — almost no abstention, or accuracy below the bars -> close the item with
                the numbers, like StreetCLIP. THIS IS A NORMAL OUTCOME.

The TEXT line of the reply is DIAGNOSTIC and is not part of any criterion. It exists to
tell "the model said nothing because the frame has no text" from "the frame has text and
the model still could not place it" — but the model overreports it: on a blank
single-colour frame it answered yes. Read the number as an upper bound, not as a count
of frames with text.

Privacy. The report and the cache are aggregates: file ids, ISO country codes and
outcome flags. The recognized text is NEVER printed, never cached and never logged — it
is the content of somebody's personal frames (the rule of measure_ocr_gate.py and
measure_streetclip.py before this). Frames classified as `document` or `screenshot` are
excluded from the sample entirely: documents are not opened at all, and a screenshot of
a map would produce a flattering result that real photos would never reproduce.

Boundaries. The script writes NOTHING to the product database — it opens it read-only
and prints a report. The Qwen loader is duplicated here instead of imported from
junk.py: this is a probe, and duplication is cheaper than coupling (F95 is moving that
loader right now, and an import would break on the merge).

Usage (from the repo root, with the venv python):
    python scripts/measure_scene_text.py --config config.yaml
    python scripts/measure_scene_text.py --sample 300 --cache scene_text.json
    python scripts/measure_scene_text.py --cache scene_text.json   # replay, no model
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sqlite3
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sorta import i18n  # noqa: E402
from sorta.config import load_config  # noqa: E402
from sorta.geodata import GeoResolver  # noqa: E402

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

# The same input size the VLM tier of junk.py uses: this measures the cost the pipeline
# would actually pay, and a bigger frame buys detail the model does not read anyway.
MAX_EDGE = 896

# Three short lines fit in far less, but a model that starts explaining itself must be
# allowed to finish the line we parse rather than be cut off mid-word.
MAX_NEW_TOKENS = 48

# The prompt. It is a parameter of the script but NOT a knob to turn until the numbers
# look good — a prompt fitted to this sample measures the fitting, not the model.
# Two things are load-bearing: the ban on guessing from vegetation/architecture (that is
# exactly what StreetCLIP already does, and badly), and the explicit permission to say
# unknown, without which the answer rate is 100% and the measurement is meaningless.
PROMPT = (
    "Read the written text in this photo: shop signs, menus, street plates, number "
    "plates, posters, packaging, tickets.\n"
    "Say where the photo was taken ONLY if the written text (or an unmistakable "
    "landmark) tells you. Do NOT guess from vegetation, architecture, clothing, "
    "people, weather or the general look of the place.\n"
    "Answer in exactly three lines and nothing else, keeping the labels:\n"
    "TEXT: yes or no — is there any readable written text in the photo\n"
    "COUNTRY: the country in English, or unknown\n"
    "CITY: the city in English, or unknown\n"
    "Write unknown whenever the photo carries no written evidence of where it was "
    "taken. Answering unknown is the correct answer for most photos.\n"
    "Use exactly this shape, with the labels:\n"
    "TEXT: <yes or no>\n"
    "COUNTRY: <country in English, or unknown>\n"
    "CITY: <city in English, or unknown>"
)
# The shape is shown with placeholders instead of a worked example on purpose: an
# example naming a real country and city anchors a greedy decoder on that very place.
# The first run (2026-07-28, 300 frames) showed the reminder is not enough on its own —
# the model answered three bare lines ("Yes.\nUnknown.\nBangkok.") on 239 of them, so
# `parse_answer` reads that shape too. With greedy decoding the FORM of the answer is
# never guaranteed, and a probe that scores the form measures its own parser.

# The pre-registered bars, in code so they cannot be relaxed after seeing the table.
MIN_CITY_ACCURACY = 0.85
MIN_COUNTRY_ACCURACY = 0.95
MIN_ANSWER_RATE = 0.15

# What the feature would run on: the place-less files of the last full run.
CANDIDATE_FILES = 5092

# Diagnostic only, deliberately NOT part of any criterion: how often the model names a
# neighbouring town instead of the city the layout uses. Folders are named after the
# city, so a near miss is still a wrong folder — but the number tells apart "does not
# know the region" from "knows the region, disagrees about the municipality".
CITY_NEAR_KM = 25.0

CACHE_VERSION = 1

# path -> the raw model reply, or None if the frame did not decode. Replaced in tests;
# the raw reply never leaves `score`.
Reader = Callable[[str], str | None]

# A labelled line, with room for the decoration a chat model wraps it in: a bullet, a
# heading marker, "**COUNTRY**:", "TEXT - yes".
_LINE_RE = re.compile(r"^[\s*_`>#\-]*(TEXT|COUNTRY|CITY)[\s*_`]*[:\-]\s*(.*)$",
                      re.IGNORECASE)

# The head of the TEXT line, used to anchor the three positions of an unlabelled reply.
_YES_NO = ("yes", "no", "true", "false", "да", "нет")

# What a chat model wraps a one-word answer in, and how it ends a sentence it was not
# asked to write.
_DECORATION = " \t*_`\"'“”«»"
_TRAILING = ".!,;:"

# Everything a model says when it means "I cannot tell", plus the leftovers of a
# half-filled template. Matched on the whole cleaned value, never on a substring: a city
# called "None" would be a bug, "Nonesuch" must not become an abstention.
_UNKNOWN_VALUES = frozenset({
    "", "-", "--", "?", "n/a", "na", "none", "null", "unknown", "unclear", "uncertain",
    "unidentified", "unspecified", "not sure", "not known", "cannot tell", "can't tell",
    "no", "no idea", "no text", "неизвестно", "не знаю",
})

# Informal spellings the GeoNames base does not carry. Kept short and hand-checked: a
# long alias list would start deciding the accuracy instead of measuring it.
_COUNTRY_ALIASES: dict[str, str] = {
    # the full stops survive `_clean` only in the middle, hence both spellings
    "usa": "US", "u.s.a.": "US", "u.s.a": "US", "u.s.": "US", "u.s": "US", "america": "US",
    "the united states": "US", "united states of america": "US",
    "uk": "GB", "u.k.": "GB", "u.k": "GB", "england": "GB", "scotland": "GB",
    "wales": "GB",
    "great britain": "GB", "britain": "GB", "the united kingdom": "GB",
    "uae": "AE", "u.a.e.": "AE", "u.a.e": "AE", "the emirates": "AE", "emirates": "AE",
    "türkiye": "TR", "turkiye": "TR", "turkey": "TR",
    "holland": "NL", "the netherlands": "NL",
    "korea": "KR", "republic of korea": "KR", "south korea": "KR",
    "russian federation": "RU", "the russian federation": "RU",
    "czech republic": "CZ", "the czech republic": "CZ",
    "vietnam": "VN", "viet nam": "VN",
}


@dataclass(frozen=True)
class Parsed:
    """What the model said, reduced to three fields.

    `country`/`city` hold the model's own words for the duration of one match and are
    then dropped — nothing built from them is stored or printed (see the module
    docstring on privacy).
    """
    has_text: bool  # DIAGNOSTIC ONLY — see the module docstring, the model overreports it
    country: str
    city: str
    labelled: bool  # the reply followed the three-line format at all


@dataclass(frozen=True)
class Answer:
    """One measured frame, already reduced to outcomes. This is what the cache holds."""
    file_id: int
    true_cc: str
    pred_cc: str            # matched ISO cc; "" — abstained or the name is not in the base
    said_country: bool      # named a country (whether or not it resolved)
    said_city: bool
    city_hit: bool          # the named city IS the city the layout would use
    city_near: bool         # a different city within CITY_NEAR_KM of it (diagnostic)
    has_text: bool          # the model REPORTED readable text — diagnostic, see the
    #                         module docstring; it says yes on a blank frame
    labelled: bool          # the reply followed the requested format
    seconds: float

    @property
    def country_correct(self) -> bool:
        return bool(self.pred_cc) and self.pred_cc == self.true_cc

    @property
    def unresolved_country(self) -> bool:
        """Named a country that neither the alias list nor the geo base recognizes."""
        return self.said_country and not self.pred_cc


@dataclass(frozen=True)
class Level:
    """One decision level: how often the model spoke, and how often it was right."""
    name: str
    answered: int
    correct: int
    total: int
    min_accuracy: float

    @property
    def accuracy(self) -> float:
        return self.correct / self.answered if self.answered else 0.0

    @property
    def answer_rate(self) -> float:
        return self.answered / self.total if self.total else 0.0

    @property
    def passes(self) -> bool:
        """Both bars at once — accuracy is meaningless at an answer rate of 2%."""
        return self.accuracy >= self.min_accuracy and self.answer_rate >= MIN_ANSWER_RATE


@dataclass(frozen=True)
class CountryRow:
    """One row of the per-country table: is the model blind in a particular country?"""
    cc: str
    n: int
    answered: int
    correct: int


def truth_rows(db_path: str) -> list[sqlite3.Row]:
    """Files whose city is known exactly from EXIF GPS — the hidden ground truth.

    `document`/`screenshot` frames are excluded here rather than filtered later, so they
    never reach the model at all. The city must be known too: a row with a country and
    no city cannot score the city level, and keeping it in would make the two levels
    silently disagree about what the sample is.
    """
    return _select(db_path, """
        SELECT f.id, f.path, p.country, p.city, p.city_geonameid
        FROM files f
        JOIN places p ON p.file_id = f.id
        LEFT JOIN media_class m ON m.file_id = f.id
        WHERE p.confidence = 'exact_gps'
          AND p.country IS NOT NULL AND p.country != ''
          AND p.city IS NOT NULL AND p.city != ''
          AND (m.verdict IS NULL OR m.verdict NOT IN ('document', 'screenshot'))
          AND f.dup_of IS NULL AND f.error IS NULL AND f.media_type = 'photo'
        ORDER BY f.id""")


def _select(db_path: str, sql: str) -> list[sqlite3.Row]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute(sql).fetchall()
    finally:
        conn.close()


def random_sample(rows: Sequence[sqlite3.Row], size: int, seed: int) -> list[sqlite3.Row]:
    """`size` files drawn at random over the whole population, deterministic for a seed.

    Not stratified per country, unlike F85b: there the question was "does the model work
    in Thailand as well as in Russia", here it is "how much of THIS collection carries
    readable place text" — and that share is a property of the collection as it is, so
    the sample has to keep its country mix.
    """
    picked = list(rows)
    random.Random(seed).shuffle(picked)
    return picked[:size]


def existing(rows: Iterable[sqlite3.Row]) -> list[sqlite3.Row]:
    """Drop rows whose file is no longer on disk (an external drive, a move)."""
    return [r for r in rows if Path(r["path"]).exists()]


def _clean(value: str) -> str:
    """Strip the decoration a chat model puts around a one-word answer.

    Layered, in a loop: a value comes back as `"Denpasar".` or `**"Bangkok"**`, so one
    pass of each strip is not enough — the trailing full stop hides the quote behind it.
    """
    text = value.strip()
    while True:
        stripped = text.strip(_DECORATION).rstrip(_TRAILING).strip()
        if stripped == text:
            return stripped
        text = stripped


def _positional(lines: Sequence[str]) -> tuple[str, str, str] | None:
    """Three bare lines of an unlabelled reply -> (text, country, city); None if unsure.

    The first real run showed this is the shape the model actually produces most of the
    time: "Yes.\\nUnknown.\\nBangkok." — it reads the sign and names the city correctly,
    it just drops the labels it was asked for. Scoring that as an abstention scored the
    parser (239 replies out of 300) rather than the hypothesis.

    The three positions are anchored on the yes/no line when there is one, so a preamble
    before the answer or a "hope this helps" after it does not shift them; with no such
    line the first three are taken as they come. Fewer than three positions from the
    anchor -> None: which of two lines is the country and which the city would be a
    guess, and a guessed city moves a file into a foreign folder in silence.
    """
    start = 0
    for i, line in enumerate(lines):
        if line.casefold().startswith(_YES_NO):
            start = i
            break
    chunk = lines[start:start + 3]
    if len(chunk) < 3:
        return None
    return chunk[0], chunk[1], chunk[2]


def parse_answer(reply: str) -> Parsed:
    """The three-line reply -> (has text, country, city).

    Tolerant on purpose: a model that adds a preamble, swaps the line order or writes
    "COUNTRY - Thailand" is still answering, and treating that as an abstention would
    measure the parser.

    Labelled lines are the primary path and win whenever there is at least one of them —
    a partly labelled reply is read by labels alone, with no rule invented for the rest.
    Only a reply with NO label anywhere falls back to the positional shape (see
    `_positional`). Either way the value is cleaned the same, and "unknown" in any of its
    spellings stays an abstention: the costs are asymmetric, an invented city silently
    moves a file into a foreign folder while an empty field does nothing.

    A city given as "Bangkok, Thailand" keeps the first part, a country given the same
    way keeps the last: that is the order the two are written in everywhere.

    `labelled` stays a report line of its own, now purely diagnostic: it says how often
    the model ignored the format, not how often the parser gave up.
    """
    fields: dict[str, str] = {}
    for line in reply.splitlines():
        match = _LINE_RE.match(line)
        if match:
            key = match.group(1).upper()
            fields.setdefault(key, _clean(match.group(2)))
    labelled = bool(fields)

    if not labelled:
        bare = [text for text in (_clean(line) for line in reply.splitlines()) if text]
        positional = _positional(bare)
        if positional is not None:
            fields = dict(zip(("TEXT", "COUNTRY", "CITY"), positional))

    def value(key: str, first_part: bool) -> str:
        raw = fields.get(key, "")
        if raw.casefold() in _UNKNOWN_VALUES:
            return ""
        parts = [_clean(p) for p in raw.split(",") if _clean(p)]
        if not parts:
            return ""
        chosen = parts[0] if first_part else parts[-1]
        return "" if chosen.casefold() in _UNKNOWN_VALUES else chosen

    has_text = fields.get("TEXT", "").casefold().startswith(("yes", "true", "да"))
    return Parsed(has_text=has_text, country=value("COUNTRY", first_part=False),
                  city=value("CITY", first_part=True), labelled=labelled)


def match_country(name: str, resolver: GeoResolver) -> str:
    """A country as the model wrote it -> ISO cc; "" if nothing recognizes the name.

    An unrecognized name counts as a WRONG answer downstream, not as an abstention: the
    model did speak. The count of them is reported separately so a parser problem cannot
    be mistaken for a model problem.
    """
    text = _clean(name)
    if not text:
        return ""
    alias = _COUNTRY_ALIASES.get(text.casefold())
    if alias:
        return alias
    curated = i18n.country_cc_by_name(text)
    if curated:
        return curated
    for lang in ("en", "ru", "ja"):
        cc = resolver.country_cc_by_name(text, lang)  # type: ignore[arg-type]
        if cc:
            return cc.upper()
    return ""


def _km(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Great-circle distance in kilometres."""
    lat1, lon1, lat2, lon2 = (math.radians(v) for v in (a[0], a[1], b[0], b[1]))
    h = (math.sin((lat2 - lat1) / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2)
    return 2 * 6371.0088 * math.asin(math.sqrt(h))


def match_city(name: str, truth_city: str, truth_gid: int | None,
               resolver: GeoResolver, near_km: float = CITY_NEAR_KM) -> tuple[bool, bool]:
    """The model's city against the truth -> (hit, near miss).

    A hit is what the product needs: the same city the layout would name the folder
    after, either by name or by geonameid (the model writes "Moscow", the base may hold
    another spelling of the same id). A near miss is a DIFFERENT city within `near_km` —
    still a wrong folder, reported only to show what kind of wrong it is.
    """
    text = _clean(name)
    if not text:
        return False, False
    if truth_city and text.casefold() == truth_city.strip().casefold():
        return True, False
    gids: list[int] = []
    for lang in ("en", "ru", "ja"):
        gids.extend(resolver.city_ids_by_name(text, lang))  # type: ignore[arg-type]
    if truth_gid is not None and truth_gid in gids:
        return True, False
    if truth_gid is None or not gids:
        return False, False
    here = resolver.coords_of(truth_gid)
    if here is None:
        return False, False
    for gid in gids:
        there = resolver.coords_of(gid)
        if there is not None and _km(here, there) <= near_km:
            return False, True
    return False, False


def score(reader: Reader, rows: Sequence[sqlite3.Row], resolver: GeoResolver) -> list[Answer]:
    """Ask the model about every row -> the outcome records.

    Undecodable frames are dropped rather than counted as abstentions: a broken file is
    a fact about the collection, and letting it in would depress the answer rate the
    criterion is measured against. The raw reply lives inside this loop and nowhere
    else, and the progress line carries counters only — never a path.
    """
    answers: list[Answer] = []
    total = len(rows)
    for done, row in enumerate(rows, start=1):
        started = time.perf_counter()
        reply = reader(row["path"])
        seconds = time.perf_counter() - started
        print(f"  {done}/{total}", end="\r", flush=True)
        if reply is None:
            continue
        parsed = parse_answer(reply)
        pred_cc = match_country(parsed.country, resolver) if parsed.country else ""
        hit, near = match_city(parsed.city, str(row["city"] or ""), row["city_geonameid"],
                               resolver) if parsed.city else (False, False)
        answers.append(Answer(
            file_id=int(row["id"]), true_cc=str(row["country"] or ""), pred_cc=pred_cc,
            said_country=bool(parsed.country), said_city=bool(parsed.city),
            city_hit=hit, city_near=near, has_text=parsed.has_text,
            labelled=parsed.labelled, seconds=seconds))
    print(" " * 40, end="\r")
    return answers


def levels(answers: Sequence[Answer]) -> tuple[Level, Level]:
    """-> (country level, city level), each with its own pre-registered bar."""
    total = len(answers)
    country = Level("страна", sum(1 for a in answers if a.said_country),
                    sum(1 for a in answers if a.country_correct), total,
                    MIN_COUNTRY_ACCURACY)
    city = Level("город", sum(1 for a in answers if a.said_city),
                 sum(1 for a in answers if a.city_hit), total, MIN_CITY_ACCURACY)
    return country, city


def country_table(answers: Sequence[Answer]) -> list[CountryRow]:
    """Per true country: frames, answers, hits. Codes and counts only."""
    by_cc: dict[str, list[Answer]] = defaultdict(list)
    for a in answers:
        by_cc[a.true_cc].append(a)
    return [CountryRow(cc, len(group), sum(1 for a in group if a.said_country),
                       sum(1 for a in group if a.country_correct))
            for cc, group in sorted(by_cc.items())]


def _percentile(values: Sequence[float], q: float) -> float:
    """The q-th percentile by nearest rank — no interpolation, no numpy."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(q * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def timing(answers: Sequence[Answer]) -> tuple[float, float, float]:
    """-> (median ms per frame, p90 ms, total seconds).

    The median, not the mean: the first frame carries the CUDA warm-up and a single
    40 MB HEIC decode would move a mean enough to change the forecast.
    """
    seconds = [a.seconds for a in answers]
    if not seconds:
        return 0.0, 0.0, 0.0
    return (1000.0 * statistics.median(seconds), 1000.0 * _percentile(seconds, 0.9),
            sum(seconds))


def forecast_minutes(median_ms: float, files: int = CANDIDATE_FILES) -> float:
    """What a full pass over the place-less files would cost, from the median frame."""
    return median_ms * files / 1000.0 / 60.0


def verdict(country: Level, city: Level) -> tuple[str, str]:
    """The pre-registered criteria applied -> (outcome letter, one line).

    Outcome C is a normal result: closing the item with numbers is worth more than a
    feature that files every seventh photo into a foreign city.
    """
    bars = (f"критерии: страна >= {MIN_COUNTRY_ACCURACY * 100:.0f}%, город >= "
            f"{MIN_CITY_ACCURACY * 100:.0f}%, невоздержаний >= {MIN_ANSWER_RATE * 100:.0f}%")
    got = (f"получено: страна {country.accuracy * 100:.1f}% при "
           f"{country.answer_rate * 100:.1f}%, город {city.accuracy * 100:.1f}% при "
           f"{city.answer_rate * 100:.1f}%")
    if country.passes and city.passes:
        return "A", f"ИСХОД A: оформляем фичу отдельным брифом — {got} ({bars})"
    if country.passes:
        return "B", (f"ИСХОД B: пишем только страну (наследование по поездке F85a + "
                     f"подсказка в UI) — {got} ({bars})")
    return "C", (f"ИСХОД C: закрываем пункт замером и записываем в бэклог с цифрами — "
                 f"{got} ({bars})")


def _pct(part: int, whole: int) -> str:
    return f"{100.0 * part / whole:.1f}%" if whole else "—"


def format_country_table(rows: Sequence[CountryRow]) -> str:
    """The per-country block. Country codes and counts only — no file is identifiable."""
    out = ["=" * 88,
           f"ПО СТРАНАМ (истина из GPS), стран(ы) в выборке: {len(rows)}",
           f"{'страна':>7} {'кадров':>8} {'ответов':>9} {'верно':>7} {'точность':>10}"]
    for r in sorted(rows, key=lambda r: -r.n):
        out.append(f"{r.cc:>7} {r.n:>8} {r.answered:>9} {r.correct:>7} "
                   f"{_pct(r.correct, r.answered):>10}")
    out.append("=" * 88)
    return "\n".join(out)


def format_report(answers: Sequence[Answer], meta: dict) -> str:
    """The block to paste into the backlog: sample, abstention, accuracy, cost."""
    total = len(answers)
    country, city = levels(answers)
    median_ms, p90_ms, total_s = timing(answers)
    with_text = sum(1 for a in answers if a.has_text)
    silent = sum(1 for a in answers if not a.said_country and not a.said_city)
    silent_with_text = sum(1 for a in answers
                           if a.has_text and not a.said_country and not a.said_city)
    out = [
        "=" * 88,
        "F96 — ЗАМЕР: МЕСТО ПО ТЕКСТУ В КАДРЕ (VLM, GPS-истина скрыта)",
        f"выборка: {total} кадров, случайная, seed {meta.get('seed')}; "
        f"document/screenshot исключены",
        f"модель: {meta.get('model')}, {meta.get('device')}, вход {meta.get('max_edge')}px, "
        f"загрузка {meta.get('load_seconds')} с, пик VRAM {meta.get('vram_peak_gb')} ГБ",
        "-" * 88,
        f"текст в кадре (по мнению модели): {with_text} ({_pct(with_text, total)})",
        f"воздержалась полностью: {silent} ({_pct(silent, total)}), "
        f"из них с текстом в кадре: {silent_with_text}",
        f"ответов не по формату: {sum(1 for a in answers if not a.labelled)}",
        f"названо стран, которых нет в базе: {sum(1 for a in answers if a.unresolved_country)}",
        "-" * 88,
        f"{'уровень':>8} {'ответов':>9} {'невоздержаний':>14} {'верно':>7} {'точность':>10} "
        f"{'порог':>8} {'взят':>6}",
    ]
    for lvl in (country, city):
        out.append(
            f"{lvl.name:>8} {lvl.answered:>9} {lvl.answer_rate * 100:>13.1f}% "
            f"{lvl.correct:>7} {lvl.accuracy * 100:>9.1f}% "
            f"{lvl.min_accuracy * 100:>7.0f}% {('да' if lvl.passes else 'нет'):>6}")
    near = sum(1 for a in answers if a.city_near)
    out += [
        f"город мимо, но в пределах {CITY_NEAR_KM:.0f} км (диагностика, в критерий не "
        f"входит): {near}",
        "-" * 88,
        f"время: медиана {median_ms:.0f} мс/кадр, p90 {p90_ms:.0f} мс, "
        f"всего {total_s / 60:.1f} мин на {total} кадров",
        f"прогноз на {CANDIDATE_FILES} файлов: {forecast_minutes(median_ms):.0f} мин "
        f"(по медиане)",
        "=" * 88,
    ]
    return "\n".join(out)


def save_cache(path: Path, answers: Sequence[Answer], meta: dict) -> None:
    """Per-file outcomes for a replay: ids, country codes and flags — never text."""
    rows = [[a.file_id, a.true_cc, a.pred_cc, int(a.said_country), int(a.said_city),
             int(a.city_hit), int(a.city_near), int(a.has_text), int(a.labelled),
             round(a.seconds, 4)] for a in answers]
    path.write_text(json.dumps({"version": CACHE_VERSION, "meta": meta, "answers": rows},
                               ensure_ascii=False), encoding="utf-8")


def load_cache(path: Path) -> tuple[list[Answer], dict]:
    """-> (answers, meta). A foreign version is an error, not a silently wrong table."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("version") != CACHE_VERSION:
        raise SystemExit(f"{path}: кэш версии {data.get('version')}, ожидается "
                         f"{CACHE_VERSION} — перемерить с --refresh")
    answers = [Answer(int(fid), str(true), str(pred), bool(sc), bool(sci), bool(hit),
                      bool(near), bool(text), bool(lab), float(sec))
               for fid, true, pred, sc, sci, hit, near, text, lab, sec in data["answers"]]
    return answers, data.get("meta", {})


def _cuda_peak_gb() -> float | None:  # pragma: no cover — hardware
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_reserved() / 2 ** 30


def qwen_reader(model_id: str = MODEL_ID, max_edge: int = MAX_EDGE,
                offline: bool = True) -> tuple[Reader, dict]:  # pragma: no cover — ML
    """The real Qwen2.5-VL reader -> (reader, meta about the load).

    Deliberately a copy of the loader in junk.py rather than an import of it: this is a
    probe that must not couple itself to a module another feature is rewriting. Loaded
    ONCE per run (peak VRAM ~20 GB), frames come from the pipeline's own preview cache
    so the measured cost is the cost the pipeline would pay.

    Greedy decoding, not sampling — for the reason written down in junk.py: Qwen's
    default generation_config has do_sample=True, and on some frames fp16 logits go to
    NaN, after which torch.multinomial fires a CUDA device-side assert that poisons the
    context for every later frame.
    """
    if offline:
        # The weights are already in the HuggingFace cache; the probe must not reach the
        # network, and a silent re-download would land in the per-frame timings.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import pillow_heif
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    from sorta import imaging

    pillow_heif.register_heif_opener()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    started = time.perf_counter()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=dtype, device_map=device)
    processor = AutoProcessor.from_pretrained(model_id)
    model.eval()
    load_seconds = time.perf_counter() - started

    def read(path: str) -> str | None:
        try:
            st = os.stat(path)
        except OSError:
            return None
        img = imaging.decode_rgb_preview(path, st.st_mtime, st.st_size, max_edge=max_edge)
        if img is None:
            return None
        messages = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": PROMPT},
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True)
        inputs = processor(text=[text], images=[img], return_tensors="pt").to(device)
        with torch.no_grad():
            out_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                     do_sample=False)
        gen_ids = out_ids[:, inputs["input_ids"].shape[1]:]
        return str(processor.batch_decode(gen_ids, skip_special_tokens=True)[0])

    return read, {"device": device, "model": model_id, "max_edge": max_edge,
                  "load_seconds": round(load_seconds, 1),
                  "max_new_tokens": MAX_NEW_TOKENS}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="F96: measure whether a VLM can read the place off the frame")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--db", help="database to read (default: the one in the config)")
    ap.add_argument("--sample", type=int, default=300,
                    help="how many GPS-truth frames to ask about (default 300)")
    ap.add_argument("--seed", type=int, default=20260728)
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--max-edge", type=int, default=MAX_EDGE)
    ap.add_argument("--allow-download", action="store_true",
                    help="do not force HF_HUB_OFFLINE (the weights should be cached)")
    ap.add_argument("--cache", help="JSON with the per-file outcomes: written after a "
                                    "measurement, replayed instead of one")
    ap.add_argument("--refresh", action="store_true", help="measure again anyway")
    args = ap.parse_args()

    cache = Path(args.cache) if args.cache else None
    if cache and cache.exists() and not args.refresh:
        answers, meta = load_cache(cache)
        print(f"кэш: {len(answers)} кадров из {cache}")
    else:
        cfg = load_config(args.config)
        db = args.db or str(cfg.database)
        rows = random_sample(existing(truth_rows(db)), args.sample, args.seed)
        if not rows:
            raise SystemExit("нет файлов с confidence='exact_gps' и городом — "
                             "сначала `sorta geo`")
        by_cc = Counter(r["country"] for r in rows)
        print(f"выборка: {len(rows)} кадров с точной GPS-истиной, {len(by_cc)} стран(ы)")
        print("  " + ", ".join(f"{cc}:{n}" for cc, n in by_cc.most_common()))

        resolver = GeoResolver()
        reader, meta = qwen_reader(args.model, args.max_edge,
                                   offline=not args.allow_download)
        meta["seed"] = args.seed
        print(f"модель загружена за {meta['load_seconds']} с на {meta['device']}")

        answers = score(reader, rows, resolver)
        peak = _cuda_peak_gb()
        meta["vram_peak_gb"] = round(peak, 2) if peak is not None else None
        print(f"измерено {len(answers)} кадров из {len(rows)}")
        if cache:
            save_cache(cache, answers, meta)
            print(f"кэш записан: {cache} (только file_id, коды стран и флаги)")

    if not answers:
        raise SystemExit("ни один кадр не декодировался — считать нечего")
    print()
    print(format_report(answers, meta))
    print(format_country_table(country_table(answers)))
    print()
    print(verdict(*levels(answers))[1])


if __name__ == "__main__":
    main()
