"""Measure landmark-CLIP precision against GPS ground truth.

The landmark stage only ever runs on files WITHOUT a resolved place, so its errors
are invisible by construction — there is nothing to compare a guess against. But the
same collection also holds files whose country is known exactly from EXIF GPS. Running
the identical classifier over those gives a precision curve for free, with no manual
labelling.

Two populations are reported separately, because they answer different questions:

* files whose true country HAS an entry in the landmark list — a fire may legitimately
  be right, so this measures ordinary precision;
* files whose true country has NO entry at all (Thailand, Indonesia, the Maldives in
  the validation collection) — here every single fire is a false positive by
  construction, no judgement call involved. This is the cleanest signal available.

Caveat worth stating out loud: GPS-bearing files are camera shots, while the files the
stage actually runs on skew towards screenshots, downloads and forwards — the very
material that produced "video game -> New York". So the precision measured here is an
optimistic UPPER BOUND on the real thing, not an estimate of it.

F131 phase 0 (`--probe`): may the VLM be asked to check a CLIP proposal at all?
------------------------------------------------------------------------------
The animals cascade worked because CLIP's failure there was PERCEPTUAL — it cannot tell
a cat from a drawing of a cat — and a VLM looks at the scene instead of at a caption, so
the two tools complement each other. The landmark failure is a different one: telling
the Charles Bridge from some other European bridge takes KNOWING what the Charles Bridge
looks like, and a 3-billion-parameter model may hold exactly the same weakness CLIP does
rather than a cure for it. If it does, the cascade would confirm wrong cities with an
air of authority — worse than no cascade at all, because "a false city is worse than no
city" (F75). Half a day of measuring answers that; a week of building answers it too,
much later.

So the probe asks the model the SAME question the cascade would ask, on frames whose
right answer is already known, and reports the number that decides the feature: the
share of WRONG proposals the model confirmed. Three groups, none of them hand-labelled:

* `confirmed` — rows the stage already resolved (`places.confidence='visual'`), i.e.
  proposals that survived F75 corroboration; the proposal is the right one;
* `rejected` — today's proposals that corroboration threw away. The stage stores no
  scores and keeps no rejected candidates (which is why phase 1 has to start by storing
  them), so the probe re-derives them by running the pipeline's own corroboration over a
  fresh CLIP pass;
* `foreign` — the strongest proposals on GPS-bearing frames whose country has no entry
  in the landmark list at all: every one of them is wrong by construction.

The criteria are PRE-REGISTERED below, in code, before the first run — a bar chosen
after seeing the table is not a bar. The same pass also prints the size of the
uncertainty band per threshold, which is the number phase 1 would need to pick a lower
`naming.landmark_threshold` from.

The run said go (2026-08-02, 104 frames: zero wrong proposals confirmed by either form,
92% accuracy on the naming one against 78% on the verification one), and phase 1 shipped
that naming question as `features.landmarks_verify`. Two things follow for this script.
The naming half is now IMPORTED from the stage rather than owned here, so a re-run prices
the question the pipeline really asks. And the re-run is owed: only 24 of the 64 wrong
proposals in the deciding pass were hard ones — a specific wrong city rather than a scene
from a country the list does not cover — which is a small sample to rest a feature on.
The hard negatives are collected by lowering the threshold in a TEMPORARY copy of the
config (`--probe-band-min` changes the printed table only, not the sample):

    python scripts/measure_landmarks.py --config <copy with landmark_threshold: 0.50> \
        --probe --probe-frames 40 --probe-pool 800

Privacy: the report is counts only. No path, no file id and no basename reaches the
output — the rule of measure_ocr_gate.py and measure_vlm_resolution.py before it.

Usage:
    python scripts/measure_landmarks.py [--config config.yaml] [--limit 2000]
                                        [--no-gps-sample 300]
    python scripts/measure_landmarks.py --probe          # F131 phase 0 (needs the VLM)
    python scripts/measure_landmarks.py --probe --probe-frames 30 --probe-band-min 0.6
"""
from __future__ import annotations

import argparse
import os
import random
import sqlite3
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sorta import i18n, imaging  # noqa: E402
from sorta.config import load_config  # noqa: E402
from sorta.geodata import GeoResolver  # noqa: E402
from sorta.landmarks import (  # noqa: E402
    LANDMARK_MAX_NEW_TOKENS,
    LANDMARK_NAMING_PROMPT,
    Landmark,
    LandmarkStats,
    _answer_words,
    _corroborate,
    _folder_hints,
    _Match,
    _parent_dir,
    batched,
    clip_classifier,
    landmark_phrase,
    landmark_prompts,
    load_landmarks,
    match_named_landmark,
    vlm_landmark_asker,
)
from sorta.naming import naming_settings  # noqa: E402

THRESHOLDS = (0.50, 0.70, 0.80, 0.85, 0.90, 0.95, 0.99)


def _fetch(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
    return conn.execute(sql, params).fetchall()


def _classify(classifier, paths: list[str], prompts: list[str], n_landmarks: int,
              batch_size: int) -> list[tuple[int, float]]:
    """-> [(best landmark index, its probability)] in the order of `paths`.

    argmax is taken over the landmark prompts only, exactly as detect_landmarks does —
    the negative prompts are there to drain probability mass, not to win.
    """
    out: list[tuple[int, float]] = []
    done = 0
    for chunk in batched(paths, batch_size):
        probs = classifier(list(chunk), prompts)
        for row in probs:
            best = int(np.argmax(row[:n_landmarks]))
            out.append((best, float(row[best])))
        done += len(chunk)
        print(f"  {done}/{len(paths)}", end="\r", flush=True)
    print(" " * 40, end="\r")
    return out


# --- F131 phase 0: the probe ------------------------------------------------------

# The three populations the probe asks about, and what each one knows.
GROUP_CONFIRMED = "confirmed"   # the proposal is right (it already survived F75)
GROUP_REJECTED = "rejected"     # the proposal is wrong (corroboration threw it away)
GROUP_FOREIGN = "foreign"       # the proposal is wrong (the country has no entry at all)
PROBE_GROUPS = (GROUP_CONFIRMED, GROUP_REJECTED, GROUP_FOREIGN)

# The two question forms of the brief. Verification is what a cascade would really run —
# CLIP proposes, the model checks — and naming is the control: a model that cannot name
# the place it is looking at cannot be checking it either, it is only agreeing.
#
# The run decided between them (naming, by a factor of two on the right proposals in all
# three passes), so the naming half is no longer the script's own: phase 1 took the
# prompt into the stage and this imports it back. A private copy would have let the two
# drift, and then this script would be measuring a question the pipeline does not ask —
# the same rule that makes the corroboration below an import rather than a reimplementation.
FORM_VERIFY = "verify"
FORM_NAMING = "naming"
VERIFY_PROMPT = ("Look at the photo. Was this photo taken at {place}? "
                 "Answer with exactly one word: yes or no.")
NAMING_PROMPT = LANDMARK_NAMING_PROMPT
# Long enough for "no, this is not ..." and short enough that a chatty model cannot turn
# the pass into a text generation benchmark.
PROBE_MAX_NEW_TOKENS = LANDMARK_MAX_NEW_TOKENS

# What the naming form did with one frame.
NAMED_PROPOSED = "proposed"   # it named the landmark CLIP proposed
NAMED_OTHER = "other"         # it named a different landmark from the list
NAMED_NONE = "none"           # it named nothing we know

# --- Pre-registered acceptance criteria (F131) --------------------------------
#
# 1. At least 50 frames with a known answer — the brief's minimum, and below it the
#    shares below are noise with a decimal point.
# 2. At most 10% of the WRONG proposals confirmed. The arithmetic behind the number:
#    lowering the threshold buys a band CLIP itself cannot split (0.980 against 0.991),
#    so the band is majority-wrong; at 30% right, a model that confirms 70% of the right
#    ones and 10% of the wrong ones leaves 0.21/(0.21+0.07) = 75% precision for
#    corroboration to work on, and at 20% wrong-confirmation that falls to 60% — a
#    cascade that mostly adds false cities. The band composition is printed by the same
#    run, so the reader can redo this arithmetic with the real numbers.
# 3. At least 70% of the RIGHT proposals confirmed. A gate that rejects a third of what
#    the stage finds today is not a gate, it is a loss: the 144 visual matches are the
#    thing being protected.
MIN_PROBE_FRAMES = 50
MAX_FALSE_CONFIRM = 0.10
MIN_TRUE_CONFIRM = 0.70

VERDICT_GO = "ИДТИ В ФАЗУ 1"
VERDICT_CLOSE = "ЗАКРЫТЬ ФИЧУ"
VERDICT_UNCLEAR = "ВЕРДИКТА НЕТ"

# The wording of a place, the word splitter and the reader of a free-form answer all
# live in the stage now (phase 1) and are imported above — `landmark_phrase`,
# `_answer_words`, `match_named_landmark`. Only the verification form is still the
# script's own: it lost the measurement, so the pipeline does not ask it and has no
# reason to carry it, while re-running the comparison is exactly what this script is for.
_words = _answer_words

AskVerifyFn = Callable[[str, Landmark], str]   # (path, proposed landmark) -> the answer
AskNamingFn = Callable[[str], str]             # (path) -> the answer


def parse_yes_no(answer: str) -> bool | None:
    """The verification answer -> True / False / None (the model did not say).

    Lenient in the one way that costs nothing and buys the answers a model actually
    gives: the keyword is looked for anywhere in the reply, because the model likes to
    explain itself. When both words are there ("no, yes it is a bridge but ...") the
    FIRST one wins — that is the one that answered the question.
    """
    text = _words(answer)
    yes, no = text.find("_yes_"), text.find("_no_")
    if yes < 0 and no < 0:
        return None
    return no < 0 or 0 <= yes < no


def question_askers(describe: Callable[[Sequence[Any], str, int], str],
                    max_edge: int) -> tuple[AskVerifyFn, AskNamingFn]:
    """The two question forms over an ALREADY LOADED runtime (naming.shared_vlm).

    The naming half IS the stage's asker (`vlm_landmark_asker`), not a copy of it: a
    re-measurement has to price the question the pipeline really asks, down to the decode.
    The verification half is built the same way here — the shared preview cache,
    Unicode/HEIC-safe — and a frame that will not decode gets an empty answer, which
    parses to "the model did not say" instead of to a confirmation.
    """
    def verify(path: str, landmark: Landmark) -> str:
        try:
            st = os.stat(path)
        except OSError:
            return ""
        image = imaging.decode_rgb_preview(path, st.st_mtime, st.st_size,
                                           max_edge=max_edge)
        if image is None:
            return ""
        return describe([image], VERIFY_PROMPT.format(place=landmark_phrase(landmark)),
                        PROBE_MAX_NEW_TOKENS)

    return verify, vlm_landmark_asker(describe, max_edge)


@dataclass(frozen=True)
class Candidate:
    """One frame to ask about. Carries the path, so it never reaches the report."""
    path: str
    group: str
    proposed: int    # index into the landmark list — what CLIP proposes for this frame
    correct: bool    # ...and whether that proposal is the right one


@dataclass(frozen=True)
class Answer:
    """What both forms said about one frame. Nothing here identifies it."""
    group: str
    correct: bool
    confirmed: bool | None   # the verification form; None — the answer did not parse
    named: str               # NAMED_PROPOSED | NAMED_OTHER | NAMED_NONE


def _confirmed(answer: Answer, form: str) -> bool:
    """Did `form` back the proposal? An answer that did not parse never counts as yes."""
    if form == FORM_VERIFY:
        return answer.confirmed is True
    return answer.named == NAMED_PROPOSED


@dataclass(frozen=True)
class FormStats:
    """One question form over the whole sample."""
    form: str
    right_total: int
    right_confirmed: int
    wrong_total: int
    wrong_confirmed: int
    unparsed: int          # answers the form could not read at all
    named_other: int       # frames where the naming form named a DIFFERENT landmark

    @property
    def total(self) -> int:
        return self.right_total + self.wrong_total

    @property
    def accuracy(self) -> float:
        """Confirming a right proposal and rejecting a wrong one, over everything."""
        if not self.total:
            return 0.0
        right = self.right_confirmed + (self.wrong_total - self.wrong_confirmed)
        return right / self.total

    @property
    def true_confirm(self) -> float:
        return self.right_confirmed / self.right_total if self.right_total else 0.0

    @property
    def false_confirm(self) -> float:
        """The number the feature is decided by: wrong proposals the model backed."""
        return self.wrong_confirmed / self.wrong_total if self.wrong_total else 0.0


def form_stats(answers: Sequence[Answer], form: str) -> FormStats:
    right = [a for a in answers if a.correct]
    wrong = [a for a in answers if not a.correct]
    unparsed = sum(1 for a in answers
                   if (a.confirmed is None if form == FORM_VERIFY
                       else a.named == NAMED_NONE))
    return FormStats(
        form=form,
        right_total=len(right),
        right_confirmed=sum(1 for a in right if _confirmed(a, form)),
        wrong_total=len(wrong),
        wrong_confirmed=sum(1 for a in wrong if _confirmed(a, form)),
        unparsed=unparsed,
        named_other=sum(1 for a in answers if a.named == NAMED_OTHER),
    )


def decide(stats: Sequence[FormStats], frames: int) -> tuple[str, str]:
    """The pre-registered verdict, from the criteria above and from nothing else."""
    if frames < MIN_PROBE_FRAMES:
        return VERDICT_UNCLEAR, (f"кадров с известным ответом {frames} < "
                                 f"{MIN_PROBE_FRAMES} — выборка не собрана")
    usable = [s for s in stats if s.right_total and s.wrong_total]
    if not usable:
        return VERDICT_UNCLEAR, "в выборке нет обеих сторон (верные и неверные предложения)"
    for s in usable:
        if s.false_confirm <= MAX_FALSE_CONFIRM and s.true_confirm >= MIN_TRUE_CONFIRM:
            return VERDICT_GO, (f"форма «{s.form}»: подтвердила {s.false_confirm:.0%} "
                                f"неверных (<= {MAX_FALSE_CONFIRM:.0%}) и "
                                f"{s.true_confirm:.0%} верных (>= {MIN_TRUE_CONFIRM:.0%})")
    worst = min(usable, key=lambda s: s.false_confirm)
    return VERDICT_CLOSE, (f"лучшая форма «{worst.form}»: подтвердила "
                           f"{worst.false_confirm:.0%} неверных предложений "
                           f"(порог {MAX_FALSE_CONFIRM:.0%}), верных — "
                           f"{worst.true_confirm:.0%} (порог {MIN_TRUE_CONFIRM:.0%})")


@dataclass(frozen=True)
class BandRow:
    """How many proposals a threshold makes, and how many corroboration lets through."""
    threshold: float
    proposals: int
    kept: int

    @property
    def dropped(self) -> int:
        return self.proposals - self.kept


def corroborated(rows: Sequence[Mapping[str, Any]], results: Sequence[tuple[int, float]],
                 landmarks: Sequence[Landmark], threshold: float, *,
                 min_group: int, dominance: float,
                 resolver: GeoResolver | None, lang: i18n.Lang,
                 ) -> tuple[list[tuple[Mapping[str, Any], int]],
                            list[tuple[Mapping[str, Any], int]]]:
    """Proposals at or above `threshold` -> the ones F75 keeps and the ones it drops.

    The corroboration itself is IMPORTED from the stage, not reimplemented here: a
    private copy would measure the script against itself, and the rejected set is the
    whole point of the `rejected` group (measure_vlm_resolution.py imports its shared
    reader for the same reason).
    """
    matches: list[_Match] = []
    index: list[tuple[Mapping[str, Any], int]] = []
    for row, (best, score) in zip(rows, results):
        if score < threshold:
            continue
        matches.append(_Match(file_id=int(row["id"]), folder=_parent_dir(row["path"]),
                              landmark=landmarks[best]))
        index.append((row, best))
    hints = _folder_hints(matches, resolver, lang)
    kept_ids = {m.file_id for m in
                _corroborate(matches, hints, min_group, dominance, LandmarkStats())}
    kept = [(row, best) for row, best in index if int(row["id"]) in kept_ids]
    dropped = [(row, best) for row, best in index if int(row["id"]) not in kept_ids]
    return kept, dropped


def band_curve(rows: Sequence[Mapping[str, Any]], results: Sequence[tuple[int, float]],
               landmarks: Sequence[Landmark], thresholds: Iterable[float], *,
               min_group: int, dominance: float,
               resolver: GeoResolver | None, lang: i18n.Lang) -> list[BandRow]:
    """The size of the uncertainty band per threshold — what phase 1 would pick from.

    The stage stores no landmark scores (there is no score column in `places`, and a
    rejected candidate is kept nowhere), so unlike the animals cascade this cannot be
    answered with a query — it has to be re-measured. Which is also why phase 1 has to
    start by storing the score.
    """
    band: list[BandRow] = []
    for threshold in thresholds:
        kept, dropped = corroborated(rows, results, landmarks, threshold,
                                     min_group=min_group, dominance=dominance,
                                     resolver=resolver, lang=lang)
        band.append(BandRow(threshold=threshold, proposals=len(kept) + len(dropped),
                            kept=len(kept)))
    return band


def confirmed_candidates(conn: sqlite3.Connection, landmarks: Sequence[Landmark],
                         limit: int, rng: random.Random) -> list[Candidate]:
    """Frames the stage already resolved visually — the proposal on them is the right one.

    Matched back to the list by (country, city): the row records where the file went,
    and that pair is what a landmark writes. A row whose city is no longer in the list
    is skipped rather than guessed at.
    """
    by_place = {(lm.country, lm.city): i for i, lm in enumerate(landmarks)}
    rows = _fetch(conn, """
        SELECT f.path, p.country, p.city FROM files f JOIN places p ON p.file_id = f.id
        WHERE p.confidence = 'visual' AND f.dup_of IS NULL AND f.error IS NULL
          AND f.media_type = 'photo'""")
    pool = [(r["path"], by_place[(r["country"], r["city"])]) for r in rows
            if (r["country"], r["city"]) in by_place]
    return [Candidate(path=path, group=GROUP_CONFIRMED, proposed=index, correct=True)
            for path, index in rng.sample(pool, min(limit, len(pool)))]


def rejected_candidates(dropped: Sequence[tuple[Mapping[str, Any], int]],
                        limit: int, rng: random.Random) -> list[Candidate]:
    """Today's proposals that F75 corroboration threw away — wrong by that verdict.

    The verdict is the trusted one on purpose: corroboration is the rule the user's own
    eye produced ("this folder is Prague, that Berlin is impossible"), and phase 1 keeps
    it as the last word over the model anyway.
    """
    return [Candidate(path=str(row["path"]), group=GROUP_REJECTED,
                      proposed=best, correct=False)
            for row, best in rng.sample(list(dropped), min(limit, len(dropped)))]


def foreign_candidates(rows: Sequence[Mapping[str, Any]],
                       results: Sequence[tuple[int, float]],
                       limit: int) -> list[Candidate]:
    """The strongest proposals on frames whose country the list cannot match at all.

    Sorted by score rather than sampled: the question worth asking is the one the
    cascade would really be asked — the proposals confident enough to reach a gate — and
    every one of them is a false positive by construction.
    """
    ranked = sorted(zip(rows, results), key=lambda pair: -pair[1][1])[:limit]
    return [Candidate(path=str(row["path"]), group=GROUP_FOREIGN,
                      proposed=best, correct=False)
            for row, (best, _score) in ranked]


def ask_all(candidates: Sequence[Candidate], verify: AskVerifyFn, naming: AskNamingFn,
            landmarks: Sequence[Landmark]) -> list[Answer]:
    """Both questions about every frame, in order; the paths stay in this function."""
    answers: list[Answer] = []
    for i, candidate in enumerate(candidates, 1):
        landmark = landmarks[candidate.proposed]
        confirmed = parse_yes_no(verify(candidate.path, landmark))
        named = match_named_landmark(naming(candidate.path), landmarks)
        answers.append(Answer(
            group=candidate.group, correct=candidate.correct, confirmed=confirmed,
            named=(NAMED_NONE if named is None
                   else NAMED_PROPOSED if named == candidate.proposed
                   else NAMED_OTHER),
        ))
        print(f"  {i}/{len(candidates)}", end="\r", flush=True)
    print(" " * 40, end="\r")
    return answers


def format_band(rows: Sequence[BandRow], current: float) -> str:
    lines = [f"\nПолоса неуверенности (кадры без места; сейчас порог {current:.2f}):",
             f"{'порог':>7} | {'предложений':>12} | {'оставит F75':>12} | {'отвергнет':>10}",
             "-" * 50]
    lines += [f"{r.threshold:>7.2f} | {r.proposals:>12} | {r.kept:>12} | {r.dropped:>10}"
              for r in rows]
    return "\n".join(lines)


def format_sample(counts: Mapping[str, int]) -> str:
    total = sum(counts.values())
    parts = ", ".join(f"{group} {counts.get(group, 0)}" for group in PROBE_GROUPS)
    warning = "" if total >= MIN_PROBE_FRAMES else f"  (< {MIN_PROBE_FRAMES}, мало!)"
    return f"\nВыборка: {total} кадров с известным ответом — {parts}{warning}"


def format_form(stats: FormStats) -> str:
    lines = [
        f"\nФорма «{stats.form}»: {stats.total} кадров, "
        f"точность {stats.accuracy:.0%}",
        f"  подтвердила ВЕРНЫХ предложений: {stats.right_confirmed} из "
        f"{stats.right_total} ({stats.true_confirm:.0%})",
        f"  подтвердила НЕВЕРНЫХ предложений: {stats.wrong_confirmed} из "
        f"{stats.wrong_total} ({stats.false_confirm:.0%})  <- решающее число",
        f"  не разобрано / ничего не названо: {stats.unparsed}",
    ]
    if stats.form == FORM_NAMING:
        lines.append(f"  назвала ДРУГОЕ место из списка: {stats.named_other}")
    return "\n".join(lines)


def format_groups(answers: Sequence[Answer]) -> str:
    """The same numbers per group — where a form fails matters as much as how often."""
    lines = ["\nПо группам (подтверждено формой «verify» / «naming» из всего):"]
    for group in PROBE_GROUPS:
        rows = [a for a in answers if a.group == group]
        if not rows:
            lines.append(f"  {group:<10} — нет кадров")
            continue
        verify = sum(1 for a in rows if _confirmed(a, FORM_VERIFY))
        naming = sum(1 for a in rows if _confirmed(a, FORM_NAMING))
        expected = "да" if rows[0].correct else "нет"
        lines.append(f"  {group:<10} {verify:>3} / {naming:>3} из {len(rows):>3}"
                     f"  (правильный ответ — «{expected}»)")
    return "\n".join(lines)


def format_verdict(stats: Sequence[FormStats], frames: int) -> str:
    verdict, why = decide(stats, frames)
    return (f"\nВЕРДИКТ ФАЗЫ 0: {verdict}\n  {why}\n"
            f"  (критерии зафиксированы до прогона: неверных <= {MAX_FALSE_CONFIRM:.0%}, "
            f"верных >= {MIN_TRUE_CONFIRM:.0%}, кадров >= {MIN_PROBE_FRAMES})")


def probe_report(answers: Sequence[Answer], counts: Mapping[str, int],
                 band: Sequence[BandRow], current: float) -> str:
    """The whole phase-0 report — counts only, nothing that identifies a frame."""
    stats = [form_stats(answers, FORM_VERIFY), form_stats(answers, FORM_NAMING)]
    return "\n".join([
        format_band(band, current),
        format_sample(counts),
        *(format_form(s) for s in stats),
        format_groups(answers),
        format_verdict(stats, len(answers)),
    ])


def run_probe(args: argparse.Namespace, cfg: Any, settings: Any,
              landmarks: Sequence[Landmark],
              conn: sqlite3.Connection) -> None:  # pragma: no cover — ML, needs a GPU
    """Phase 0 end to end: CLIP for the proposals, the VLM for the two questions."""
    from sorta.naming import shared_vlm

    if not landmarks:
        raise SystemExit("список ландмарок пуст — предлагать нечего и проверять нечего")
    rng = random.Random(args.seed)
    prompts = landmark_prompts(landmarks)
    covered = {lm.country for lm in landmarks}
    min_group = int(getattr(settings, "landmark_group_min", 5))
    dominance = float(getattr(settings, "landmark_group_dominance", 0.6))
    resolver = GeoResolver()
    lang = i18n.normalize_lang(cfg.language)
    classifier = clip_classifier(settings)

    unknown = _fetch(conn, """
        SELECT f.id, f.path FROM files f JOIN places p ON p.file_id = f.id
        WHERE p.confidence = 'unknown' AND f.dup_of IS NULL AND f.error IS NULL
          AND f.media_type = 'photo' ORDER BY f.id""")
    if not unknown:
        raise SystemExit("нет кадров без места — стадии ландмарок нечего предлагать")
    print(f"CLIP по {len(unknown)} кадрам без места (это же даёт полосу неуверенности)...")
    unknown_scores = _classify(classifier, [r["path"] for r in unknown], prompts,
                               len(landmarks), settings.clip_batch_size)
    band = band_curve(unknown, unknown_scores, landmarks,
                      [t for t in THRESHOLDS if t >= args.probe_band_min],
                      min_group=min_group, dominance=dominance,
                      resolver=resolver, lang=lang)
    _kept, dropped = corroborated(unknown, unknown_scores, landmarks,
                                  settings.landmark_threshold,
                                  min_group=min_group, dominance=dominance,
                                  resolver=resolver, lang=lang)

    foreign_pool = _fetch(conn, f"""
        SELECT f.id, f.path FROM files f JOIN places p ON p.file_id = f.id
        WHERE p.confidence = 'exact_gps' AND f.dup_of IS NULL AND f.error IS NULL
          AND f.media_type = 'photo' AND p.country IS NOT NULL
          AND p.country NOT IN ({','.join('?' * len(covered))})""", tuple(sorted(covered)))
    foreign_pool = rng.sample(foreign_pool, min(args.probe_pool, len(foreign_pool)))
    foreign_scores = []
    if foreign_pool:
        print(f"CLIP по {len(foreign_pool)} кадрам из непокрытых стран...")
        foreign_scores = _classify(classifier, [r["path"] for r in foreign_pool], prompts,
                                   len(landmarks), settings.clip_batch_size)

    candidates = (confirmed_candidates(conn, landmarks, args.probe_frames, rng)
                  + rejected_candidates(dropped, args.probe_frames, rng)
                  + foreign_candidates(foreign_pool, foreign_scores, args.probe_frames))
    counts = Counter(c.group for c in candidates)
    print(format_sample(counts))
    if not candidates:
        raise SystemExit("выборку собрать не удалось — нечего спрашивать у модели")

    print(f"\nдва вопроса о каждом из {len(candidates)} кадров "
          f"({cfg.vlm.model}, max_edge={cfg.vlm.max_edge})...")
    verify, naming = question_askers(shared_vlm(cfg.vlm.model), cfg.vlm.max_edge)
    answers = ask_all(candidates, verify, naming, landmarks)
    print(probe_report(answers, counts, band, settings.landmark_threshold))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--limit", type=int, default=2000,
                    help="how many GPS-ground-truth files to sample")
    ap.add_argument("--no-gps-sample", type=int, default=300,
                    help="how many place-less files to sample for the qualitative pass")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--probe", action="store_true",
                    help="F131 phase 0: ask the VLM about known frames instead of "
                         "measuring the CLIP threshold (needs the [vlm] extra)")
    ap.add_argument("--probe-frames", type=int, default=20,
                    help="frames per group (confirmed / rejected / foreign)")
    ap.add_argument("--probe-pool", type=int, default=400,
                    help="how many files from uncovered countries to classify to find "
                         "the strongest wrong proposals")
    ap.add_argument("--probe-band-min", type=float, default=0.50,
                    help="the lower edge of the uncertainty band to report")
    args = ap.parse_args()

    cfg = load_config(args.config)
    settings = naming_settings(cfg)
    landmarks = load_landmarks(settings.landmarks_file)
    prompts = landmark_prompts(landmarks)  # the same set the pipeline scores against
    covered = {lm.country for lm in landmarks}
    random.seed(args.seed)

    conn = sqlite3.connect(f"file:{cfg.database}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    if args.probe:  # phase 0 answers a different question and pays for a different pass
        run_probe(args, cfg, settings, landmarks, conn)
        conn.close()
        return

    truth = _fetch(conn, """
        SELECT f.path, p.country FROM files f JOIN places p ON p.file_id = f.id
        WHERE p.confidence = 'exact_gps' AND p.country IS NOT NULL
          AND f.dup_of IS NULL AND f.error IS NULL AND f.media_type = 'photo'""")
    if not truth:
        raise SystemExit(
            "no exact_gps rows with a country — run `sorta geo` first "
            "(before F65 this table was empty by definition)")

    by_country = Counter(r["country"] for r in truth)
    print(f"ground truth: {len(truth)} files with a known country")
    print("  " + ", ".join(f"{cc} {n}" for cc, n in by_country.most_common(10)))
    print(f"landmark list covers: {', '.join(sorted(covered))}")
    uncovered_n = sum(n for cc, n in by_country.items() if cc not in covered)
    print(f"  of those, {uncovered_n} files are in countries the list cannot match "
          f"— every fire on them is a false positive by construction\n")

    sample = random.sample(list(truth), min(args.limit, len(truth)))
    print(f"classifying {len(sample)} files (CLIP; first pass also builds previews)...")
    classifier = clip_classifier(settings)
    results = _classify(classifier, [r["path"] for r in sample], prompts,
                        len(landmarks), settings.clip_batch_size)

    print(f"\n{'порог':>7} | {'сработ.':>8} | {'верно':>6} | {'неверно':>8} | "
          f"{'точность':>9} | {'из них в непокрытых странах':>28}")
    print("-" * 92)
    for threshold in THRESHOLDS:
        fired = correct = wrong_uncovered = 0
        for row, (best, score) in zip(sample, results):
            if score < threshold:
                continue
            fired += 1
            if landmarks[best].country == row["country"]:
                correct += 1
            elif row["country"] not in covered:
                wrong_uncovered += 1
        wrong = fired - correct
        precision = f"{correct / fired * 100:.1f}%" if fired else "—"
        print(f"{threshold:>7.2f} | {fired:>8} | {correct:>6} | {wrong:>8} | "
              f"{precision:>9} | {wrong_uncovered:>28}")

    current = settings.landmark_threshold
    print(f"\n(в config.yaml сейчас landmark_threshold = {current})")

    print("\nЧТО ИМЕННО СРАБАТЫВАЕТ при текущем пороге (предсказание <- истина):")
    confusion: dict[str, Counter] = defaultdict(Counter)
    for row, (best, score) in zip(sample, results):
        if score >= current:
            confusion[landmarks[best].name][row["country"]] += 1
    if not confusion:
        print("  ничего не сработало")
    for name, counts in sorted(confusion.items(), key=lambda kv: -sum(kv[1].values())):
        detail = ", ".join(f"{cc}:{n}" for cc, n in counts.most_common())
        print(f"  {name:<24} {sum(counts.values()):>4}  ({detail})")

    if args.no_gps_sample:
        no_gps = _fetch(conn, """
            SELECT f.id, f.path FROM files f JOIN places p ON p.file_id = f.id
            WHERE p.confidence = 'unknown' AND f.dup_of IS NULL AND f.error IS NULL
              AND f.media_type = 'photo'""")
        if no_gps:
            pick = random.sample(list(no_gps), min(args.no_gps_sample, len(no_gps)))
            print(f"\nДля сравнения — {len(pick)} файлов БЕЗ места "
                  f"(это те, на которых стадия реально работает; истины нет):")
            res2 = _classify(classifier, [r["path"] for r in pick], prompts,
                             len(landmarks), settings.clip_batch_size)
            for threshold in (current, 0.95, 0.99):
                n = sum(1 for _b, s in res2 if s >= threshold)
                print(f"  порог {threshold:.2f}: сработало на {n} из {len(pick)} "
                      f"({n / len(pick) * 100:.1f}%)")
            hits = Counter(landmarks[b].name for b, s in res2 if s >= current)
            for name, n in hits.most_common():
                print(f"    {name}: {n}")
            # What the fires actually ARE. Only filenames and the junk verdict are
            # printed — never image content. If they cluster on screenshot/meme, the
            # cheap fix is a media_class gate rather than threshold tuning.
            print("\n  Что именно сработало (имя файла + вердикт junk):")
            for row, (best, score) in zip(pick, res2):
                if score < current:
                    continue
                verdict = conn.execute(
                    "SELECT verdict FROM media_class WHERE file_id = ?",
                    (row["id"],)).fetchone()
                print(f"    {score:.3f} {landmarks[best].name:<20} "
                      f"[{verdict['verdict'] if verdict else 'нет'}] "
                      f"{Path(row['path']).name}")
    conn.close()


if __name__ == "__main__":
    main()
