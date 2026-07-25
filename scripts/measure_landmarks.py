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

Usage:
    python scripts/measure_landmarks.py [--config config.yaml] [--limit 2000]
                                        [--no-gps-sample 300]
"""
from __future__ import annotations

import argparse
import random
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sorta.config import load_config  # noqa: E402
from sorta.landmarks import _NEGATIVE_PROMPTS, batched, clip_classifier, load_landmarks  # noqa: E402
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--limit", type=int, default=2000,
                    help="how many GPS-ground-truth files to sample")
    ap.add_argument("--no-gps-sample", type=int, default=300,
                    help="how many place-less files to sample for the qualitative pass")
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    cfg = load_config(args.config)
    settings = naming_settings(cfg)
    landmarks = load_landmarks(settings.landmarks_file)
    prompts = [lm.prompt for lm in landmarks] + list(_NEGATIVE_PROMPTS)
    covered = {lm.country for lm in landmarks}
    random.seed(args.seed)

    conn = sqlite3.connect(f"file:{cfg.database}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

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
