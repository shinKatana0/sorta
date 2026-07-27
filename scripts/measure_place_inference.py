"""Measure place inheritance (F85a) against GPS ground truth.

Inheritance is invisible by construction: a file that got its place from a neighbour has
no coordinates of its own to check the guess against. But the same collection holds tens
of thousands of files whose place IS known exactly from EXIF GPS. Hide the coordinates of
a random slice of those, run the real `geo.resolve_places` over the result and compare
what it inferred with what the file actually was — precision and recall on thousands of
examples, with no manual labelling and without opening a single image.

The session level and the trip level are reported separately: they answer different
questions (how well six hours predict a place vs. how well a whole trip does), and F85a
only ships if the trip level clears 95% precision.

Ground truth comes from the `places` table of the collection itself (the `exact_gps`
rows the last `sorta geo` wrote), so the measurement needs no network and no bundled
GeoNames data — the recorded answers are replayed instead of the provider. It follows
that the run being measured has to have happened: run `sorta geo` first.

Two caveats worth stating out loud:

* files WITH GPS are camera shots, while the files inheritance actually runs on skew
  towards screenshots, forwards and messenger copies. So this is an optimistic upper
  bound on the real precision, not an estimate of it;
* hiding coordinates thins out the donors as well. With `--hide 0.2` a fifth of the
  evidence disappears, which makes the measured recall a lower bound.

Output holds aggregates only — no paths, no file names, no coordinates.

Usage:
    python scripts/measure_place_inference.py [--config config.yaml] [--hide 0.2]
                                              [--seed 17] [--repeat 5]
"""
from __future__ import annotations

import argparse
import random
import sqlite3
import sys
import tempfile
from collections import Counter
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sorta import geo  # noqa: E402
from sorta.config import Config, load_config  # noqa: E402
from sorta.db import connect  # noqa: E402

_LEVELS = ("exact_gps", "session_inferred", "trip_inferred", "unknown")


class _RecordedResolver:
    """Replays the places the recorded run already resolved for these coordinates.

    Stands in for the whole provider layer (`geo._resolver_for`), so the measurement is
    of the INHERITANCE, not of Nominatim or of the bundled base: a coordinate gets back
    exactly the place the collection has on file for it. A coordinate with no recorded
    place (unresolved, or the null-island sentinel) comes back empty, which is what the
    real run did with it too.
    """

    def __init__(self, by_coord: dict[tuple[float, float], geo._Place]) -> None:
        self._by_coord = by_coord

    def resolve_places(self, coords, progress=None):
        del progress
        return [self._by_coord.get(c, geo._UNKNOWN_PLACE) for c in coords]


def _load(db_path: Path) -> list[sqlite3.Row]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute(
            """SELECT f.id, f.taken_at, f.taken_at_confidence, f.gps_lat, f.gps_lon,
                      p.country, p.country_name, p.city, p.city_geonameid,
                      p.district_geonameid, p.district_name, p.confidence
               FROM files f LEFT JOIN places p ON p.file_id = f.id
               WHERE f.dup_of IS NULL AND f.error IS NULL"""
        ).fetchall()
    finally:
        conn.close()


def _coords(row: sqlite3.Row) -> tuple[float, float] | None:
    lat, lon = geo._coord(row["gps_lat"]), geo._coord(row["gps_lon"])
    if lat is None or lon is None or geo._is_null_island(lat, lon):
        return None
    return (lat, lon)


def _place_of(row: sqlite3.Row) -> geo._Place:
    return geo._Place(
        country=row["country"], city_geonameid=row["city_geonameid"],
        district_geonameid=row["district_geonameid"], city=row["city"],
        district_name=row["district_name"], country_name=row["country_name"],
    )


def _build_db(rows: list[sqlite3.Row], hidden: set[int], path: Path) -> sqlite3.Connection:
    """A scratch index with the same ids, dates and (optionally hidden) coordinates.

    Paths are synthetic on purpose: nothing in the geo stage reads them, and a report
    that cannot hold a real path cannot leak one.
    """
    conn = connect(path)
    with conn:
        conn.executemany(
            """INSERT INTO files (id, path, size, mtime, ext, media_type, taken_at,
                   taken_at_source, taken_at_confidence, gps_lat, gps_lon, indexed_at)
               VALUES (?, ?, 0, 0, 'jpg', 'photo', ?, 'exif', ?, ?, ?, '2026-01-01')""",
            [(r["id"], f"/m/{r['id']}.jpg", r["taken_at"], r["taken_at_confidence"],
              None if r["id"] in hidden else r["gps_lat"],
              None if r["id"] in hidden else r["gps_lon"])
             for r in rows],
        )
    return conn


def _run(cfg: Config, rows: list[sqlite3.Row], truth: dict[tuple[float, float], geo._Place],
         hidden: set[int], tmp: Path, tag: str) -> dict[int, tuple[str, geo._Place]]:
    """One full `resolve_places` over a copy of the collection -> {id: (confidence, place)}."""
    db = tmp / f"{tag}.db"
    conn = _build_db(rows, hidden, db)
    try:
        with patch.object(geo, "_resolver_for", return_value=_RecordedResolver(truth)):
            geo.resolve_places(cfg, conn, progress=lambda done, total: None)
        return {
            r["file_id"]: (r["confidence"], _place_of(r))
            for r in conn.execute(
                """SELECT file_id, country, country_name, city, city_geonameid,
                          district_geonameid, district_name, confidence FROM places""")
        }
    finally:
        conn.close()


def _breakdown(result: dict[int, tuple[str, geo._Place]]) -> Counter:
    return Counter(conf for conf, _p in result.values())


def _print_levels(title: str, current: Counter, before: Counter | None) -> None:
    total = sum(current.values()) or 1
    print(f"\n{title} ({total} files)")
    print(f"  {'level':<18} {'files':>8} {'share':>8}   {'was':>8}")
    for level in _LEVELS:
        was = "" if before is None else f"{before.get(level, 0):>8}"
        print(f"  {level:<18} {current.get(level, 0):>8} "
              f"{current.get(level, 0) * 100 / total:>7.1f}% {was:>8}")
    if before is not None:
        delta = before.get("unknown", 0) - current.get("unknown", 0)
        print(f"  -> the trip level places {delta} more files "
              f"({delta * 100 / total:.1f}% of the collection)")


def _score(hidden: set[int], truth_place: dict[int, geo._Place],
           result: dict[int, tuple[str, geo._Place]], into: dict[str, Counter]) -> None:
    """Add one round of hidden files to the running counts of each level."""
    into["_all"]["hidden"] += len(hidden)
    for fid in hidden:
        got = result.get(fid)
        if got is None or got[0] not in into:
            continue
        c = into[got[0]]
        c["predicted"] += 1
        true = truth_place[fid]
        if true.country and got[1].country == true.country:
            c["country_ok"] += 1
        true_key = geo._city_key(true)
        if true_key is None:
            c["no_truth_city"] += 1  # country-level truth: cannot judge the city
        elif geo._city_key(got[1]) == true_key and got[1].country == true.country:
            c["correct"] += 1
        else:
            c["wrong"] += 1


def _print_scores(counts: dict[str, Counter]) -> float | None:
    """The precision table. Returns the trip level's precision — the acceptance number."""
    hidden = counts["_all"]["hidden"]
    print(f"\nhidden GPS files scored: {hidden}")
    print(f"  {'level':<18} {'predicted':>10} {'city ok':>8} {'city wrong':>11} "
          f"{'precision':>10} {'recall':>8} {'country ok':>11}")
    trip_precision = None
    for level in ("session_inferred", "trip_inferred"):
        c = counts[level]
        judged = c["correct"] + c["wrong"]
        value = c["correct"] * 100 / judged if judged else None
        if level == "trip_inferred":
            trip_precision = value
        precision = f"{value:.1f}%" if value is not None else "—"
        recall = f"{c['predicted'] * 100 / hidden:.1f}%" if hidden else "—"
        country = (f"{c['country_ok'] * 100 / c['predicted']:.1f}%"
                   if c["predicted"] else "—")
        print(f"  {level:<18} {c['predicted']:>10} {c['correct']:>8} {c['wrong']:>11} "
              f"{precision:>10} {recall:>8} {country:>11}")
        if c["no_truth_city"]:
            print(f"  {'':<18} ({c['no_truth_city']} of them have no city in the ground "
                  f"truth — country level only, not counted in precision)")
    return trip_precision


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--hide", type=float, default=0.2,
                    help="fraction of the GPS files whose coordinates are hidden")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--repeat", type=int, default=5,
                    help="rounds with different seeds, pooled into one table. The trip "
                         "level reaches only a few percent of the hidden files, so a "
                         "single round leaves ~100 cases and one mistake moves the "
                         "precision by a whole point")
    args = ap.parse_args()

    cfg = load_config(args.config)
    rows = _load(Path(cfg.database))
    if not rows:
        raise SystemExit(f"the index at {cfg.database} is empty — run `sorta index` first")

    truth: dict[tuple[float, float], geo._Place] = {}
    truth_place: dict[int, geo._Place] = {}
    gps_ids: list[int] = []
    for r in rows:
        coord = _coords(r)
        if coord is None:
            continue
        gps_ids.append(r["id"])
        if r["confidence"] == "exact_gps" and r["country"]:
            truth.setdefault(coord, _place_of(r))
            truth_place[r["id"]] = _place_of(r)
    if not truth:
        raise SystemExit(
            "no exact_gps places recorded — run `sorta geo` first: the ground truth of "
            "this measurement is what that run resolved")
    print(f"index: {len(rows)} canonical files, {len(gps_ids)} with usable GPS, "
          f"{len(truth_place)} of them with a resolved place (the ground truth)")
    print(f"trip thresholds (from the events section): "
          f"gap {cfg.events.trip_merge_gap_hours} h, {cfg.events.trip_merge_max_km} km; "
          f"session gap {cfg.geo.session_gap_hours} h")

    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        # 1) what the two levels do to the collection as it is. The "was" column is the
        #    same run with trip merging switched off, i.e. every trip is a single
        #    session — the state before F85a.
        no_trips = replace(cfg, events=replace(cfg.events, trip_merge_gap_hours=0))
        before = _breakdown(_run(no_trips, rows, truth, set(), tmp, "before"))
        after = _breakdown(_run(cfg, rows, truth, set(), tmp, "after"))
        _print_levels("places on the real collection", after, before)

        # 2) the same pipeline with a slice of the ground truth hidden, `repeat` times
        counts = {"_all": Counter(), "session_inferred": Counter(),
                  "trip_inferred": Counter()}
        ids = sorted(truth_place)
        for round_no in range(args.repeat):
            random.seed(args.seed + round_no)
            hidden = set(random.sample(ids, int(len(ids) * args.hide)))
            if not hidden:
                raise SystemExit("--hide is too small: no files to measure on")
            _score(hidden, truth_place,
                   _run(cfg, rows, truth, hidden, tmp, f"hidden{round_no}"), counts)
        precision = _print_scores(counts)

    print("\nprecision = correct city among the inferences the level made;\n"
          "recall    = share of the hidden files the level reached at all.\n"
          "F85a ships only if trip_inferred precision stays above 95%.")
    if precision is not None:
        print(f"\ntrip_inferred precision: {precision:.1f}% — "
              f"{'ACCEPTED' if precision > 95 else 'BELOW THE BAR, do not ship'}")


if __name__ == "__main__":
    main()
