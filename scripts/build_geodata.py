"""F26/G1 data-prep: assemble bundled geo data from GeoNames (needs NETWORK, one-off).

Output in data/geo/:
  places.tsv    — geonameid, lat, lon, fcode, cc, admin1, admin2, name_en, population
                  (all cities1000 places — city PPLA*/PPLC and district are derived from them)
  names.tsv     — geonameid, lang, name  (ru/en/ja, from alternateNamesV2; includes
                  geonameids of cities, admin1 regions AND countries)
  admin1.tsv    — cc, admin1, geonameid, name_en  (admin1CodesASCII; trip name
                  by region, G-#19: the "Bali"/"Phuket" province)
  countries.tsv — cc, geonameid, name_en  (countryInfo; country trip-name fallback)

Prints sizes and language coverage (incl. ja by DISTRICTS — the open question F26).
The sorta runtime does NOT download this — it reads the ready bundled files (offline).

Run:  uv run python scripts/build_geodata.py
"""
from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

GEO = "https://download.geonames.org/export/dump/"
LANGS = {"ru", "en", "ja"}
CITY_FCODES = {"PPLC", "PPLA", "PPLA2", "PPLA3", "PPLA4"}
OUT = Path("data/geo")


def download(name: str) -> bytes:
    print(f"  downloading {name} …", flush=True)
    return urllib.request.urlopen(GEO + name, timeout=300).read()


def build_places() -> tuple[set[str], int]:
    """cities1000 → places.tsv; returns (set of geonameid, count of PPLA*/PPLC cities)."""
    data = download("cities1000.zip")
    ids: set[str] = set()
    ncity = 0
    OUT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as z, \
            (OUT / "places.tsv").open("w", encoding="utf-8", newline="\n") as out:
        with z.open("cities1000.txt") as fh:
            for raw in io.TextIOWrapper(fh, encoding="utf-8"):
                p = raw.rstrip("\n").split("\t")
                if len(p) < 19:
                    continue
                gid, name, ascii_, _alt, lat, lon = p[0], p[1], p[2], p[3], p[4], p[5]
                fcode, cc = p[7], p[8]
                admin1, admin2 = p[10], p[11]
                pop = p[14]
                ids.add(gid)
                if fcode in CITY_FCODES:
                    ncity += 1
                out.write(f"{gid}\t{lat}\t{lon}\t{fcode}\t{cc}\t{admin1}\t{admin2}\t"
                          f"{ascii_ or name}\t{pop}\n")
    return ids, ncity


def build_admin1() -> dict[str, tuple[str, str]]:
    """admin1CodesASCII.txt → admin1.tsv (cc, admin1, geonameid, name_en).

    The admin1CodesASCII key is "CC.admin1" (e.g. "ID.02"), matching cc+admin1 in
    places.tsv. Returns {geonameid: (cc.admin1_key, name_en)} — for adding
    region geonameids to the alternateNames capture and to the stats.
    """
    raw = download("admin1CodesASCII.txt").decode("utf-8")
    out_rows: list[tuple[str, str, str, str]] = []  # cc, admin1, gid, name_en
    regions: dict[str, tuple[str, str]] = {}
    for line in raw.splitlines():
        p = line.split("\t")
        if len(p) < 4:
            continue
        key, name_en, gid = p[0], p[1], p[3]  # p[2] — asciiname (== name_en here)
        if "." not in key or not gid:
            continue
        cc, admin1 = key.split(".", 1)
        if not cc or not admin1:
            continue
        out_rows.append((cc, admin1, gid, name_en))
        regions[gid] = (key, name_en)
    with (OUT / "admin1.tsv").open("w", encoding="utf-8", newline="\n") as out:
        for cc, admin1, gid, name_en in out_rows:
            out.write(f"{cc}\t{admin1}\t{gid}\t{name_en}\n")
    return regions


def build_countries() -> dict[str, tuple[str, str]]:
    """countryInfo.txt → countries.tsv (cc, geonameid, name_en).

    countryInfo — a TSV with '#' comments; cc=col0, Country=col4, geonameid=col16.
    Returns {geonameid: (cc, name_en)} for the alternateNames capture.
    """
    raw = download("countryInfo.txt").decode("utf-8")
    out_rows: list[tuple[str, str, str]] = []  # cc, gid, name_en
    countries: dict[str, tuple[str, str]] = {}
    for line in raw.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        p = line.split("\t")
        if len(p) < 17:
            continue
        cc, name_en, gid = p[0], p[4], p[16]
        if not cc or not gid:
            continue
        out_rows.append((cc, gid, name_en))
        countries[gid] = (cc, name_en)
    with (OUT / "countries.tsv").open("w", encoding="utf-8", newline="\n") as out:
        for cc, gid, name_en in out_rows:
            out.write(f"{cc}\t{gid}\t{name_en}\n")
    return countries


def build_names(ids: set[str]) -> dict[str, dict[str, str]]:
    """alternateNamesV2 (streamed) → names.tsv for our geonameids × ru/en/ja.

    `ids` includes cities1000 cities + admin1 regions + countries (G-#19), so
    names.tsv gets localized region/country names for trip names.
    Returns {geonameid: {lang: name}} for coverage statistics.
    """
    data = download("alternateNamesV2.zip")
    names: dict[str, dict[str, str]] = {}
    preferred: set[tuple[str, str]] = set()
    with zipfile.ZipFile(io.BytesIO(data)) as z, z.open("alternateNamesV2.txt") as fh:
        for raw in io.TextIOWrapper(fh, encoding="utf-8"):
            p = raw.rstrip("\n").split("\t")
            if len(p) < 4:
                continue
            gid, lang, name = p[1], p[2], p[3]
            if lang not in LANGS or gid not in ids:
                continue
            is_pref = len(p) > 4 and p[4] == "1"
            key = (gid, lang)
            # a preferred name wins; otherwise the first one seen
            if key in preferred and not is_pref:
                continue
            names.setdefault(gid, {})[lang] = name
            if is_pref:
                preferred.add(key)
    with (OUT / "names.tsv").open("w", encoding="utf-8", newline="\n") as out:
        for gid, per in names.items():
            for lang, nm in per.items():
                out.write(f"{gid}\t{lang}\t{nm}\n")
    return names


def main() -> None:
    print("=== build_geodata (F26/G1) ===", flush=True)
    ids, ncity = build_places()
    print(f"places: {len(ids)} places (of them PPLA*/PPLC cities: {ncity})", flush=True)
    regions = build_admin1()
    print(f"admin1: {len(regions)} regions", flush=True)
    countries = build_countries()
    print(f"countries: {len(countries)} countries", flush=True)
    # region/country geonameids are also captured for localized names (G-#19)
    names = build_names(ids | set(regions) | set(countries))

    def cov(lang: str) -> int:
        return sum(1 for per in names.values() if lang in per)

    total = len(ids)
    print("\n=== NAME COVERAGE (share of places with a name in the language) ===", flush=True)
    for lang in ("en", "ru", "ja"):
        c = cov(lang)
        print(f"  {lang}: {c}/{total}  ({100 * c // max(total, 1)}%)", flush=True)

    # ja by DISTRICTS (non-cities) — the key open question
    import csv
    city_ids, all_ids = set(), set()
    with (OUT / "places.tsv").open(encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t"):
            all_ids.add(row[0])
            if row[3] in CITY_FCODES:
                city_ids.add(row[0])
    dist_ids = all_ids - city_ids
    ja_city = sum(1 for g in city_ids if "ja" in names.get(g, {}))
    ja_dist = sum(1 for g in dist_ids if "ja" in names.get(g, {}))
    ru_dist = sum(1 for g in dist_ids if "ru" in names.get(g, {}))
    print("\n=== ja/ru by level ===", flush=True)
    print(f"  cities:    ja {ja_city}/{len(city_ids)}", flush=True)
    print(f"  districts: ja {ja_dist}/{len(dist_ids)}   ru {ru_dist}/{len(dist_ids)}", flush=True)

    # coverage of localized names for regions/countries (G-#19)
    reg_ru = sum(1 for g in regions if "ru" in names.get(g, {}))
    cc_ru = sum(1 for g in countries if "ru" in names.get(g, {}))
    print("\n=== region/country ru coverage (G-#19) ===", flush=True)
    print(f"  regions:   ru {reg_ru}/{len(regions)}", flush=True)
    print(f"  countries: ru {cc_ru}/{len(countries)}", flush=True)

    for f in ("places.tsv", "names.tsv", "admin1.tsv", "countries.tsv"):
        print(f"\n{f}: {(OUT / f).stat().st_size // 1024} KB", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
