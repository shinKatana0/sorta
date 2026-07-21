"""A geo-hierarchy validation spike (F26). Needs NETWORK — run on your own machine.

Checks the offline hypothesis on real coordinates from photos.db:
  1) does "nearest city" (cities15000) give the actual city (SPb, not a district)?
  2) does the feature code separate a city (PPLA/PPLC) from a district (PPLX)?
  3) are native names (Cyrillic/Japanese) reachable from the dumps?

Run:  uv run python scripts/geo_spike.py
Downloads ~10-15 MB into a temp folder (cities15000.zip + RU.zip), commits nothing.
Based on the results we decide the data source and the size of the bundled set.
"""
from __future__ import annotations

import io
import sqlite3
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

GEO = "https://download.geonames.org/export/dump/"
COLS = ["geonameid", "name", "asciiname", "alternatenames", "lat", "lon",
        "fclass", "fcode", "cc", "cc2", "admin1", "admin2", "admin3", "admin4",
        "population", "elevation", "dem", "tz", "moddate"]
# "real city" feature codes (seat of admin) vs other populated places
CITY_FCODES = {"PPLC", "PPLA", "PPLA2", "PPLA3", "PPLA4"}


def fetch_txt(zip_name: str, txt_name: str, dst: Path) -> Path:
    out = dst / txt_name
    if out.exists():
        return out
    print(f"  downloading {zip_name} …")
    data = urllib.request.urlopen(GEO + zip_name, timeout=60).read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        z.extract(txt_name, dst)
    print(f"  {txt_name}: {out.stat().st_size // 1024} KB")
    return out


def load_rows(txt: Path) -> list[dict]:
    rows = []
    with open(txt, encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 19:
                continue
            rows.append(dict(zip(COLS, p)))
    return rows


def has_cyrillic(s: str) -> bool:
    return any("Ѐ" <= ch <= "ӿ" for ch in s)


def has_japanese(s: str) -> bool:
    return any("぀" <= ch <= "ヿ" or "一" <= ch <= "鿿" for ch in s)


def main() -> None:
    tmp = Path(tempfile.gettempdir()) / "sorta_geo_spike"
    tmp.mkdir(exist_ok=True)
    cities_txt = fetch_txt("cities15000.zip", "cities15000.txt", tmp)
    ru_txt = fetch_txt("RU.zip", "RU.txt", tmp)

    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # else a cp1251 console crashes

    all_cities = load_rows(cities_txt)
    # ONLY real cities (seat of admin) — districts (PPLX) are excluded from the city level
    cities = [c for c in all_cities if c["fcode"] in CITY_FCODES]
    cxy = np.radians(np.array([[float(c["lat"]), float(c["lon"])] for c in cities]))

    def nearest_city(lat: float, lon: float) -> dict:
        p = np.radians([lat, lon])
        dlat = cxy[:, 0] - p[0]
        dlon = cxy[:, 1] - p[1]
        a = np.sin(dlat / 2) ** 2 + np.cos(cxy[:, 0]) * np.cos(p[0]) * np.sin(dlon / 2) ** 2
        d = 2 * np.arcsin(np.sqrt(a))
        return cities[int(np.argmin(d))]

    # coordinates from the real collection
    conn = sqlite3.connect("photos.db")
    samples = []
    for like in ("%Akademicheskoe%", "%Krestovskiy%", "%Taganskiy%", "%Ban Khao Lak%", "%Sochi%"):
        r = conn.execute(
            """SELECT f.gps_lat, f.gps_lon, p.city FROM files f JOIN places p ON p.file_id = f.id
               WHERE p.city LIKE ? AND f.gps_lat IS NOT NULL LIMIT 1""", (like,)).fetchone()
        if r:
            samples.append(r)
    conn.close()

    print("\n=== STRUCTURE: district (rg) -> nearest CITY (cities15000) ===")
    for lat, lon, cur_city in samples:
        c = nearest_city(lat, lon)
        alts = c["alternatenames"].split(",") if c["alternatenames"] else []
        ru = next((a for a in alts if has_cyrillic(a)), "—")
        ja = next((a for a in alts if has_japanese(a)), "—")
        print(f"  rg='{cur_city}' -> city='{c['name']}' fcode={c['fcode']} "
              f"pop={c['population']}  ru='{ru}' ja='{ja}'")

    print("\n=== DISTRICT: names/feature in the RU dump (for 4 SPb districts) ===")
    ru_rows = load_rows(ru_txt)
    by_name = {}
    for r in ru_rows:
        by_name.setdefault(r["name"], r)
    for nm in ("Akademicheskoe", "Krestovskiy ostrov", "Centralniy", "Petrogradka"):
        r = by_name.get(nm)
        if not r:
            print(f"  {nm}: not found in RU.txt")
            continue
        alts = r["alternatenames"].split(",") if r["alternatenames"] else []
        ru = next((a for a in alts if has_cyrillic(a)), "—")
        ja = next((a for a in alts if has_japanese(a)), "—")
        print(f"  {nm}: fcode={r['fcode']} admin1={r['admin1']} admin2={r['admin2']} "
              f"ru='{ru}' ja='{ja}'  (alt total: {len(alts)})")

    print(f"\ncities15000: {len(cities)} cities (PPLA*/PPLC) of {len(all_cities)}; RU.txt: {len(ru_rows)} places.")
    print("The conclusion on Japanese/source size — we discuss based on this output.")


if __name__ == "__main__":
    main()
