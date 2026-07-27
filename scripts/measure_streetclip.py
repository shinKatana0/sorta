"""F85b (discovery): is StreetCLIP worth a gigabyte ON THIS collection?

The published numbers for the model (74% country-level, 92% above 0.7 confidence) were
measured on somebody else's distribution — street-level panoramas spread over the
whole world. Half of this collection is four or five countries and a large part of it
is not street at all: interiors, food, portraits. A gigabyte of weights and a second
model in the pipeline cannot be bought on a foreign benchmark, so this script measures
the model here, on the material it would actually run on.

Method — the hidden check of F85a: take the files whose country is known EXACTLY from
EXIF GPS, hide the coordinates, ask StreetCLIP, compare. No labelling, no human looks
at an image. The sample is stratified per country, because 10k Saint Petersburg frames
would produce a headline accuracy that says nothing about Bali.

The label set is EVERY country in the bundled geo base (252), not the eight present in
the collection. Scoring against a short list would measure a task the pipeline never
faces: in production the true country is not known to be among a handful, and the model
would have to pick out of everything.

Pre-registered decision rule (written before the first run, see `verdict`): build the
feature only if some confidence threshold reaches precision >= 95% while still covering
>= 20% of the candidates. A wrong place is worse than an empty one — an empty folder
gets sorted by hand, a photo moved to the wrong country is never found again.

Privacy: aggregates only. No path, no basename, no image is ever printed or written;
the cache holds file ids and country codes (see the same rule in measure_ocr_gate.py).

This is discovery. Nothing here is imported by the pipeline, and the pipeline is not
touched: the script owns the model, the prompts and the tables.

Result of the run this script was written for (2026-07-27, 1 056 GPS-truth frames,
250 per country, 252 country labels) — the answer is NO, do not build the feature:

    accuracy 36.0% overall (claimed 74%), and worst exactly where the collection is
    heaviest: TR 16.8%, TH 30.4%, RU 36.8%, ID 54.4%;
    the best threshold is 0.90 — precision 93.9% at 6.2% coverage, so it clears
    neither bar; at the lowest threshold that still covers 20% precision is 81.8%,
    i.e. ~264 of the candidate files would land in the wrong country;
    confidence carries little signal: frames WITH faces fire almost as often as
    landscapes (11.4% vs 17.6% above 0.70) and are wrong more often, and the
    place-less frames — 43% of them with faces — fire at the same rate as honest
    GPS-bearing street shots;
    cost, for the record: 1.7 GB of weights, 3.5 GB peak VRAM, 48 ms/frame on a
    5090 (~5.5 min for 6 800 files), model licensed CC BY-NC 4.0.

The full write-up with every table lives in the F85b brief, which is not part of the
published tree; the script stays so the measurement can be repeated when a different
model (or a different collection) makes the question worth asking again.

Usage (from the repo root, with the venv python):
    python scripts/measure_streetclip.py --config config.yaml
    python scripts/measure_streetclip.py --per-country 250 --cache streetclip.json
    python scripts/measure_streetclip.py --cache streetclip.json   # replay, no model
    python scripts/measure_streetclip.py --faces-sample 300        # point 4 of the brief
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sorta.config import load_config  # noqa: E402
from sorta.geodata import GeoResolver  # noqa: E402
from sorta.landmarks import batched  # noqa: E402

MODEL_ID = "geolocal/StreetCLIP"

# The caption template of the paper, at country level. It is a parameter of the script
# but NOT a knob to turn until the numbers look good: the brief forbids tuning prompts
# against the result, because a prompt fitted to this sample measures the fitting, not
# the model.
PROMPT_TEMPLATE = "A Street View photo in {country}."

THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9)

# The decision rule, in code so it cannot be quietly relaxed after seeing the table.
MIN_PRECISION = 0.95
MIN_COVERAGE = 0.20

# What the feature would have to run on: the files with no place and no GPS-bearing
# neighbour (8 556 place-less minus the 1 758 F85a can still reach).
CANDIDATE_FILES = 6800

CACHE_VERSION = 1

# (paths, batch size) -> best country index + its probability per path; None at a
# position — the file did not decode. Replaced in tests.
Scorer = Callable[[list[str], int], list[tuple[int, float] | None]]


@dataclass(frozen=True)
class Pred:
    """One measured file. `true_cc` is empty for the no-truth (place-less) pass."""
    file_id: int
    true_cc: str
    pred_cc: str
    prob: float
    has_faces: bool

    @property
    def correct(self) -> bool:
        return bool(self.true_cc) and self.true_cc == self.pred_cc


@dataclass(frozen=True)
class CurveRow:
    """One point of the threshold curve: what firing above it buys and costs."""
    threshold: float
    fired: int
    correct: int
    total: int

    @property
    def precision(self) -> float:
        return self.correct / self.fired if self.fired else 0.0

    @property
    def coverage(self) -> float:
        return self.fired / self.total if self.total else 0.0


@dataclass(frozen=True)
class CountryRow:
    """One row of the per-country table, with what the model offers instead."""
    cc: str
    n: int
    correct: int
    confusions: list[tuple[str, int]]

    @property
    def accuracy(self) -> float:
        return self.correct / self.n if self.n else 0.0


def load_country_labels(resolver: GeoResolver) -> list[tuple[str, str]]:
    """The label set: (ISO cc, English name) for every country in the bundled base.

    Read straight from countries.tsv rather than through the resolver, because the
    resolver exposes a name per cc but not the full list, and this script must not
    grow the public surface of a module it does not own.
    """
    path = resolver.data_dir / "countries.tsv"
    if not path.exists():
        raise SystemExit(f"{path} not found — run `python scripts/build_geodata.py`")
    labels: list[tuple[str, str]] = []
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) >= 3 and row[0] and row[2]:
                labels.append((row[0], row[2]))
    if not labels:
        raise SystemExit(f"{path} is empty — nothing to score against")
    return sorted(set(labels))


def ground_truth_rows(db_path: str) -> list[sqlite3.Row]:
    """Files whose country is known exactly from EXIF GPS — the hidden ground truth."""
    return _select(db_path, """
        SELECT f.id, f.path, p.country,
               EXISTS(SELECT 1 FROM faces fa
                      WHERE fa.file_id = f.id AND fa.bbox != '[]') AS has_faces
        FROM files f JOIN places p ON p.file_id = f.id
        WHERE p.confidence = 'exact_gps' AND p.country IS NOT NULL AND p.country != ''
          AND f.dup_of IS NULL AND f.error IS NULL AND f.media_type = 'photo'
        ORDER BY f.id""")


def placeless_rows(db_path: str) -> list[sqlite3.Row]:
    """Files with no place at all — the population the feature would run on."""
    return _select(db_path, """
        SELECT f.id, f.path, '' AS country,
               EXISTS(SELECT 1 FROM faces fa
                      WHERE fa.file_id = f.id AND fa.bbox != '[]') AS has_faces
        FROM files f JOIN places p ON p.file_id = f.id
        WHERE p.confidence = 'unknown'
          AND f.dup_of IS NULL AND f.error IS NULL AND f.media_type = 'photo'
        ORDER BY f.id""")


def _select(db_path: str, sql: str) -> list[sqlite3.Row]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute(sql).fetchall()
    finally:
        conn.close()


def stratified_sample(rows: Sequence[sqlite3.Row], per_country: int,
                      seed: int) -> list[sqlite3.Row]:
    """Up to `per_country` files from EVERY country, deterministic for a seed.

    Without this the headline number is the accuracy on the dominant country and
    nothing else: 10 227 of the 14 249 GPS-bearing files here are Russian, so a model
    that answers "Russia" to everything would score 72% and look usable.
    """
    by_cc: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for row in rows:
        by_cc[row["country"]].append(row)
    rnd = random.Random(seed)
    picked: list[sqlite3.Row] = []
    for cc in sorted(by_cc):
        group = list(by_cc[cc])
        rnd.shuffle(group)
        picked.extend(group[:per_country])
    return picked


def existing(rows: Iterable[sqlite3.Row]) -> list[sqlite3.Row]:
    """Drop rows whose file is no longer on disk (an external drive, a move)."""
    return [r for r in rows if Path(r["path"]).exists()]


def threshold_curve(preds: Sequence[Pred],
                    thresholds: Sequence[float] = THRESHOLDS) -> list[CurveRow]:
    """Precision and coverage as a function of the confidence threshold.

    This is the table the decision hangs on: the threshold is what separates "apply the
    prediction" from "leave the place unknown", so accuracy at threshold 0 is a fact
    about the model while these rows are facts about the feature.
    """
    total = len(preds)
    rows: list[CurveRow] = []
    for threshold in thresholds:
        fired = [p for p in preds if p.prob >= threshold]
        rows.append(CurveRow(threshold, len(fired),
                             sum(1 for p in fired if p.correct), total))
    return rows


def country_table(preds: Sequence[Pred], top_confusions: int = 3) -> list[CountryRow]:
    """Per true country: how many frames, how many right, what is offered instead."""
    by_cc: dict[str, list[Pred]] = defaultdict(list)
    for p in preds:
        by_cc[p.true_cc].append(p)
    rows: list[CountryRow] = []
    for cc in sorted(by_cc):
        group = by_cc[cc]
        wrong = Counter(p.pred_cc for p in group if not p.correct)
        rows.append(CountryRow(cc, len(group), sum(1 for p in group if p.correct),
                               wrong.most_common(top_confusions)))
    return rows


def verdict(curve: Sequence[CurveRow], min_precision: float = MIN_PRECISION,
            min_coverage: float = MIN_COVERAGE) -> tuple[bool, str]:
    """The pre-registered criterion applied to the curve -> (do it?, one line).

    A threshold qualifies only if it clears BOTH bars at once. The lowest qualifying
    threshold is reported, since among qualifying ones it covers the most files.
    """
    for row in curve:
        if row.precision >= min_precision and row.coverage >= min_coverage:
            return True, (
                f"ДЕЛАТЬ: порог {row.threshold:.2f} даёт precision "
                f"{row.precision * 100:.1f}% при покрытии {row.coverage * 100:.1f}% "
                f"(критерий: >= {min_precision * 100:.0f}% и >= "
                f"{min_coverage * 100:.0f}%)")
    best = max(curve, key=lambda r: r.precision, default=None)
    detail = (f"лучший порог {best.threshold:.2f}: precision {best.precision * 100:.1f}%"
              f" при покрытии {best.coverage * 100:.1f}%") if best else "нет данных"
    return False, (
        f"НЕ ДЕЛАТЬ: ни один порог не даёт precision >= {min_precision * 100:.0f}% "
        f"при покрытии >= {min_coverage * 100:.0f}% — {detail}")


def face_split(preds: Sequence[Pred]) -> dict[str, tuple[int, int, int, int]]:
    """Point 4 of the brief: does the model behave differently on people shots?

    -> group -> (frames, correct, fired above 0.7, correct among those). A portrait in
    a cafe carries no street evidence, so the interesting number is not the accuracy
    but what the model does with its confidence: a group that fires just as often and
    is right far less is the dangerous failure mode, not the harmless one.
    """
    out: dict[str, tuple[int, int, int, int]] = {}
    for name, group in (("с лицами", [p for p in preds if p.has_faces]),
                        ("без лиц", [p for p in preds if not p.has_faces])):
        confident = [p for p in group if p.prob >= 0.7]
        out[name] = (len(group), sum(1 for p in group if p.correct),
                     len(confident), sum(1 for p in confident if p.correct))
    return out


def _pct(part: int, whole: int) -> str:
    return f"{100.0 * part / whole:.1f}%" if whole else "—"


def format_country_table(rows: Sequence[CountryRow]) -> str:
    """The per-country block. Country codes and counts only — no file is identifiable."""
    total = sum(r.n for r in rows)
    correct = sum(r.correct for r in rows)
    out = [
        "=" * 88,
        f"ПО СТРАНАМ (без порога, argmax по полному списку стран), "
        f"{len(rows)} стран(ы) в выборке",
        f"{'страна':>7} {'кадров':>8} {'угадано':>9} {'точность':>10}  "
        f"что предлагает вместо",
    ]
    for r in sorted(rows, key=lambda r: -r.n):
        instead = ", ".join(f"{cc}:{n}" for cc, n in r.confusions) or "—"
        out.append(f"{r.cc:>7} {r.n:>8} {r.correct:>9} {r.accuracy * 100:>9.1f}%  "
                   f"{instead}")
    out.append(f"{'ИТОГО':>7} {total:>8} {correct:>9} {_pct(correct, total):>10}")
    out.append("=" * 88)
    return "\n".join(out)


def format_curve(curve: Sequence[CurveRow]) -> str:
    """The threshold block: precision against coverage, plus the estimated damage."""
    out = [
        "=" * 88,
        "ПОРОГ УВЕРЕННОСТИ: precision против покрытия",
        f"{'порог':>6} {'сработало':>11} {'покрытие':>10} {'верно':>7} {'неверно':>8} "
        f"{'precision':>10} {'ошибок на 6800':>15}",
    ]
    for r in curve:
        wrong = r.fired - r.correct
        estimated = round(CANDIDATE_FILES * r.coverage * (1 - r.precision))
        out.append(
            f"{r.threshold:>6.2f} {r.fired:>11} {r.coverage * 100:>9.1f}% "
            f"{r.correct:>7} {wrong:>8} {r.precision * 100:>9.1f}% {estimated:>15}")
    out.append("=" * 88)
    out.append(f"«ошибок на {CANDIDATE_FILES}» — оценка числа файлов, уехавших в чужую "
               f"страну,\nесли применить порог ко всем безместным кандидатам.")
    return "\n".join(out)


def format_face_split(split: dict[str, tuple[int, int, int, int]]) -> str:
    out = ["=" * 88, "НЕ-ПЕЙЗАЖИ: кадры с лицами против остальных (та же GPS-истина)",
           f"{'группа':>10} {'кадров':>8} {'точность':>10} {'>= 0.70':>9} "
           f"{'precision >= 0.70':>19}"]
    for name, (n, correct, fired, fired_ok) in split.items():
        out.append(f"{name:>10} {n:>8} {_pct(correct, n):>10} {_pct(fired, n):>9} "
                   f"{_pct(fired_ok, fired):>19}")
    out.append("=" * 88)
    return "\n".join(out)


def format_placeless(preds: Sequence[Pred],
                     thresholds: Sequence[float] = THRESHOLDS) -> str:
    """The no-truth pass: how loudly the model speaks where nobody can check it.

    There is no ground truth for these files by construction — that is the whole
    reason the feature would exist. What can be read off is confidence: if the model
    is as sure here as it is on the GPS-bearing street shots, its confidence carries
    no information about whether it has any evidence.
    """
    total = len(preds)
    with_faces = sum(1 for p in preds if p.has_faces)
    out = ["=" * 88,
           f"БЕЗ МЕСТА (истины нет — только распределение уверенности), {total} кадров, "
           f"из них с лицами {_pct(with_faces, total)}"]
    for threshold in thresholds:
        fired = [p for p in preds if p.prob >= threshold]
        faces = sum(1 for p in fired if p.has_faces)
        out.append(f"  порог {threshold:.2f}: сработало {len(fired):>5} "
                   f"({_pct(len(fired), total):>6}), из них с лицами {_pct(faces, len(fired))}")
    top = Counter(p.pred_cc for p in preds if p.prob >= 0.7).most_common(8)
    out.append("  топ стран при 0.70: " + (", ".join(f"{cc}:{n}" for cc, n in top) or "—"))
    out.append("=" * 88)
    return "\n".join(out)


def save_cache(path: Path, truth: Sequence[Pred], placeless: Sequence[Pred],
               meta: dict) -> None:
    """Per-file aggregates for a replay. File ids and country codes — never paths."""
    def rows(preds: Sequence[Pred]) -> list[list]:
        return [[p.file_id, p.true_cc, p.pred_cc, round(p.prob, 6), int(p.has_faces)]
                for p in preds]

    path.write_text(json.dumps({
        "version": CACHE_VERSION, "meta": meta,
        "truth": rows(truth), "placeless": rows(placeless),
    }, ensure_ascii=False), encoding="utf-8")


def load_cache(path: Path) -> tuple[list[Pred], list[Pred], dict]:
    """-> (truth preds, place-less preds, meta). A foreign version is an error, not a
    silently wrong table."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("version") != CACHE_VERSION:
        raise SystemExit(f"{path}: кэш версии {data.get('version')}, ожидается "
                         f"{CACHE_VERSION} — перемерить с --refresh")

    def preds(raw: list[list]) -> list[Pred]:
        return [Pred(int(fid), str(true), str(pred), float(prob), bool(faces))
                for fid, true, pred, prob, faces in raw]

    return preds(data["truth"]), preds(data.get("placeless") or []), data.get("meta", {})


def classify(scorer: Scorer, rows: Sequence[sqlite3.Row], labels: Sequence[tuple[str, str]],
             batch_size: int, label: str = "") -> tuple[list[Pred], float]:
    """Run the scorer over rows -> (predictions, seconds spent scoring).

    Undecodable files are dropped rather than counted as a miss: a broken file is a
    fact about the collection, and letting it depress the accuracy would flatter the
    threshold curve (a zero-probability row never fires but does inflate the total).
    """
    preds: list[Pred] = []
    done = 0
    started = time.perf_counter()
    for chunk in batched(rows, batch_size):
        scored = scorer([r["path"] for r in chunk], batch_size)
        for row, result in zip(chunk, scored):
            if result is None:
                continue
            best, prob = result
            preds.append(Pred(int(row["id"]), str(row["country"]), labels[best][0],
                              float(prob), bool(row["has_faces"])))
        done += len(chunk)
        print(f"  {label}{done}/{len(rows)}", end="\r", flush=True)
    print(" " * 50, end="\r")
    return preds, time.perf_counter() - started


def bench_batches(scorer: Scorer, paths: Sequence[str],
                  sizes: Sequence[int]) -> list[tuple[int, float]]:
    """-> [(batch size, ms per frame)] over the SAME warm frames.

    Run after the main pass on purpose: by then every preview is in the shared cache,
    so what is timed is the model and not the first decode of a 40 MB HEIC.
    """
    out: list[tuple[int, float]] = []
    for size in sizes:
        started = time.perf_counter()
        for chunk in batched(list(paths), size):
            scorer(list(chunk), size)
        out.append((size, 1000.0 * (time.perf_counter() - started) / max(1, len(paths))))
    return out


def _cuda_peak_gb() -> float | None:  # pragma: no cover — hardware
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_reserved() / 2 ** 30


def _hf_cache_bytes(model_id: str) -> int:  # pragma: no cover — filesystem
    """Bytes the weights occupy in the HuggingFace cache — the real download size."""
    home = os.environ.get("HF_HOME") or os.path.join(Path.home(), ".cache", "huggingface")
    folder = Path(home) / "hub" / ("models--" + model_id.replace("/", "--"))
    if not folder.exists():
        return 0
    return sum(f.stat().st_size for f in folder.rglob("*") if f.is_file())


def streetclip_scorer(labels: Sequence[tuple[str, str]], model_id: str,
                      template: str) -> tuple[Scorer, dict]:  # pragma: no cover — ML
    """The real StreetCLIP zero-shot scorer -> (scorer, meta about the load).

    transformers, not open_clip: StreetCLIP is published as HuggingFace CLIPModel
    weights, and open_clip (which junk/landmarks use) cannot read them without a
    conversion — converting the checkpoint would make this a measurement of the
    conversion. Frames come from the pipeline's own preview cache, so the cost measured
    here is the cost the pipeline would pay on a collection it has already indexed.
    """
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np
    import pillow_heif
    import torch
    from transformers import CLIPModel, CLIPProcessor

    from sorta import imaging

    pillow_heif.register_heif_opener()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    started = time.perf_counter()
    model = CLIPModel.from_pretrained(model_id).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_id)
    load_seconds = time.perf_counter() - started

    size = processor.image_processor.crop_size["height"]
    draft = size * 2  # decode with headroom, exactly as clip_classifier does
    pool = ThreadPoolExecutor(max_workers=max(1, min(os.cpu_count() or 4, 16)))

    prompts = [template.format(country=name) for _cc, name in labels]
    with torch.no_grad():
        text = processor(text=prompts, return_tensors="pt", padding=True).to(device)
        text_feats = model.get_text_features(**text)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
    logit_scale = model.logit_scale.exp().item()

    def _load(path: str):
        try:
            st = os.stat(path)
            return imaging.decode_rgb_preview(path, st.st_mtime, st.st_size,
                                              max_edge=draft)
        except Exception:
            return None

    def scorer(paths: list[str], _batch_size: int) -> list[tuple[int, float] | None]:
        images = list(pool.map(_load, paths))
        out: list[tuple[int, float] | None] = [None] * len(paths)
        valid = [i for i, im in enumerate(images) if im is not None]
        if not valid:
            return out
        pixels = processor(images=[images[i] for i in valid],
                           return_tensors="pt")["pixel_values"].to(device)
        with torch.no_grad():
            feats = model.get_image_features(pixel_values=pixels)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            probs = (logit_scale * feats @ text_feats.T).softmax(dim=-1).cpu().numpy()
        for j, i in enumerate(valid):
            best = int(np.argmax(probs[j]))
            out[i] = (best, float(probs[j][best]))
        return out

    return scorer, {"device": device, "load_seconds": round(load_seconds, 1),
                    "image_size": size, "model": model_id, "prompt": template,
                    "labels": len(labels)}


def main() -> None:
    ap = argparse.ArgumentParser(description="F85b: measure StreetCLIP on this collection")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--db", help="database to read (default: the one in the config)")
    ap.add_argument("--per-country", type=int, default=250,
                    help="cap of GPS-truth files per country (default 250)")
    ap.add_argument("--seed", type=int, default=20260727)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--thresholds", type=float, nargs="+", default=list(THRESHOLDS))
    ap.add_argument("--faces-sample", type=int, default=0, metavar="N",
                    help="also score N place-less files (no truth) — point 4")
    ap.add_argument("--bench", type=int, default=0, metavar="N",
                    help="time N warm frames at batch 1/4/8/16/32")
    ap.add_argument("--cache", help="JSON with the per-file aggregates: written after a "
                                    "measurement, replayed instead of one")
    ap.add_argument("--refresh", action="store_true", help="measure again anyway")
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--prompt", default=PROMPT_TEMPLATE,
                    help="caption template (do NOT tune it against the result)")
    args = ap.parse_args()

    thresholds = sorted(args.thresholds)
    cache = Path(args.cache) if args.cache else None
    if cache and cache.exists() and not args.refresh:
        truth_preds, placeless_preds, meta = load_cache(cache)
        print(f"кэш: {len(truth_preds)} кадров с истиной, {len(placeless_preds)} без "
              f"места, из {cache}")
    else:
        cfg = load_config(args.config)
        db = args.db or str(cfg.database)
        labels = load_country_labels(GeoResolver())
        # `existing` runs BEFORE the stratification, not after: a country whose files
        # live on a drive that is not plugged in would otherwise silently lose most of
        # its quota and the sample would stop being stratified.
        rows = stratified_sample(existing(ground_truth_rows(db)), args.per_country,
                                 args.seed)
        if not rows:
            raise SystemExit("нет файлов с confidence='exact_gps' — сначала `sorta geo`")
        by_cc = Counter(r["country"] for r in rows)
        print(f"выборка: {len(rows)} кадров с точной GPS-истиной, "
              f"{len(by_cc)} стран(ы), кап {args.per_country} на страну")
        print("  " + ", ".join(f"{cc}:{n}" for cc, n in by_cc.most_common()))
        print(f"меряем {args.model} против {len(labels)} стран, "
              f"промпт {args.prompt!r}")

        scorer, meta = streetclip_scorer(labels, args.model, args.prompt)
        print(f"модель загружена за {meta['load_seconds']} с на {meta['device']}, "
              f"вход {meta['image_size']}px, "
              f"кэш весов {_hf_cache_bytes(args.model) / 2 ** 20:.0f} МБ")

        truth_preds, seconds = classify(scorer, rows, labels, args.batch)
        meta["seconds_per_frame"] = round(seconds / max(1, len(truth_preds)), 4)
        meta["batch"] = args.batch
        print(f"истина: {len(truth_preds)} кадров за {seconds:.0f} с "
              f"({1000 * seconds / max(1, len(truth_preds)):.0f} мс/кадр)")

        placeless_preds = []
        if args.faces_sample:
            pick = existing(placeless_rows(db))
            rnd = random.Random(args.seed)
            rnd.shuffle(pick)
            pick = pick[:args.faces_sample]
            placeless_preds, _sec = classify(scorer, pick, labels, args.batch, "без места ")
            print(f"без места: {len(placeless_preds)} кадров")

        if args.bench:
            warm = [r["path"] for r in rows[:args.bench]]
            for size, ms in bench_batches(scorer, warm, (1, 4, 8, 16, 32)):
                print(f"  батч {size:>3}: {ms:.0f} мс/кадр")

        peak = _cuda_peak_gb()
        meta["vram_peak_gb"] = round(peak, 2) if peak is not None else None
        meta["weights_mb"] = round(_hf_cache_bytes(args.model) / 2 ** 20)
        if cache:
            save_cache(cache, truth_preds, placeless_preds, meta)
            print(f"кэш записан: {cache} (только file_id и коды стран, без путей)")

    print()
    print(format_country_table(country_table(truth_preds)))
    curve = threshold_curve(truth_preds, thresholds)
    print(format_curve(curve))
    print(format_face_split(face_split(truth_preds)))
    if placeless_preds:
        print(format_placeless(placeless_preds, thresholds))
    if meta:
        print(f"модель {meta.get('model')}, {meta.get('labels')} стран, "
              f"промпт {meta.get('prompt')!r}, {meta.get('device')}, "
              f"загрузка {meta.get('load_seconds')} с, веса {meta.get('weights_mb')} МБ, "
              f"пик VRAM {meta.get('vram_peak_gb')} ГБ, "
              f"{meta.get('seconds_per_frame')} с/кадр (батч {meta.get('batch')})")
    print()
    print(verdict(curve)[1])


if __name__ == "__main__":
    main()
