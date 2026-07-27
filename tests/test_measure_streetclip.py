"""F85b: the StreetCLIP probe — everything about it that is not the model itself.

The script decides whether a gigabyte of weights enters the project, so the arithmetic
behind that decision has to be trustworthy without a GPU: the sampling, the tables, the
threshold curve and the pre-registered criterion are pure functions over per-file
aggregates and are tested here with a fake scorer, no transformers and no photo.

Two of these tests are about the brief rather than about code, and they stay even if
they look pedantic:

* the criterion (precision >= 95% at coverage >= 20%) was written down BEFORE the
  measurement — `test_verdict_*` pins it so a disappointing table cannot be met by
  quietly lowering the bar;
* nothing the script prints or caches may identify a file — a place probe must not turn
  into a list of where somebody's photos were taken (the rule of the document verdict,
  and of measure_ocr_gate.py before this).
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

from sorta.db import connect

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "measure_streetclip.py"


def _load_script():
    """Import scripts/measure_streetclip.py — it is a script, not a package module."""
    spec = importlib.util.spec_from_file_location("measure_streetclip", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


probe = _load_script()

LABELS = [("FR", "France"), ("ID", "Indonesia"), ("RU", "Russia"), ("TH", "Thailand")]


def pred(true_cc: str, pred_cc: str, prob: float, has_faces: bool = False,
         file_id: int = 1):
    return probe.Pred(file_id, true_cc, pred_cc, prob, has_faces)


def row(file_id: int, country: str, path: str = "x.jpg", has_faces: int = 0):
    """A stand-in for the sqlite3.Row the queries return (same subscript access)."""
    return {"id": file_id, "path": path, "country": country, "has_faces": has_faces}


class TestStratifiedSample(unittest.TestCase):
    """Without stratification the headline number is the dominant country's accuracy."""

    def _rows(self):
        return ([row(i, "RU") for i in range(100)]
                + [row(100 + i, "TH") for i in range(30)]
                + [row(200 + i, "MV") for i in range(3)])

    def test_every_country_is_capped_at_the_quota(self):
        picked = probe.stratified_sample(self._rows(), 10, seed=1)
        counts = {cc: sum(1 for r in picked if r["country"] == cc)
                  for cc in ("RU", "TH", "MV")}
        # a country with fewer files than the quota contributes all of them, not none
        self.assertEqual(counts, {"RU": 10, "TH": 10, "MV": 3})

    def test_deterministic_for_a_seed_and_sensitive_to_it(self):
        ids = [r["id"] for r in probe.stratified_sample(self._rows(), 5, seed=7)]
        self.assertEqual(ids, [r["id"] for r in probe.stratified_sample(
            self._rows(), 5, seed=7)])
        self.assertNotEqual(ids, [r["id"] for r in probe.stratified_sample(
            self._rows(), 5, seed=8)])

    def test_no_duplicates_and_no_invented_rows(self):
        picked = probe.stratified_sample(self._rows(), 200, seed=3)
        ids = [r["id"] for r in picked]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(picked), 100 + 30 + 3)  # quota above every group size

    def test_empty_input(self):
        self.assertEqual(probe.stratified_sample([], 10, seed=1), [])


class TestThresholdCurve(unittest.TestCase):
    """The threshold is what decides 'apply' vs 'leave unknown' — hence this table."""

    def _preds(self):
        # 4 files: 0.9 right, 0.8 wrong, 0.6 right, 0.4 right
        return [pred("RU", "RU", 0.9), pred("TH", "ID", 0.8),
                pred("ID", "ID", 0.6), pred("FR", "FR", 0.4)]

    def test_precision_and_coverage_per_threshold(self):
        curve = {r.threshold: r for r in probe.threshold_curve(
            self._preds(), (0.5, 0.9))}
        self.assertEqual((curve[0.5].fired, curve[0.5].correct), (3, 2))
        self.assertAlmostEqual(curve[0.5].precision, 2 / 3)
        self.assertAlmostEqual(curve[0.5].coverage, 3 / 4)
        self.assertEqual((curve[0.9].fired, curve[0.9].correct), (1, 1))
        self.assertAlmostEqual(curve[0.9].precision, 1.0)

    def test_threshold_is_inclusive(self):
        curve = probe.threshold_curve([pred("RU", "RU", 0.7)], (0.7,))
        self.assertEqual(curve[0].fired, 1)

    def test_nothing_fires_is_zero_precision_not_a_crash(self):
        curve = probe.threshold_curve(self._preds(), (0.99,))
        self.assertEqual((curve[0].fired, curve[0].precision, curve[0].coverage),
                         (0, 0.0, 0.0))

    def test_coverage_is_over_the_whole_sample(self):
        # the denominator must stay the sample, not the fired rows — otherwise every
        # threshold covers 100% and the criterion becomes unfalsifiable
        curve = probe.threshold_curve(self._preds(), (0.85,))
        self.assertEqual(curve[0].total, 4)
        self.assertAlmostEqual(curve[0].coverage, 0.25)


class TestCountryTable(unittest.TestCase):
    def test_counts_accuracy_and_what_is_offered_instead(self):
        preds = [pred("TH", "TH", 0.9), pred("TH", "ID", 0.8), pred("TH", "ID", 0.7),
                 pred("TH", "MY", 0.6), pred("RU", "RU", 0.9)]
        rows = {r.cc: r for r in probe.country_table(preds)}
        self.assertEqual((rows["TH"].n, rows["TH"].correct), (4, 1))
        self.assertAlmostEqual(rows["TH"].accuracy, 0.25)
        self.assertEqual(rows["TH"].confusions[0], ("ID", 2))
        self.assertEqual(rows["RU"].confusions, [])

    def test_confusions_are_limited_and_ordered(self):
        preds = ([pred("TH", "ID", 0.5)] * 3 + [pred("TH", "MY", 0.5)] * 2
                 + [pred("TH", "VN", 0.5)] + [pred("TH", "LA", 0.5)])
        [row_th] = probe.country_table(preds, top_confusions=2)
        self.assertEqual(row_th.confusions, [("ID", 3), ("MY", 2)])


class TestVerdict(unittest.TestCase):
    """The criterion from the brief, pinned so the table cannot be met halfway."""

    def test_a_qualifying_threshold_says_do_it(self):
        curve = [probe.CurveRow(0.7, fired=30, correct=29, total=100)]
        ok, line = probe.verdict(curve)
        self.assertTrue(ok)
        self.assertIn("ДЕЛАТЬ", line)

    def test_precision_without_coverage_does_not_qualify(self):
        # perfect precision on 5% of the files is not a feature, it is a rounding error
        ok, line = probe.verdict([probe.CurveRow(0.9, fired=5, correct=5, total=100)])
        self.assertFalse(ok)
        self.assertIn("НЕ ДЕЛАТЬ", line)

    def test_coverage_without_precision_does_not_qualify(self):
        ok, _line = probe.verdict([probe.CurveRow(0.5, fired=90, correct=83, total=100)])
        self.assertFalse(ok)

    def test_the_bars_are_the_ones_written_in_the_brief(self):
        self.assertEqual((probe.MIN_PRECISION, probe.MIN_COVERAGE), (0.95, 0.20))
        # exactly on both bars — the criterion is inclusive
        ok, _line = probe.verdict([probe.CurveRow(0.8, fired=20, correct=19, total=100)])
        self.assertTrue(ok)

    def test_the_lowest_qualifying_threshold_wins(self):
        curve = [probe.CurveRow(0.5, fired=50, correct=20, total=100),
                 probe.CurveRow(0.7, fired=40, correct=39, total=100),
                 probe.CurveRow(0.9, fired=25, correct=25, total=100)]
        ok, line = probe.verdict(curve)
        self.assertTrue(ok)
        self.assertIn("0.70", line)  # 0.70 covers more than 0.90 and still clears 95%

    def test_empty_curve_is_a_no(self):
        ok, line = probe.verdict([])
        self.assertFalse(ok)
        self.assertIn("нет данных", line)


class TestFaceSplit(unittest.TestCase):
    def test_people_shots_are_reported_apart(self):
        preds = [pred("RU", "RU", 0.9, has_faces=True),
                 pred("RU", "TH", 0.8, has_faces=True),
                 pred("RU", "TH", 0.3, has_faces=True),
                 pred("RU", "RU", 0.9)]
        split = probe.face_split(preds)
        self.assertEqual(split["с лицами"], (3, 1, 2, 1))  # 3 frames, 1 right, 2 loud
        self.assertEqual(split["без лиц"], (1, 1, 1, 1))

    def test_both_groups_exist_even_when_empty(self):
        split = probe.face_split([pred("RU", "RU", 0.9)])
        self.assertEqual(split["с лицами"], (0, 0, 0, 0))


class TestClassify(unittest.TestCase):
    """The seam between the model and the arithmetic."""

    def test_label_index_becomes_a_country_code(self):
        rows = [row(1, "RU"), row(2, "TH")]

        def scorer(paths, _batch):
            return [(2, 0.91), (3, 0.42)]  # RU, TH in LABELS order

        preds, seconds = probe.classify(scorer, rows, LABELS, batch_size=8)
        self.assertEqual([(p.file_id, p.true_cc, p.pred_cc, p.prob) for p in preds],
                         [(1, "RU", "RU", 0.91), (2, "TH", "TH", 0.42)])
        self.assertGreaterEqual(seconds, 0.0)

    def test_undecodable_files_are_dropped_not_counted_as_misses(self):
        # a broken file that stayed in the sample would never fire but would still
        # inflate the denominator, i.e. flatter every coverage number
        rows = [row(1, "RU"), row(2, "TH"), row(3, "ID")]

        def scorer(paths, _batch):
            return [(2, 0.9), None, (1, 0.7)]

        preds, _sec = probe.classify(scorer, rows, LABELS, batch_size=8)
        self.assertEqual([p.file_id for p in preds], [1, 3])

    def test_batches_cover_every_row(self):
        rows = [row(i, "RU") for i in range(7)]
        seen: list[int] = []

        def scorer(paths, _batch):
            seen.append(len(paths))
            return [(2, 0.5)] * len(paths)

        preds, _sec = probe.classify(scorer, rows, LABELS, batch_size=3)
        self.assertEqual(seen, [3, 3, 1])
        self.assertEqual(len(preds), 7)

    def test_progress_output_carries_no_path(self):
        rows = [row(1, "RU", path=r"D:\SORT\2019 Bali\IMG_0001.jpg")]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            probe.classify(lambda paths, _b: [(2, 0.9)], rows, LABELS, batch_size=4)
        self.assertNotIn("IMG_0001", buf.getvalue())
        self.assertNotIn("SORT", buf.getvalue())


class TestBenchBatches(unittest.TestCase):
    def test_one_row_per_size_over_the_same_frames(self):
        calls: list[int] = []

        def scorer(paths, batch):
            calls.append(batch)
            return [(0, 0.5)] * len(paths)

        result = probe.bench_batches(scorer, ["a", "b", "c", "d"], (1, 4))
        self.assertEqual([size for size, _ms in result], [1, 4])
        self.assertEqual(calls, [1, 1, 1, 1, 4])
        self.assertTrue(all(ms >= 0.0 for _size, ms in result))


class TestCache(unittest.TestCase):
    def test_round_trip(self):
        truth = [pred("RU", "TH", 0.812345, has_faces=True, file_id=11)]
        placeless = [pred("", "FR", 0.5, file_id=22)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.json"
            probe.save_cache(path, truth, placeless, {"device": "cuda"})
            back_truth, back_placeless, meta = probe.load_cache(path)
        self.assertEqual(back_truth[0].file_id, 11)
        self.assertEqual((back_truth[0].true_cc, back_truth[0].pred_cc), ("RU", "TH"))
        self.assertAlmostEqual(back_truth[0].prob, 0.812345)
        self.assertTrue(back_truth[0].has_faces)
        self.assertEqual((back_placeless[0].true_cc, back_placeless[0].pred_cc),
                         ("", "FR"))
        self.assertEqual(meta["device"], "cuda")

    def test_a_foreign_version_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.json"
            path.write_text(json.dumps({"version": 99, "truth": [], "placeless": []}),
                            encoding="utf-8")
            with self.assertRaises(SystemExit):
                probe.load_cache(path)

    def test_the_cache_holds_no_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.json"
            probe.save_cache(path, [pred("RU", "RU", 0.9, file_id=5)], [], {})
            text = path.read_text(encoding="utf-8")
        self.assertNotIn("jpg", text.lower())
        self.assertNotIn("sort", text.lower())


class TestReportPrivacy(unittest.TestCase):
    """A table about places must not become a list of where the photos were taken."""

    def test_the_report_blocks_render_from_aggregates_only(self):
        preds = [pred("RU", "RU", 0.9, has_faces=True), pred("TH", "ID", 0.8),
                 pred("ID", "ID", 0.4)]
        text = "\n".join([
            probe.format_country_table(probe.country_table(preds)),
            probe.format_curve(probe.threshold_curve(preds)),
            probe.format_face_split(probe.face_split(preds)),
            probe.format_placeless([pred("", "RU", 0.9, has_faces=True)]),
        ])
        self.assertIn("RU", text)
        self.assertIn("ИТОГО", text)
        for forbidden in (".jpg", "\\", "/"):
            self.assertNotIn(forbidden, text)

    def test_the_curve_estimates_the_damage_on_the_real_candidates(self):
        # 50% coverage at 80% precision over 6800 candidates -> ~680 files in the
        # wrong country; the number the decision is actually about
        curve = [probe.CurveRow(0.7, fired=50, correct=40, total=100)]
        self.assertIn("680", probe.format_curve(curve))


class TestQueries(unittest.TestCase):
    """The SQL runs against the real schema, so a column rename cannot go unnoticed."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db = Path(self.tmp.name) / "test.db"
        conn = connect(self.db)
        with conn:
            for i, (path, cc, conf, dup, err) in enumerate([
                ("a.jpg", "RU", "exact_gps", None, None),
                ("b.jpg", "TH", "exact_gps", None, None),
                ("c.jpg", None, "unknown", None, None),
                ("d.jpg", "RU", "exact_gps", 1, None),        # a duplicate
                ("e.jpg", "RU", "exact_gps", None, "boom"),   # unreadable
                ("f.jpg", "", "exact_gps", None, None),       # cc never resolved
            ], start=1):
                conn.execute(
                    """INSERT INTO files (id, path, size, mtime, ext, media_type,
                           dup_of, error, indexed_at)
                       VALUES (?, ?, 1, 0, 'jpg', 'photo', ?, ?, '2026-01-01')""",
                    (i, path, dup, err))
                conn.execute(
                    """INSERT INTO places (file_id, country, confidence, updated_at)
                       VALUES (?, ?, ?, '2026-01-01')""", (i, cc, conf))
            conn.execute(
                "INSERT INTO faces (file_id, bbox, embedding) VALUES (2, '[1,2,3,4]', x'00')")
            conn.execute(
                "INSERT INTO faces (file_id, bbox, embedding) VALUES (1, '[]', x'00')")
        conn.close()

    def test_ground_truth_takes_only_exact_gps_with_a_country(self):
        rows = probe.ground_truth_rows(str(self.db))
        self.assertEqual([r["id"] for r in rows], [1, 2])  # no dup, no error, no empty cc

    def test_ground_truth_marks_files_with_faces(self):
        faces = {r["id"]: r["has_faces"] for r in probe.ground_truth_rows(str(self.db))}
        self.assertEqual(faces[2], 1)
        self.assertEqual(faces[1], 0)  # an empty bbox is "detection ran, found nobody"

    def test_placeless_rows_are_the_population_the_feature_would_run_on(self):
        rows = probe.placeless_rows(str(self.db))
        self.assertEqual([r["id"] for r in rows], [3])
        self.assertEqual(rows[0]["country"], "")

    def test_existing_drops_files_that_are_no_longer_on_disk(self):
        here = str(Path(__file__).resolve())
        rows = [row(1, "RU", path=here), row(2, "RU", path=here + ".missing")]
        self.assertEqual([r["id"] for r in probe.existing(rows)], [1])


class TestCountryLabels(unittest.TestCase):
    def test_the_bundled_base_is_the_label_set(self):
        labels = probe.load_country_labels(probe.GeoResolver())
        by_cc = dict(labels)
        self.assertGreater(len(labels), 200)  # the whole world, not the collection
        self.assertEqual(by_cc["RU"], "Russia")
        self.assertEqual(by_cc["ID"], "Indonesia")
        self.assertEqual(labels, sorted(set(labels)))  # stable order: index -> cc

    def test_a_missing_base_is_a_clear_exit_not_an_empty_label_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                probe.load_country_labels(probe.GeoResolver(tmp))

    def test_the_prompt_template_is_country_level(self):
        self.assertIn("{country}", probe.PROMPT_TEMPLATE)
        self.assertEqual(probe.PROMPT_TEMPLATE.format(country="Japan"),
                         "A Street View photo in Japan.")


if __name__ == "__main__":
    unittest.main()
