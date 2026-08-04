"""F184: the measurement of the products the model saw and did not recognize.

The script answers one question with numbers: of the 288 frames the model was asked about
and called photographs, how many does a query over the stored vectors reach, at what depth
and at what price — against a random draw of the same depth, without which "the query finds
N" has nothing to be compared against.

Four properties are what the brief pins, and they are what is tested here:

1. **Not one model call.** The whole cheapness of the feature rests on the vectors being
   on disk already, so the test replaces every VLM entry point of the project with a
   recorder and runs the script end to end: the count has to be zero.
2. **A document never reaches the eye list**, whatever the query and whatever the labels
   say — and it is excluded in SQL rather than after the fact.
3. **Depth is a parameter**, not a literal: the grid moves from the command line and every
   number in the table follows it.
4. **The baseline is drawn from the same set** the ranking came out of.

No model, no GPU, no photograph: everything below is arithmetic over per-frame aggregates,
a temporary index and a fake ranking.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

from sorta import junk, naming
from sorta.config import DEFAULT_SAVED_SLICES
from sorta.db import connect

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "measure_product_recall.py"


def _load_script():
    """Import scripts/measure_product_recall.py — it is a script, not a package module."""
    spec = importlib.util.spec_from_file_location("measure_product_recall", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


recall = _load_script()

PHOTO = "photo"
PRODUCT = "product"
DOCUMENT = "document"


def frame(file_id=1, verdict=PHOTO, asked=True, is_product=True):
    return recall.Frame(file_id=file_id, verdict=verdict, asked=asked,
                        is_product=is_product)


class IndexCase(unittest.TestCase):
    """A temporary index to put labelled frames into."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db = Path(self.tmp.name) / "test.db"
        self.conn = connect(self.db)
        self.addCleanup(self.conn.close)

    def add(self, name, verdict=PHOTO, source="vlm", dup_of=None, error=None):
        """One file with its classification row -> its file id."""
        path = Path(self.tmp.name) / name
        cur = self.conn.execute(
            """INSERT INTO files (path, size, mtime, ext, media_type, indexed_at,
                                  dup_of, error)
               VALUES (?, 1, 0, 'jpg', 'photo', '2026-01-01', ?, ?)""",
            (str(path), dup_of, error))
        if verdict is not None:
            self.conn.execute(
                """INSERT INTO media_class (file_id, verdict, source, score, updated_at,
                       tier) VALUES (?, ?, ?, NULL, '2026-01-01', 'vlm')""",
                (cur.lastrowid, verdict, source))
        self.conn.commit()
        return int(cur.lastrowid)

    def patch(self, target, name, value):
        original = getattr(target, name)
        setattr(target, name, value)
        self.addCleanup(setattr, target, name, original)

    def write_json(self, name, data):
        path = Path(self.tmp.name) / name
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def run_main(self, argv, ranked=()):
        """`main` with the config and the ranking stood in for — no model, no config file."""
        cfg = type("Cfg", (), {
            "database": str(self.db),
            "features": type("Features", (), {"saved_slices": DEFAULT_SAVED_SLICES})(),
        })()
        self.patch(recall, "load_config", lambda _path: cfg)
        self.patch(recall, "ranker",
                   lambda _cfg, _conn, _index: lambda _queries, _depth: list(ranked))
        self.patch(sys, "argv", ["measure_product_recall.py", *argv])
        self.printed = io.StringIO()
        with contextlib.redirect_stdout(self.printed):
            return recall.main()

    def output(self):
        return self.printed.getvalue()


class TestNoModelIsEverAsked(IndexCase):
    """Test 1 of the brief, and the main one: the script makes no VLM call at all.

    Everything it needs is on disk — `clip_embeddings` are computed for the whole
    collection by the fast tier — and that is the entire reason this measurement is cheap.
    A model loaded here by accident would not fail: it would quietly turn a free
    measurement into an hours-long one, so the guard has to be a test.
    """

    def setUp(self):
        super().setUp()
        self.calls: list[str] = []
        for module, name in ((junk, "qwen_vlm_classifier"),
                             (junk, "qwen_vlm_classifier_factory"),
                             (junk, "vlm_classifier_from"),
                             (junk, "qwen_vlm_pet"),
                             (junk, "qwen_vlm_junk_rescue"),
                             (naming, "shared_vlm"),
                             (naming, "qwen_vlm")):
            self.patch(module, name, self.recorder(f"{module.__name__}.{name}"))

    def recorder(self, name):
        def called(*_args, **_kwargs):
            self.calls.append(name)
            raise AssertionError(f"{name} was called by a measurement that must not "
                                 f"touch the model")
        return called

    def labelled(self):
        """One product the model missed, one it found — enough for a whole run."""
        missed = self.add("missed.jpg", verdict=PHOTO)
        found = self.add("found.jpg", verdict=PRODUCT)
        labels = self.write_json("labels.json", {str(missed): True, str(found): True})
        return missed, found, labels

    def test_a_whole_run_asks_the_model_nothing(self):
        missed, found, labels = self.labelled()
        code = self.run_main(["--labels", str(labels), "--depths", "1,2"],
                             ranked=[missed, found])
        self.assertEqual(code, 0)
        self.assertEqual(self.calls, [])

    def test_writing_the_eye_worksheet_asks_the_model_nothing(self):
        _missed, _found, labels = self.labelled()
        sheet = Path(self.tmp.name) / "why.json"
        self.assertEqual(
            self.run_main(["--labels", str(labels), "--write-reasons", str(sheet)]), 0)
        self.assertEqual(self.calls, [])

    def test_the_script_ranks_vectors_and_never_decodes_a_frame(self):
        """The frames are not even on disk: nothing in this path opens an image."""
        missed, found, labels = self.labelled()
        self.assertFalse((Path(self.tmp.name) / "missed.jpg").exists())
        self.assertEqual(
            self.run_main(["--labels", str(labels)], ranked=[missed, found]), 0)
        self.assertEqual(self.calls, [])


class TestTheEyeListHoldsNoDocument(IndexCase):
    """Test 2 of the brief: `verdict='document'` is out of the list under any query.

    Products sit next to passports, certificates and medical forms, so the exclusion is in
    the SQL that builds the list (F133 — a hidden line is not a rule) rather than in a
    filter over the result that somebody has to remember to apply.
    """

    def test_a_document_is_not_a_candidate_for_the_eye_list(self):
        paper = self.add("paper.jpg", verdict=DOCUMENT)
        photo = self.add("photo.jpg", verdict=PHOTO)
        self.assertEqual(set(recall.eye_candidates(self.conn, [paper, photo])), {photo})

    def test_an_unclassified_frame_is_not_mistaken_for_a_document(self):
        unclassified = self.add("new.jpg", verdict=None)
        self.assertEqual(set(recall.eye_candidates(self.conn, [unclassified])),
                         {unclassified})

    def test_the_sample_drops_it_even_when_the_labels_call_it_a_product(self):
        paper = self.add("paper.jpg", verdict=DOCUMENT)
        photo = self.add("photo.jpg", verdict=PHOTO)
        missed = [frame(file_id=paper, verdict=DOCUMENT), frame(file_id=photo)]
        sample = recall.eye_sample(self.conn, missed, size=10, seed=1)
        self.assertEqual(set(sample), {photo})

    def test_asking_for_more_frames_than_there_are_does_not_let_one_in(self):
        paper = self.add("paper.jpg", verdict=DOCUMENT)
        missed = [frame(file_id=paper, verdict=DOCUMENT)]
        self.assertEqual(recall.eye_sample(self.conn, missed, size=500, seed=1), {})

    def test_the_worksheet_written_for_a_person_carries_no_document(self):
        paper = self.add("paper.jpg", verdict=DOCUMENT)
        photo = self.add("photo.jpg", verdict=PHOTO)
        labels = self.write_json("labels.json", {str(paper): True, str(photo): True})
        sheet = Path(self.tmp.name) / "why.json"
        self.run_main(["--labels", str(labels), "--write-reasons", str(sheet)])
        written = json.loads(sheet.read_text(encoding="utf-8"))
        self.assertEqual(set(written), {str(photo)})

    def test_the_exclusion_is_in_the_query_and_not_in_the_marking(self):
        """The rule has to be readable in the SQL, which is the only place it holds."""
        self.assertIn("mc.verdict != '{document}'", recall.EYE_SQL)

    def test_a_missed_product_hidden_under_document_is_reported_as_unreachable(self):
        rows = [recall.Row(depth=10, found=1, shown=8, hidden=2, random_found=0,
                           misses=4)]
        text = recall.format_documents(rows, unreachable=1, misses=4)
        self.assertIn("скрыто до 2", text)
        self.assertIn("1 из 4", text)
        self.assertIn("75.0%", text)  # the ceiling the rule leaves on the recall

    def test_a_rule_that_costs_nothing_here_says_that_instead_of_a_ceiling(self):
        """On today's population it costs nothing: the vector index holds photographs
        only, so a missed product cannot be sitting under `document` in the first place."""
        rows = [recall.Row(depth=10, found=1, shown=10, hidden=0, random_found=0,
                           misses=4)]
        text = recall.format_documents(rows, unreachable=0, misses=4)
        self.assertIn("ни один из 4", text)
        self.assertNotIn("недостижима", text)


class TestDepthIsAParameter(IndexCase):
    """Test 3 of the brief: the grid comes from the caller, not from a literal.

    Depth is the one lever of completeness the measurements confirmed, so a script that
    could not move it would be unable to answer the question it exists for.
    """

    def test_the_grid_parses_in_the_shapes_a_person_types(self):
        self.assertEqual(recall.parse_depths("200,400,800"), [200, 400, 800])
        self.assertEqual(recall.parse_depths("800 200 400"), [200, 400, 800])
        self.assertEqual(recall.parse_depths("400,400,200"), [200, 400])

    def test_an_empty_or_meaningless_grid_stops_the_run(self):
        for text in ("", "   ", "0", "-5"):
            with self.subTest(text=text), self.assertRaises(SystemExit):
                recall.parse_depths(text)

    def test_the_default_grid_doubles(self):
        grid = recall.DEFAULT_DEPTHS
        self.assertEqual(list(grid), [grid[0] * 2 ** i for i in range(len(grid))])

    def test_the_table_follows_the_grid_it_is_given(self):
        ranked = list(range(1, 101))
        rows = recall.sweep(ranked, {}, missed=[3, 40], depths=(2, 50))
        self.assertEqual([r.depth for r in rows], [2, 50])
        self.assertEqual([r.found for r in rows], [0, 2])

    def test_a_single_depth_is_a_grid_too(self):
        rows = recall.sweep(list(range(10)), {}, missed=[1], depths=(1,))
        self.assertEqual([(r.depth, r.found) for r in rows], [(1, 0)])

    def test_the_grid_reaches_the_printed_table_through_main(self):
        product = self.add("a.jpg", verdict=PHOTO)
        labels = self.write_json("labels.json", {str(product): True})
        self.run_main(["--labels", str(labels), "--depths", "7,13"],
                      ranked=[product, *range(100, 200)])
        table = self.output()
        self.assertIn("глубины 7, 13", table)
        for depth in recall.DEFAULT_DEPTHS:
            self.assertNotIn(f"  {depth:>8} ", table)


class TestTheBaselineIsTheSameSet(IndexCase):
    """Test 4 of the brief: the random draw comes out of the ranked population itself.

    A baseline drawn from a wider set would answer an easier question than the ranking is
    being asked, and the comparison between them is the whole point of printing it.
    """

    def test_every_drawn_frame_comes_from_the_ranking(self):
        ranked = list(range(500))
        drawn = recall.random_draw(ranked, 50, random.Random(1))
        self.assertEqual(len(drawn), 50)
        self.assertTrue(set(drawn) <= set(ranked))

    def test_a_draw_deeper_than_the_population_is_the_population(self):
        drawn = recall.random_draw([1, 2, 3], 99, random.Random(1))
        self.assertEqual(sorted(drawn), [1, 2, 3])

    def test_the_same_seed_draws_the_same_frames(self):
        ranked = list(range(200))
        rows = [recall.sweep(ranked, {}, [7, 8, 9], (20,), seed=5) for _ in range(2)]
        self.assertEqual(rows[0][0].random_found, rows[1][0].random_found)

    def test_a_population_that_is_all_misses_gives_the_baseline_all_of_them(self):
        """Drawn from the same set: a set of nothing but misses cannot miss."""
        ranked = [1, 2, 3, 4, 5]
        row = recall.sweep(ranked, {}, ranked, (3,), seed=1)[0]
        self.assertEqual(row.random_found, 3)

    def test_a_population_with_no_misses_in_it_gives_the_baseline_none(self):
        row = recall.sweep([1, 2, 3], {}, missed=[99], depths=(3,), seed=1)[0]
        self.assertEqual((row.found, row.random_found), (0, 0))

    def test_at_full_depth_the_ranking_and_chance_meet(self):
        """The last honest check that both count over the same frames."""
        ranked = list(range(50))
        row = recall.sweep(ranked, {}, missed=[0, 25, 49], depths=(50,), seed=3)[0]
        self.assertEqual((row.found, row.random_found), (3, 3))


class TestTheTwoLosses(unittest.TestCase):
    """The split the whole feature stands on: the answer against the selection.

    A frame the model was asked about and called a photograph is a loss of the ANSWER —
    288 of them — and a frame it was never asked about is a loss of the gate, which the
    brief closed with numbers. Mixing the two would send the next feature after a threshold.
    """

    def test_a_frame_the_model_answered_wrong_is_a_miss_of_the_model(self):
        f = frame(verdict=PHOTO, asked=True)
        self.assertTrue(f.missed_by_model)
        self.assertFalse(f.missed_by_gate or f.found)

    def test_a_frame_the_model_never_saw_is_a_miss_of_the_gate(self):
        f = frame(verdict=PHOTO, asked=False)
        self.assertTrue(f.missed_by_gate)
        self.assertFalse(f.missed_by_model or f.found)

    def test_a_product_already_filed_as_one_is_not_a_miss_at_all(self):
        for asked in (True, False):
            with self.subTest(asked=asked):
                f = frame(verdict=PRODUCT, asked=asked)
                self.assertTrue(f.found)
                self.assertFalse(f.missed_by_model or f.missed_by_gate)

    def test_a_frame_nobody_called_a_product_is_outside_every_bucket(self):
        f = frame(verdict=PHOTO, is_product=False)
        self.assertFalse(f.found or f.missed_by_model or f.missed_by_gate)

    def test_an_unclassified_frame_counts_as_a_miss_and_not_as_a_find(self):
        f = frame(verdict=None, asked=False)
        self.assertTrue(f.missed_by_gate)
        self.assertFalse(f.found)

    def test_the_buckets_partition_the_labelled_products(self):
        frames = [frame(1, PHOTO, True), frame(2, PHOTO, False), frame(3, PRODUCT, True),
                  frame(4, PHOTO, True, is_product=False)]
        products = [f for f in frames if f.is_product]
        counted = sum(1 for f in products
                      if f.found or f.missed_by_model or f.missed_by_gate)
        self.assertEqual(counted, len(products))

    def test_the_head_of_the_report_prints_both_losses(self):
        frames = [frame(1, PHOTO, True), frame(2, PHOTO, False), frame(3, PRODUCT, True)]
        text = recall.format_population(frames)
        self.assertIn("пропустила МОДЕЛЬ", text)
        self.assertIn("пропустил ГЕЙТ", text)
        self.assertIn("33.3%", text)  # one of three products in each bucket


class TestTheIndexIsRead(IndexCase):
    """What the labelled sheet becomes once the index is asked about it."""

    def test_the_verdict_and_who_decided_it_travel_with_the_frame(self):
        asked = self.add("a.jpg", verdict=PHOTO, source="vlm")
        unasked = self.add("b.jpg", verdict=PHOTO, source="clip")
        frames = recall.labelled_frames(self.conn, {asked: True, unasked: True})
        self.assertEqual({f.file_id: f.asked for f in frames},
                         {asked: True, unasked: False})

    def test_duplicates_and_broken_files_are_outside_the_measurement(self):
        keeper = self.add("keep.jpg")
        dup = self.add("dup.jpg", dup_of=1)
        broken = self.add("broken.jpg", error="decode failed")
        frames = recall.labelled_frames(
            self.conn, {keeper: True, dup: True, broken: True})
        self.assertEqual([f.file_id for f in frames], [keeper])

    def test_a_label_for_a_frame_that_is_not_in_the_index_is_dropped(self):
        keeper = self.add("keep.jpg")
        frames = recall.labelled_frames(self.conn, {keeper: True, 99999: True})
        self.assertEqual([f.file_id for f in frames], [keeper])

    def test_a_frame_with_no_classification_row_keeps_its_label(self):
        raw = self.add("raw.jpg", verdict=None)
        frames = recall.labelled_frames(self.conn, {raw: True})
        self.assertEqual((frames[0].verdict, frames[0].asked), (None, False))

    def test_the_verdicts_of_the_whole_index_are_what_the_price_is_read_from(self):
        product = self.add("p.jpg", verdict=PRODUCT)
        paper = self.add("d.jpg", verdict=DOCUMENT)
        self.assertEqual(recall.all_verdicts(self.conn),
                         {product: PRODUCT, paper: DOCUMENT})


class TestPriceAndRecall(unittest.TestCase):
    """The arithmetic of one row — recall without a price is half an answer."""

    def row(self, **kwargs):
        base = dict(depth=100, found=10, shown=80, hidden=5, random_found=2, misses=40)
        return recall.Row(**{**base, **kwargs})

    def test_recall_is_over_the_missed_products(self):
        self.assertAlmostEqual(self.row().recall, 0.25)
        self.assertAlmostEqual(self.row().random_recall, 0.05)

    def test_the_price_is_the_frames_a_person_looks_at_per_find(self):
        self.assertAlmostEqual(self.row().price, 8.0)

    def test_a_list_that_found_nothing_has_no_price(self):
        self.assertIsNone(self.row(found=0).price)

    def test_a_list_beaten_by_chance_keeps_its_lift_of_zero(self):
        """0.0 is a fact about the ranking — chance found two frames and it found none —
        and printing it as «—» would hide the worst case the table can hold."""
        beaten = self.row(found=0, random_found=2)
        self.assertEqual(beaten.lift, 0.0)
        self.assertIn("0.0x", recall.format_depths([beaten], population=100))

    def test_the_lift_is_against_chance_at_the_same_depth(self):
        self.assertAlmostEqual(self.row().lift, 5.0)

    def test_a_baseline_that_found_nothing_gives_no_ratio_instead_of_infinity(self):
        self.assertIsNone(self.row(random_found=0).lift)

    def test_an_empty_set_of_misses_is_not_a_division_by_zero(self):
        empty = self.row(misses=0, found=0, random_found=0)
        self.assertEqual((empty.recall, empty.random_recall), (0.0, 0.0))

    def test_products_and_documents_are_not_charged_to_the_person(self):
        """Already-filed products need no review and documents are never shown."""
        ranked = [1, 2, 3, 4]
        verdicts = {1: PRODUCT, 2: DOCUMENT, 3: PHOTO}
        row = recall.sweep(ranked, verdicts, missed=[3], depths=(4,))[0]
        self.assertEqual((row.shown, row.hidden), (2, 1))  # frames 3 and 4 are shown

    def test_the_table_prints_the_price_and_the_baseline(self):
        text = recall.format_depths([self.row()], population=1000)
        for part in ("глубина", "цена", "случайно", "подъём", "8 кадр/шт", "5.0x"):
            self.assertIn(part, text)


class TestLabels(unittest.TestCase):
    """The sheet of 999 marks, in the shapes such a sheet arrives in."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def load(self, data):
        path = Path(self.tmp.name) / "labels.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return recall.load_labels(path)

    def test_a_flat_sheet_of_marks(self):
        self.assertEqual(self.load({"1": True, "2": False}), {1: True, 2: False})

    def test_the_three_layers_of_the_sample_are_merged(self):
        data = {"marked_product": {"1": True}, "answered_photo": {"2": True},
                "never_asked": {"3": False}}
        self.assertEqual(self.load(data), {1: True, 2: True, 3: False})

    def test_a_list_of_rows_in_either_shape(self):
        self.assertEqual(self.load([[1, True], [2, False]]), {1: True, 2: False})
        self.assertEqual(self.load([{"file_id": 3, "product": True}]), {3: True})

    def test_a_cell_still_holding_null_is_not_a_mark(self):
        self.assertEqual(self.load({"1": True, "2": None}), {1: True})

    def test_words_are_read_the_way_a_person_wrote_them(self):
        self.assertEqual(self.load({"1": "product", "2": "photo", "3": "yes"}),
                         {1: True, 2: False, 3: True})

    def test_an_unrecognized_mark_stops_the_run_instead_of_becoming_false(self):
        with self.assertRaises(SystemExit):
            self.load({"1": "maybe"})

    def test_a_sheet_with_no_marks_at_all_stops_the_run(self):
        with self.assertRaises(SystemExit):
            self.load({"1": None})

    def test_a_shape_nobody_recognizes_stops_the_run(self):
        with self.assertRaises(SystemExit):
            self.load("nonsense")


class TestReasons(unittest.TestCase):
    """Hypothesis A: the breakdown of the missed frames by why the model said no."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def sheet(self, data):
        path = Path(self.tmp.name) / "why.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_the_three_reasons_of_the_brief_are_the_vocabulary(self):
        self.assertEqual(set(recall.REASONS),
                         {"borderline", "narrow", "feature_missing", "other"})

    def test_a_worksheet_is_file_ids_and_nothing_else(self):
        path = Path(self.tmp.name) / "sheet.json"
        self.assertEqual(recall.write_reason_template(path, [7, 8]), 2)
        self.assertEqual(json.loads(path.read_text(encoding="utf-8")),
                         {"7": None, "8": None})

    def test_an_unanswered_cell_is_not_a_reason(self):
        marks = recall.load_reasons(self.sheet({"1": "narrow", "2": None}))
        self.assertEqual(marks, {1: "narrow"})

    def test_the_counts_are_per_reason(self):
        counts = recall.count_reasons({1: "narrow", 2: "narrow", 3: "borderline"})
        self.assertEqual(dict(counts), {"narrow": 2, "borderline": 1})

    def test_the_breakdown_names_the_sample_it_describes(self):
        text = recall.format_reasons(recall.count_reasons({1: "narrow"}), marked=1,
                                     sample=60, misses=288)
        self.assertIn("размечено 1 из 60", text)
        self.assertIn("288", text)
        self.assertIn("narrow", text)

    def test_a_reason_outside_the_vocabulary_is_kept_and_flagged(self):
        counts = recall.count_reasons({1: "watermark"})
        text = recall.format_reasons(counts, marked=1, sample=60, misses=288)
        self.assertIn("watermark", text)
        self.assertIn("вне словаря", text)

    def test_an_empty_sheet_says_so_instead_of_printing_a_table(self):
        text = recall.format_reasons(recall.count_reasons({}), marked=0, sample=60,
                                     misses=288)
        self.assertIn("null", text)


class TestOutcome(unittest.TestCase):
    """A / B / C by the pre-registered criteria, including at their boundaries."""

    def rows(self, found, random_found, misses=288, depth=3200, shown=1000):
        return [recall.Row(depth=depth, found=found, shown=shown, hidden=0,
                           random_found=random_found, misses=misses)]

    def test_the_thresholds_are_the_ones_the_brief_registered(self):
        self.assertEqual((recall.RECALL_MIN, recall.LIFT_MIN), (0.5, 2.0))

    def test_half_the_misses_with_a_lift_is_outcome_a(self):
        letter, why = recall.decide(self.rows(found=200, random_found=20))
        self.assertEqual(letter, "A")
        self.assertIn("второе мнение запросом работает", why)
        self.assertIn("кадров на находку", why)

    def test_a_real_but_shallow_signal_is_outcome_b(self):
        letter, why = recall.decide(self.rows(found=100, random_found=10))
        self.assertEqual(letter, "B")
        self.assertIn("мерить глубже", why)

    def test_no_lift_over_chance_is_outcome_c(self):
        letter, why = recall.decide(self.rows(found=200, random_found=150))
        self.assertEqual(letter, "C")
        self.assertIn("разборе причин", why)

    def test_the_thresholds_are_inclusive(self):
        at_a = self.rows(found=144, random_found=72)  # exactly 50% and exactly 2.0x
        self.assertEqual(recall.decide(at_a)[0], "A")

    def test_the_deepest_row_is_the_one_decided_on(self):
        """The prefixes grow, so the deepest list is where the recall is highest —
        picking the best of several depths would be choosing a number after seeing it."""
        shallow = recall.Row(depth=100, found=200, shown=90, hidden=0, random_found=1,
                             misses=288)
        deep = recall.Row(depth=3200, found=20, shown=3000, hidden=0, random_found=15,
                          misses=288)
        self.assertEqual(recall.decide([shallow, deep])[0], "C")

    def test_a_chance_that_found_nothing_is_not_a_refusal_to_decide(self):
        letter, why = recall.decide(self.rows(found=200, random_found=0))
        self.assertEqual(letter, "A")
        self.assertIn("случайный отбор не нашёл ничего", why)

    def test_nothing_to_measure_is_said_and_not_divided_by_zero(self):
        letter, why = recall.decide(self.rows(found=0, random_found=0, misses=0))
        self.assertEqual(letter, "C")
        self.assertIn("не пропустила", why)

    def test_an_empty_table_stops_instead_of_deciding(self):
        with self.assertRaises(SystemExit):
            recall.decide([])

    def test_the_outcome_line_carries_the_letter(self):
        self.assertTrue(recall.format_outcome(
            self.rows(found=200, random_found=20)).startswith("ИСХОД A"))


class TestQueries(unittest.TestCase):
    """The phrases: the slice the product already ships, not a private copy of it."""

    def config(self, slices):
        return type("Cfg", (), {"features": type("F", (), {"saved_slices": slices})()})()

    def test_the_shipped_slice_is_what_is_measured(self):
        cfg = self.config(DEFAULT_SAVED_SLICES)
        products = next(s for s in DEFAULT_SAVED_SLICES if s.name == "products")
        self.assertEqual(recall.product_queries(cfg, None), list(products.queries))

    def test_a_retuned_slice_is_followed(self):
        retuned = type("S", (), {"name": "products", "queries": ("a shop shelf",)})()
        self.assertEqual(recall.product_queries(self.config((retuned,)), None),
                         ["a shop shelf"])

    def test_the_command_line_overrides_it(self):
        self.assertEqual(
            recall.product_queries(self.config(DEFAULT_SAVED_SLICES), ["a price tag"]),
            ["a price tag"])

    def test_a_config_without_the_slice_falls_back_to_the_shipped_phrases(self):
        cfg = self.config(())
        self.assertTrue(recall.product_queries(cfg, None))


class TestReportIdentifiesNothing(unittest.TestCase):
    """The aggregate report is counts: no path, no basename, no file id.

    Paths appear only in the eye list, and only when `--paths` was asked for — the one
    job that cannot be done without seeing the frames.
    """

    def test_no_frame_identity_reaches_the_printed_tables(self):
        rows = [recall.Row(depth=200, found=30, shown=180, hidden=4, random_found=3,
                           misses=288)]
        frames = [frame(1, PHOTO, True), frame(2, PRODUCT, True), frame(3, PHOTO, False)]
        text = "\n".join([recall.format_population(frames), recall.format_depths(rows, 20),
                          recall.format_documents(rows, 5, 288),
                          recall.format_outcome(rows)])
        for leak in ("/photos", ".jpg", "IMG_", "\\", "file_id", "id="):
            self.assertNotIn(leak, text)


class TestMainRefusesWhenThereIsNothingToMeasure(IndexCase):
    """The two states where a table would be a claim about nothing."""

    def test_labels_that_belong_to_another_collection_stop_the_run(self):
        labels = self.write_json("labels.json", {"4242": True})
        with self.assertRaises(SystemExit) as caught:
            self.run_main(["--labels", str(labels)])
        self.assertIn("не найден в индексе", str(caught.exception))

    def test_a_sample_the_model_missed_nothing_in_stops_the_run(self):
        found = self.add("found.jpg", verdict=PRODUCT)
        labels = self.write_json("labels.json", {str(found): True})
        with self.assertRaises(SystemExit) as caught:
            self.run_main(["--labels", str(labels)], ranked=[found])
        self.assertIn("не пропустила", str(caught.exception))

    def test_a_full_run_prints_the_table_the_reasons_hint_and_the_outcome(self):
        missed = self.add("missed.jpg", verdict=PHOTO)
        other = self.add("other.jpg", verdict=PHOTO)
        labels = self.write_json("labels.json", {str(missed): True, str(other): False})
        self.assertEqual(
            self.run_main(["--labels", str(labels), "--depths", "1,2"],
                          ranked=[missed, other]), 0)

    def test_a_filled_reason_sheet_is_read_by_the_run(self):
        missed = self.add("missed.jpg", verdict=PHOTO)
        labels = self.write_json("labels.json", {str(missed): True})
        sheet = self.write_json("why.json", {str(missed): "narrow"})
        self.assertEqual(
            self.run_main(["--labels", str(labels), "--depths", "1",
                           "--reasons", str(sheet)], ranked=[missed]), 0)


if __name__ == "__main__":
    unittest.main()
