"""F160: the detector's answer reaches the screen — and the rule stays written once.

F154 shipped the detector: it ranks candidates, runs the model, stores `detections` and
overrides the CLIP label in both directions. What it could not reach is where the answer is
actually READ. Since F137 the album, the "Animals" tab and the Overview counter derive the
verdict at read time through `sorter.animal_auto_sql`, and that expression knew nothing
about the new table — so a run spent three minutes, the boxes went into the database, and
nothing a user looks at moved.

The cause is wider than one missing branch. "What counts as an animal" is written TWICE, on
purpose (`junk.pet_label` labels the one frame a stage has just scored; `animal_auto_sql`
answers "which files" over a whole index, and a Python loop over 20 000 rows is not that
question) — and by now four things decide it: the CLIP score (F122), the VLM answer (F130),
the user (F124) and the detector (F154). Every one of them had to be written into both
halves, and nothing checked that they were.

So the main test of this file is not "the detector shows up in the slice". It is THE CASE
TABLE, below: every combination of score, answer, detection and manual mark, run through
both spellings, asserted equal row by row. A fifth source that lands in only one of them
fails here rather than in a user's album.

The case table covers what the WRITER can produce (`detect.pack_boxes` over
`detect.animal_boxes`) plus the ways a stored row degrades: no row at all, a row from
another detector, a row with no boxes, a box of a class that is not an animal, and a column
that is not JSON. A hand-edited row whose box COORDINATES are not numbers is outside both
spellings' contract and is not claimed here — the classes and the scores are what either
side reads.

No model is loaded anywhere in this file, and no stage is run: every case writes the rows a
run would have left and then only moves the config, which is the property F137 bought and
this feature has to keep.
"""
from __future__ import annotations

import dataclasses
import io
import itertools
import json
import sqlite3
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from sorta import ui
from sorta.config import Config, DetectConfig, FeaturesConfig
from sorta.db import connect
from sorta.detect import (
    STORE_FLOOR,
    Detection,
    DetectorSettings,
    animal_boxes,
    best_animal,
    detector_settings,
    pack_boxes,
    unpack_boxes,
)
from sorta.junk import PET_VLM_DEPICTION, PET_VLM_NONE, PET_VLM_REAL, pet_label
from sorta.sorter import animal_auto_sql, animal_ids_sql, plan_album

from tests.test_ui_animals import AnimalsTestBase

# The boxes of one frame as the stage stores them: `detect.animal_boxes` at the storage
# floor, packed by `detect.pack_boxes`. Written through the writer's own functions rather
# than by hand, so a change in either would show up here instead of in a live archive.
_CASES_BOXES = {
    "none": None,                                            # the detector never examined
    "empty": [],                                             # looked, found nothing
    "weak": [Detection("cat", 0.30, (1.0, 2.0, 3.0, 4.0))],  # below the 0.5 threshold
    "strong": [Detection("cat", 0.90, (1.0, 2.0, 3.0, 4.0))],
    "crowd": [Detection("dog", 0.95, (1.0, 2.0, 3.0, 4.0)),
              Detection("cat", 0.40, (5.0, 6.0, 7.0, 8.0))],
    "person": [Detection("person", 0.99, (1.0, 2.0, 3.0, 4.0))],  # never an animal
}
# The same stored answer, but written by ANOTHER detector — not this one's answer, so the
# frame reads as one nobody has examined (`junk._DetectorPass._stored` keys the same way).
_OTHER_MODEL = "retinanet_resnet50_fpn"

_SCORES = (None, 0.10, 0.35, 0.80, 0.95)
_ANSWERS = (None, PET_VLM_REAL, PET_VLM_DEPICTION, PET_VLM_NONE)
_MARKS = (None, True, False)
# Thresholds the whole table is replayed at. The point of deriving the verdict at read
# time is that these move without a run, so a parity that only holds at the defaults is
# not the parity being claimed.
_THRESHOLD_GRID = (
    # (pet_threshold, pet_candidate_threshold, detector_threshold)
    (0.70, 0.30, 0.50),
    (0.20, 0.30, 0.50),
    (0.70, 0.90, 0.50),   # the gate has risen: stored answers stop counting
    (0.70, 0.30, 0.20),   # the detector's threshold re-chosen downwards, off the boxes
    (0.70, 0.30, 0.95),   # ...and upwards
)


def detector_answer(conn: sqlite3.Connection, file_id: int,
                    detector: DetectorSettings) -> bool | None:
    """What the detector said about one frame — the READER's half, in Python.

    True: it examined the frame and found an animal at or above the threshold in force
    now; False: it examined the frame and found none; None: it never examined it (no row,
    a row from another detector, or the whole tier switched off). None is a REFUSAL and
    falls through to the cheaper tiers — the one thing it must never be is a "no".

    Every decision here is taken by the product's own functions (`unpack_boxes`,
    `best_animal`), so this is the glue that reads a row and not a third spelling of the
    rule; the rule itself is `junk.pet_label` below.
    """
    if not detector.enabled:
        return None
    row = conn.execute(
        "SELECT boxes FROM detections WHERE file_id = ? AND model = ?",
        (file_id, detector.model)).fetchone()
    if row is None:
        return None
    return best_animal(unpack_boxes(row["boxes"]), detector.threshold) is not None


class AnimalRuleCase(unittest.TestCase):
    """A database of frames written as a run leaves them, and the two spellings over it."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cfg = Config(sources=[self.root], database=self.root / "test.db", raw={})
        self.cfg.features = FeaturesConfig(pets=True, detector=True)
        self.cfg.detect = DetectConfig(enabled=True)
        self.conn = connect(self.cfg.database)
        self._n = 0

    def tearDown(self):
        self.conn.close()
        self.tmp.cleanup()

    # --- fixtures, each row exactly as the stage writes it ---------------------------

    def add_file(self, name: str) -> int:
        self._n += 1
        cur = self.conn.execute(
            """INSERT INTO files (path, size, mtime, ext, media_type, hash, hash_algo,
                   taken_at, taken_at_source, taken_at_confidence, indexed_at)
               VALUES (?, 10, 0, 'jpg', 'photo', ?, 'sha1', '2022-05-01T10:00:00',
                       'exif', 'high', '2026-01-01')""",
            (f"/photos/{self._n}_{name}", f"hash{self._n}"))
        self.conn.commit()
        return int(cur.lastrowid)

    def write_quality(self, file_id: int, *, score: float | None,
                      vlm: str | None = None, pet: str | None = None) -> None:
        self.conn.execute(
            """INSERT INTO frame_quality (file_id, sharpness, pet, pet_score, pet_vlm,
                   source, updated_at)
               VALUES (?, 100.0, ?, ?, ?, 'clip', '2026-01-01')""",
            (file_id, pet, score, vlm))
        self.conn.commit()

    def write_detection(self, file_id: int, boxes: list[Detection], *,
                        model: str | None = None, raw: str | None = None) -> None:
        """One `detections` row, written the way `junk._DetectorPass._examine` writes it.

        `label`/`score` are the best box AT THE THRESHOLD OF THAT RUN and `boxes` holds
        everything above the storage floor — the asymmetry the reader has to cope with,
        so it is reproduced rather than smoothed over. `raw` overrides the column outright
        and is how the corrupt-row cases get written.
        """
        kept = animal_boxes(boxes, STORE_FLOOR)
        best = best_animal(kept, self.cfg.features.detector_threshold)
        self.conn.execute(
            """INSERT INTO detections (file_id, label, score, boxes, model, updated_at)
               VALUES (?, ?, ?, ?, ?, '2026-01-01')""",
            (file_id, None if best is None else best.label,
             None if best is None else float(best.score),
             pack_boxes(kept) if raw is None else raw,
             model or self.cfg.detect.model))
        self.conn.commit()

    def mark_by_hand(self, file_id: int, is_animal: bool) -> None:
        self.conn.execute(
            "INSERT INTO manual_pet (file_id, is_animal, updated_at) VALUES (?, ?, 'x')",
            (file_id, 1 if is_animal else 0))
        self.conn.commit()

    # --- the two spellings ------------------------------------------------------------

    def thresholds(self, **kwargs: float) -> None:
        self.cfg.features = dataclasses.replace(self.cfg.features, **kwargs)

    def detector(self) -> DetectorSettings:
        return detector_settings(self.cfg)

    def sql_ids(self) -> list[int]:
        """The animal slice by the SQL spelling — the expression every reader uses."""
        return sorted(int(r[0]) for r in self.conn.execute(
            animal_ids_sql(self.cfg.features, self.detector())))

    def python_is_animal(self, file_id: int) -> bool:
        """The same verdict for one frame by the Python spelling (`junk.pet_label`).

        The manual mark is applied on top, in Python, exactly where `animal_ids_sql`
        applies its COALESCE: the user outranks every model (F124), and that order is not
        the detector's business.
        """
        manual = self.conn.execute(
            "SELECT is_animal FROM manual_pet WHERE file_id = ?", (file_id,)).fetchone()
        if manual is not None:
            return bool(manual["is_animal"])
        row = self.conn.execute(
            "SELECT pet_score, pet_vlm FROM frame_quality WHERE file_id = ?",
            (file_id,)).fetchone()
        if row is None:
            # No `frame_quality` row: there is nothing for the reader to correlate a
            # detection by, and the stage skips such a frame for the same reason (see
            # `junk._DetectorPass._relabel`). Not an animal, at any threshold.
            return False
        features = self.cfg.features
        return pet_label(
            row["pet_vlm"], row["pet_score"], features.pet_threshold,
            candidate_threshold=features.pet_candidate_threshold,
            detected=detector_answer(self.conn, file_id, self.detector())) is not None


class TestTheTwoSpellingsAreOneRule(AnimalRuleCase):
    """THE test of the feature: the full case table through both halves, row by row."""

    def build_table(self) -> dict[int, tuple]:
        """Every combination of score, answer, detection and manual mark — one file each."""
        cases = {}
        for score, answer, boxes_kind, mark in itertools.product(
                _SCORES, _ANSWERS, _CASES_BOXES, _MARKS):
            file_id = self.add_file(f"{score}_{answer}_{boxes_kind}_{mark}.jpg")
            self.write_quality(file_id, score=score, vlm=answer)
            boxes = _CASES_BOXES[boxes_kind]
            if boxes is not None:
                self.write_detection(file_id, boxes)
            if mark is not None:
                self.mark_by_hand(file_id, mark)
            cases[file_id] = (score, answer, boxes_kind, mark)
        return cases

    def test_every_row_of_the_table_gets_the_same_verdict_from_both(self):
        cases = self.build_table()
        for pet, candidate, detect_threshold in _THRESHOLD_GRID:
            self.thresholds(pet_threshold=pet, pet_candidate_threshold=candidate,
                            detector_threshold=detect_threshold)
            in_sql = set(self.sql_ids())
            for file_id, case in cases.items():
                with self.subTest(case=case, thresholds=(pet, candidate,
                                                         detect_threshold)):
                    self.assertEqual(file_id in in_sql,
                                     self.python_is_animal(file_id))

    def test_the_table_is_not_trivially_one_answer(self):
        """A parity test over a table nothing ever selects from would pass by accident."""
        self.build_table()
        selected = len(self.sql_ids())
        self.assertGreater(selected, 20)
        self.assertLess(selected, len(_SCORES) * len(_ANSWERS) * len(_CASES_BOXES)
                        * len(_MARKS) - 20)

    def test_they_agree_with_the_detector_switched_off_as_well(self):
        cases = self.build_table()
        self.cfg.detect = DetectConfig(enabled=False)
        in_sql = set(self.sql_ids())
        for file_id, case in cases.items():
            with self.subTest(case=case):
                self.assertEqual(file_id in in_sql, self.python_is_animal(file_id))

    def test_a_broken_boxes_column_costs_the_frame_its_answer_and_not_the_query(self):
        """`unpack_boxes` is lenient; the SQL has to be lenient in the same places.

        A column that is not JSON at all would make `json_extract` raise and take every
        animal query down with it — the slice, the counter and the tab at once.
        """
        for raw in ('{', '', 'not json', '[42]', '[["cat","x",1,2,3,4]]', '[[]]'):
            with self.subTest(stored=raw):
                file_id = self.add_file("broken.jpg")
                self.write_quality(file_id, score=0.95)
                self.write_detection(file_id, [], raw=raw)
                self.assertEqual(file_id in set(self.sql_ids()),
                                 self.python_is_animal(file_id))


class TestTheDetectorReachesTheSlice(AnimalRuleCase):
    """Brief tests 2-5: what the tier does, stated one frame at a time."""

    def frame(self, name: str, *, score: float | None, vlm: str | None = None,
              boxes: list[Detection] | None = None, model: str | None = None) -> int:
        file_id = self.add_file(name)
        self.write_quality(file_id, score=score, vlm=vlm)
        if boxes is not None:
            self.write_detection(file_id, boxes, model=model)
        return file_id

    def test_an_animal_found_below_the_clip_threshold_is_in_the_slice(self):
        """The recall half: 87% against the CLIP label's 33% is made of these frames."""
        fid = self.frame("dark_cat.jpg", score=0.10, boxes=_CASES_BOXES["strong"])
        self.assertEqual(self.sql_ids(), [fid])

    def test_a_confident_clip_frame_with_nothing_detected_leaves_the_slice(self):
        """The precision half, and the direction that costs a user something — so it is
        checked that the frame WAS in the slice before the detector had its say."""
        fid = self.frame("fur_coat.jpg", score=0.95, boxes=_CASES_BOXES["empty"])
        self.assertEqual(self.sql_ids(), [])
        self.cfg.detect = DetectConfig(enabled=False)
        self.assertEqual(self.sql_ids(), [fid])

    def test_a_frame_the_detector_never_examined_keeps_the_previous_verdict(self):
        """A refusal is never read as "no animal" — the rule the whole stage is built on:
        below the candidate depth, an error on the frame, the model unavailable."""
        above = self.frame("cat.jpg", score=0.95)
        self.frame("desk.jpg", score=0.10)
        self.assertEqual(self.sql_ids(), [above])

    def test_boxes_from_another_detector_are_not_this_one_s_answer(self):
        fid = self.frame("cat.jpg", score=0.95, boxes=_CASES_BOXES["empty"],
                         model=_OTHER_MODEL)
        self.assertEqual(self.sql_ids(), [fid])

    def test_a_box_below_the_confidence_threshold_is_not_an_animal(self):
        self.frame("blur.jpg", score=0.95, boxes=_CASES_BOXES["weak"])
        self.assertEqual(self.sql_ids(), [])

    def test_the_threshold_is_re_chosen_off_the_stored_boxes_without_a_run(self):
        """F137's property, applied to `features.detector_threshold`: the boxes are stored
        with their scores precisely so that this needs no new pass over any image — and
        that includes LOWERING it under the value the run happened to store."""
        fid = self.frame("blur.jpg", score=0.10, boxes=_CASES_BOXES["weak"])
        self.assertEqual(self.sql_ids(), [])
        self.thresholds(detector_threshold=0.20)
        self.assertEqual(self.sql_ids(), [fid])
        self.thresholds(detector_threshold=0.95)
        self.assertEqual(self.sql_ids(), [])

    def test_the_best_box_of_a_crowd_decides(self):
        fid = self.frame("dog_and_cat.jpg", score=0.10, boxes=_CASES_BOXES["crowd"])
        self.assertEqual(self.sql_ids(), [fid])

    def test_a_person_box_never_makes_an_animal(self):
        """The boundary the measurement drew: 42% precision on people against ~100% from
        the face boxes (F152). The people slice is not this feature's, in either half."""
        self.frame("crowd.jpg", score=0.95, boxes=_CASES_BOXES["person"])
        self.assertEqual(self.sql_ids(), [])

    def test_the_vlm_answer_still_outranks_the_detector(self):
        """A box detector calls a drawn cat a cat, which is the error F130 exists to
        remove — so a frame that check has answered about keeps its answer, both ways."""
        toy = self.frame("toy.jpg", score=0.95, vlm=PET_VLM_DEPICTION,
                         boxes=_CASES_BOXES["strong"])
        cat = self.frame("cat.jpg", score=0.95, vlm=PET_VLM_REAL,
                         boxes=_CASES_BOXES["empty"])
        self.assertEqual(self.sql_ids(), [cat])
        self.assertNotIn(toy, self.sql_ids())

    def test_the_manual_mark_outranks_the_detector_too(self):
        """F124 is untouched: the person looked at the frame, and so did the model."""
        found = self.frame("cat.jpg", score=0.10, boxes=_CASES_BOXES["strong"])
        missed = self.frame("kitten.jpg", score=0.95, boxes=_CASES_BOXES["empty"])
        self.mark_by_hand(found, is_animal=False)
        self.mark_by_hand(missed, is_animal=True)
        self.assertEqual(self.sql_ids(), [missed])

    def test_a_detection_without_a_quality_row_does_not_invent_a_verdict(self):
        """The stage skips such a frame as well (`_relabel`): `frame_quality`'s population
        is written by the quality half under its own incrementality, and a box is not a
        reason to give a frame a row with no sharpness in it."""
        file_id = self.add_file("orphan.jpg")
        self.write_detection(file_id, _CASES_BOXES["strong"])
        self.assertEqual(self.sql_ids(), [])


class TestTheSwitchedOffDetectorChangesNothing(AnimalRuleCase):
    """Brief test 6 / F145: a tier nobody switched on decides nothing, at all, ever."""

    def stock(self) -> list[int]:
        """A frame of every kind, each with boxes that would move it if they counted."""
        ids = []
        for name, score, vlm, kind in (
                ("cat.jpg", 0.95, None, "empty"),
                ("dark_cat.jpg", 0.10, None, "strong"),
                ("toy.jpg", 0.95, PET_VLM_DEPICTION, "strong"),
                ("coat.jpg", 0.35, None, "person")):
            file_id = self.add_file(name)
            self.write_quality(file_id, score=score, vlm=vlm)
            self.write_detection(file_id, _CASES_BOXES[kind])
            ids.append(file_id)
        return ids

    def test_the_expression_is_byte_for_byte_the_one_without_the_tier(self):
        """Not "a branch that happens to be false" — the same string. A switched-off
        feature that still costs a subquery per row is a feature that can still be wrong.
        """
        without = animal_auto_sql(self.cfg.features)
        for master, feature in ((False, True), (True, False), (False, False)):
            with self.subTest(master=master, feature=feature):
                self.cfg.detect = DetectConfig(enabled=master)
                self.thresholds(detector=feature)
                self.assertEqual(
                    animal_auto_sql(self.cfg.features, detector=self.detector()),
                    without)
                self.assertNotIn("detections",
                                 animal_ids_sql(self.cfg.features, self.detector()))

    def test_no_verdict_moves_when_either_switch_is_off(self):
        self.stock()
        self.cfg.detect = DetectConfig(enabled=False)
        self.thresholds(detector=False)
        expected = self.sql_ids()
        for master, feature in ((False, True), (True, False), (False, False)):
            with self.subTest(master=master, feature=feature):
                self.cfg.detect = DetectConfig(enabled=master)
                self.thresholds(detector=feature)
                self.assertEqual(self.sql_ids(), expected)

    def test_switching_it_on_is_what_moves_them(self):
        """The other half of the same claim: if nothing moved either way, the test above
        would be passing on a slice the detector could never reach."""
        self.stock()
        self.cfg.detect = DetectConfig(enabled=False)
        off = self.sql_ids()
        self.cfg.detect = DetectConfig(enabled=True)
        self.thresholds(detector=True)
        self.assertNotEqual(self.sql_ids(), off)


class TestOneNumberInEveryConsumer(AnimalsTestBase):
    """Brief test 7: the counter, the tab and the album over one collection, one number.

    The web app reads the LIVE config per request, so the switch is flipped between two
    reads with nothing re-run in between — the same way F137 moves a threshold.
    """

    def setUp(self):
        super().setUp()
        self.cfg.features = dataclasses.replace(
            self.cfg.features, pets=True, detector=True)
        self.cfg.detect = DetectConfig(enabled=True)

    def store(self, name: str, *, score: float | None, vlm: str | None = None,
              boxes: list[Detection] | None = None) -> int:
        file_id, _path, _content = self.add_photo_file(name)
        self.conn.execute(
            """INSERT INTO frame_quality (file_id, sharpness, pet, pet_score, pet_vlm,
                   source, updated_at)
               VALUES (?, 100.0, NULL, ?, ?, 'clip', '2026-01-01')""",
            (file_id, score, vlm))
        if boxes is not None:
            kept = animal_boxes(boxes, STORE_FLOOR)
            best = best_animal(kept, self.cfg.features.detector_threshold)
            self.conn.execute(
                """INSERT INTO detections (file_id, label, score, boxes, model,
                       updated_at)
                   VALUES (?, ?, ?, ?, ?, '2026-01-01')""",
                (file_id, None if best is None else best.label,
                 None if best is None else float(best.score),
                 pack_boxes(kept), self.cfg.detect.model))
        self.conn.commit()
        return file_id

    def collection(self) -> dict[str, int]:
        return {
            # CLIP says no, the detector finds a cat -> in the slice once it is on
            "dark_cat.jpg": self.store("dark_cat.jpg", score=0.10,
                                       boxes=_CASES_BOXES["strong"]),
            # CLIP says yes, the detector finds nothing -> out of the slice once it is on
            "fur_coat.jpg": self.store("fur_coat.jpg", score=0.95,
                                       boxes=_CASES_BOXES["empty"]),
            # never examined: the CLIP verdict stands either way
            "cat.jpg": self.store("cat.jpg", score=0.90),
            "desk.jpg": self.store("desk.jpg", score=0.05),
        }

    def album_ids(self) -> list[int]:
        with redirect_stdout(io.StringIO()):
            report = plan_album(self.cfg, self.conn, "animal", "", self.root / "album")
        return sorted(it.file_id for it in report.plan)

    def tab_animal_ids(self) -> list[int]:
        return sorted(it["file_id"] for it in self.animals()["items"] if it["is_animal"])

    def overview_animals(self) -> int:
        _status, body, _ctype = self.get("/api/overview")
        return json.loads(body)["collection"]["animals"]

    def assert_all_agree(self, expected: list[int]) -> None:
        self.assertEqual(self.album_ids(), expected)
        self.assertEqual(self.tab_animal_ids(), expected)
        self.assertEqual(self.animals()["animals"], len(expected))
        self.assertEqual(self.overview_animals(), len(expected))

    def test_the_answer_reaches_the_counter_the_tab_and_the_album_at_once(self):
        ids = self.collection()
        self.start_server()
        self.assert_all_agree(sorted([ids["dark_cat.jpg"], ids["cat.jpg"]]))

    def test_and_they_all_go_back_together_when_it_is_switched_off(self):
        ids = self.collection()
        self.start_server()
        self.cfg.detect = DetectConfig(enabled=False)
        self.assert_all_agree(sorted([ids["fur_coat.jpg"], ids["cat.jpg"]]))

    def test_the_tab_lists_the_frame_the_detector_found(self):
        """The page and its counter, not only the number: a card the tier has withdrawn
        must not sit there claiming to be an animal, and one it found has to appear."""
        ids = self.collection()
        self.start_server()
        listed = sorted(it["file_id"] for it in self.animals()["items"])
        self.assertIn(ids["dark_cat.jpg"], listed)
        self.assertNotIn(ids["fur_coat.jpg"], listed)

    def test_a_manual_mark_still_wins_through_the_web_app(self):
        ids = self.collection()
        self.start_server()
        self.conn.execute(
            "INSERT INTO manual_pet (file_id, is_animal, updated_at) VALUES (?, 1, 'x')",
            (ids["fur_coat.jpg"],))
        self.conn.commit()
        self.assert_all_agree(
            sorted([ids["dark_cat.jpg"], ids["fur_coat.jpg"], ids["cat.jpg"]]))


class TestTheCaptionNamesTheMeasurement(unittest.TestCase):
    """Brief test 8: the slice promises what it was measured at, in all three languages.

    F158 made this a rule: a caption that keeps an old precision while the rule under it
    buys recall is spending the reader's trust. The detector is 62% precision at 87%
    recall against the cascade's 82% / 64% — a real trade in both directions, chosen by a
    switch, so both numbers have to be on the line.
    """

    def test_both_measurements_are_named_in_every_language(self):
        for lang in ("ru", "en", "ja"):
            with self.subTest(lang=lang):
                intro = ui._UI_STRINGS["animals_intro"][lang]
                for number in ("82", "64", "62", "87"):
                    self.assertIn(number, intro)
                self.assertNotIn("92", intro)

    def test_the_caption_says_which_switch_chooses_between_them(self):
        """A number a reader cannot tell applies to them is not a promise but a footnote."""
        for lang in ("ru", "en", "ja"):
            with self.subTest(lang=lang):
                self.assertIn("features.detector", ui._UI_STRINGS["animals_intro"][lang])


if __name__ == "__main__":
    unittest.main()
