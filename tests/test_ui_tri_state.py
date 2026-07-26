"""F82: one folder tree with three states in the web app.

Before this, the tree could only say "do not scan" — the files never enter the index —
while the other, much softer meaning ("already sorted by hand: index it, just do not lay
it out") existed only in config.yaml and on the command line. A live user picked the
only tool the interface offered and lost his manually sorted trips from search,
statistics and the whole app. Both meanings now sit on the same node, so the tests here
care above all that the two never get confused: that the file written by F81 keeps
meaning "do not scan", that a folder cannot end up in both sections, and that the
summary reports the two numbers apart.

The endpoint still only reads metadata and writes one settings file; every path from the
client is validated before it is used.
"""
from __future__ import annotations

import unittest
import urllib.parse

import yaml

from tests.test_ui_source_tree import SourceTreeTestBase

from sorta import ui
from sorta.indexer import load_excludes, save_excludes


def _js_function(html: str, name: str) -> str:
    """The body of one JS function of the page — the tree lives in the template, and
    asserting on the rendered source is how the other UI tests reach it."""
    start = html.index("function " + name + "(")
    depth, i = 0, html.index("{", start)
    for j in range(i, len(html)):
        if html[j] == "{":
            depth += 1
        elif html[j] == "}":
            depth -= 1
            if depth == 0:
                return html[start:j + 1]
    raise AssertionError("не найдено тело " + name)


class TestOldFileStillReadsAsDoNotScan(SourceTreeTestBase):
    """§1: the user already has an `excludes.yaml` in the F81 shape. Losing it is the
    one failure this feature cannot afford."""

    def write_old_format(self, *rels: str) -> None:
        self.excludes_file.write_text(
            yaml.safe_dump({self.src_dir.resolve().as_posix(): list(rels)},
                           allow_unicode=True),
            encoding="utf-8")

    def test_a_flat_list_under_a_root_is_skip_scan(self):
        self.write_old_format("Movies", "DCIM3/Screenshots")

        loaded = load_excludes(self.excludes_file)

        self.assertEqual(loaded.for_root(self.src_dir), {"Movies", "DCIM3/Screenshots"})
        self.assertEqual(loaded.layout_for_root(self.src_dir), frozenset())

    def test_the_tree_shows_the_old_entries_as_do_not_scan(self):
        self.make_file("Movies/a.mp4")
        self.write_old_format("Movies")
        self.start_server()

        _status, data = self.get_tree(self.src_dir)

        self.assertEqual(data["skip_scan"], ["Movies"])
        self.assertEqual(data["skip_layout"], [])

    def test_rewriting_such_a_file_keeps_the_entries(self):
        self.make_file("Movies/a.mp4")
        self.make_file("Греция/b.jpg")
        self.write_old_format("Movies")
        self.start_server()

        # the user adds a "do not lay out" folder and touches nothing else
        status, data = self.post("/api/source-tree/excludes", {
            "root": str(self.src_dir), "skip_scan": ["Movies"],
            "skip_layout": ["Греция"]})

        self.assertEqual(status, 200)
        loaded = load_excludes(self.excludes_file)
        self.assertEqual(loaded.for_root(self.src_dir), {"Movies"})
        self.assertEqual(loaded.layout_for_root(self.src_dir), {"Греция"})
        self.assertEqual(data["skip_scan"], ["Movies"])
        self.assertEqual(data["skip_layout"], ["Греция"])

    def test_writing_only_the_scan_section_keeps_the_layout_one(self):
        # `sorta index --exclude-dir` passes no layout list at all: it has an opinion
        # about scanning and none about the layout, so it must leave that half alone.
        save_excludes(self.excludes_file, self.src_dir, ["Movies"], ["Греция"])

        save_excludes(self.excludes_file, self.src_dir, ["Movies", "Screenshots"])

        loaded = load_excludes(self.excludes_file)
        self.assertEqual(loaded.for_root(self.src_dir), {"Movies", "Screenshots"})
        self.assertEqual(loaded.layout_for_root(self.src_dir), {"Греция"})

    def test_a_root_with_only_the_old_key_is_untouched_by_another_root(self):
        other = self.root / "archive"
        (other / "temp").mkdir(parents=True)
        self.write_old_format("Movies")
        self.start_server()

        self.post("/api/source-tree/excludes",
                  {"root": str(other), "skip_layout": ["temp"]})

        loaded = load_excludes(self.excludes_file)
        self.assertEqual(loaded.for_root(self.src_dir), {"Movies"})
        self.assertEqual(loaded.layout_for_root(other), {"temp"})


class TestSavingBothSections(SourceTreeTestBase):
    def test_the_file_gets_both_sections_keyed_by_root(self):
        self.make_file("Movies/a.mp4", size=40)
        self.make_file("Греция/b.jpg", size=2)
        self.start_server()

        status, _data = self.post("/api/source-tree/excludes", {
            "root": str(self.src_dir), "skip_scan": ["Movies"],
            "skip_layout": ["Греция"]})

        self.assertEqual(status, 200)
        raw = yaml.safe_load(self.excludes_file.read_text(encoding="utf-8"))
        self.assertEqual(raw, {self.src_dir.resolve().as_posix():
                               {"skip_scan": ["Movies"], "skip_layout": ["Греция"]}})

    def test_other_roots_are_not_touched(self):
        other = self.root / "archive"
        (other / "temp").mkdir(parents=True)
        self.make_file("Movies/a.mp4")
        self.make_file("Греция/b.jpg")
        self.start_server()

        self.post("/api/source-tree/excludes",
                  {"root": str(other), "skip_scan": ["temp"]})
        self.post("/api/source-tree/excludes", {
            "root": str(self.src_dir), "skip_scan": ["Movies"],
            "skip_layout": ["Греция"]})

        loaded = load_excludes(self.excludes_file)
        self.assertEqual(loaded.for_root(other), {"temp"})
        self.assertEqual(loaded.layout_for_root(other), frozenset())
        self.assertEqual(loaded.for_root(self.src_dir), {"Movies"})
        self.assertEqual(loaded.layout_for_root(self.src_dir), {"Греция"})

    def test_an_empty_pair_drops_the_root_key(self):
        self.make_file("Movies/a.mp4")
        self.start_server()

        self.post("/api/source-tree/excludes",
                  {"root": str(self.src_dir), "skip_scan": ["Movies"]})
        self.post("/api/source-tree/excludes",
                  {"root": str(self.src_dir), "skip_scan": [], "skip_layout": []})

        self.assertFalse(load_excludes(self.excludes_file))
        self.assertNotIn("Movies", self.excludes_file.read_text(encoding="utf-8"))

    def test_a_missing_section_means_an_empty_one(self):
        self.make_file("Греция/b.jpg")
        self.start_server()

        status, data = self.post("/api/source-tree/excludes",
                                 {"root": str(self.src_dir), "skip_layout": ["Греция"]})

        self.assertEqual(status, 200)
        self.assertEqual((data["skip_scan"], data["skip_layout"]), ([], ["Греция"]))


class TestTheTwoStatesExcludeEachOther(SourceTreeTestBase):
    """§2: a folder is either not scanned or not laid out — never both."""

    def test_marking_do_not_scan_clears_do_not_lay_out(self):
        self.make_file("Греция/b.jpg")
        self.start_server()
        self.post("/api/source-tree/excludes",
                  {"root": str(self.src_dir), "skip_layout": ["Греция"]})

        status, data = self.post("/api/source-tree/excludes",
                                 {"root": str(self.src_dir), "skip_scan": ["Греция"]})

        self.assertEqual(status, 200)
        self.assertEqual((data["skip_scan"], data["skip_layout"]), (["Греция"], []))
        loaded = load_excludes(self.excludes_file)
        self.assertEqual(loaded.for_root(self.src_dir), {"Греция"})
        self.assertEqual(loaded.layout_for_root(self.src_dir), frozenset())

    def test_marking_do_not_lay_out_clears_do_not_scan(self):
        self.make_file("Греция/b.jpg")
        self.start_server()
        self.post("/api/source-tree/excludes",
                  {"root": str(self.src_dir), "skip_scan": ["Греция"]})

        status, data = self.post("/api/source-tree/excludes",
                                 {"root": str(self.src_dir), "skip_layout": ["Греция"]})

        self.assertEqual(status, 200)
        self.assertEqual((data["skip_scan"], data["skip_layout"]), ([], ["Греция"]))
        loaded = load_excludes(self.excludes_file)
        self.assertEqual(loaded.for_root(self.src_dir), frozenset())
        self.assertEqual(loaded.layout_for_root(self.src_dir), {"Греция"})

    def test_a_hand_written_overlap_resolves_to_do_not_scan(self):
        self.excludes_file.write_text(
            yaml.safe_dump({self.src_dir.resolve().as_posix():
                            {"skip_scan": ["Греция"], "skip_layout": ["Греция"]}},
                           allow_unicode=True),
            encoding="utf-8")

        loaded = load_excludes(self.excludes_file)

        self.assertEqual(loaded.for_root(self.src_dir), {"Греция"})
        self.assertEqual(loaded.layout_for_root(self.src_dir), frozenset())

    def test_one_node_carries_one_state_in_the_tree(self):
        html = ui._render_index_html("ru")
        body = _js_function(html, "collectExcludes")
        # a single data-state attribute per node -> "ticked in both" is not expressible
        self.assertIn('var state = marks[i].getAttribute("data-state");', body)
        self.assertIn('if (state === "scan") result.skip_scan.push', body)
        self.assertIn('else if (state === "layout") result.skip_layout.push', body)
        self.assertIn('var TRI_STATES = ["", "layout", "scan"];', html)


class TestSummaryShowsBothNumbers(SourceTreeTestBase):
    def test_the_endpoint_reports_the_two_counts_apart(self):
        self.make_file("Movies/a.mp4", size=64)
        self.make_file("Screenshots/s.png", size=8)
        self.make_file("Греция/b.jpg", size=999)
        self.make_file("Франция/c.jpg", size=999)
        self.start_server()
        self.post("/api/source-tree/excludes", {
            "root": str(self.src_dir), "skip_scan": ["Movies", "Screenshots"],
            "skip_layout": ["Греция", "Франция"]})

        status, data = self.get("/api/source-tree/excludes?path="
                                + urllib.parse.quote(str(self.src_dir)))

        self.assertEqual(status, 200)
        self.assertEqual((data["count"], data["files"], data["size"]), (2, 2, 72))
        self.assertEqual(data["layout_count"], 2)
        # the size belongs to "do not scan" only: a "do not lay out" folder is still
        # read and indexed, so counting its bytes as saved would be a lie
        self.assertEqual(data["skip_layout"], ["Греция", "Франция"])

    def test_the_collapsed_line_prints_both(self):
        html = ui._render_index_html("ru")
        body = _js_function(html, "excludesSummaryText")
        self.assertIn("I18N.excludes_summary,", body)
        self.assertIn("I18N.excludes_summary_layout", body)
        self.assertIn('parts.join(" · ")', body)
        self.assertIn("I18N.excludes_summary_none", body)

    def test_both_summary_strings_exist_in_three_languages(self):
        for key in ("excludes_summary", "excludes_summary_layout",
                    "excludes_summary_none", "excludes_button", "excludes_title",
                    "excludes_hint", "excludes_saved",
                    "tri_none_label", "tri_none_hint",
                    "tri_layout_label", "tri_layout_hint",
                    "tri_scan_label", "tri_scan_hint"):
            with self.subTest(key=key):
                entry = ui._UI_STRINGS[key]
                self.assertEqual(set(entry), {"ru", "en", "ja"})
                self.assertTrue(all(entry.values()))

    def test_the_two_states_are_named_the_way_the_brief_names_them(self):
        self.assertEqual(ui._t("tri_scan_label", "ru"), "не сканировать")
        self.assertEqual(ui._t("tri_layout_label", "ru"), "не раскладывать")
        # each has its own one-line explanation — the difference is what a live user
        # got wrong, so it is spelled out rather than implied
        self.assertIn("индекс", ui._t("tri_scan_hint", "ru"))
        self.assertIn("остаются в индексе", ui._t("tri_layout_hint", "ru"))


class TestTheStateIsVisibleAtAGlance(unittest.TestCase):
    """§3: three states on a node, readable without hovering."""

    def setUp(self):
        self.html = ui._render_index_html("ru")

    def test_the_mark_carries_a_glyph_and_a_caption(self):
        body = _js_function(self.html, "triText")
        self.assertIn('"☒ " + I18N.tri_scan_label', body)
        self.assertIn('"◐ " + I18N.tri_layout_label', body)
        self.assertIn('return "☐";', body)

    def test_colour_is_not_the_only_signal(self):
        # both states are styled, but each also spells its name out in the button
        self.assertIn('.tri-state[data-state="layout"]', self.html)
        self.assertIn('.tri-state[data-state="scan"]', self.html)

    def test_the_panel_explains_all_three_states(self):
        legend = self.html.split('id="excludes-legend"')[1].split("</ul>")[0]
        for key in ("tri_none_label", "tri_layout_label", "tri_scan_label",
                    "tri_none_hint", "tri_layout_hint", "tri_scan_hint"):
            with self.subTest(key=key):
                self.assertIn(ui._t(key, "ru"), legend)

    def test_a_parent_sets_the_state_of_its_subtree(self):
        body = _js_function(self.html, "setSubtreeState")
        self.assertIn("setTriState(marks[i], state);", body)
        self.assertIn("marks[i].disabled = !!state;", body)
        self.assertIn("if (ul) setSubtreeState(ul, next);", self.html)

    def test_saved_states_come_back_onto_the_tree(self):
        body = _js_function(self.html, "renderExcludesTree")
        self.assertIn('(data.skip_layout || []).forEach', body)
        self.assertIn('(data.skip_scan || []).forEach', body)

    def test_the_save_posts_both_sections(self):
        self.assertIn("{ root: src, skip_scan: picked.skip_scan, "
                      "skip_layout: picked.skip_layout }", self.html)


class TestClientPathsAreStillValidated(SourceTreeTestBase):
    """§1: the values come from outside — an exclusion may narrow the walk, never
    move it. Same rule as F81, now for both sections."""

    def test_an_arbitrary_root_is_rejected(self):
        self.start_server()

        status, data = self.post("/api/source-tree/excludes",
                                 {"root": "../elsewhere", "skip_layout": ["Греция"]})

        self.assertEqual(status, 400)
        self.assertIn("error", data)
        self.assertFalse(self.excludes_file.exists())

    def test_escaping_entries_are_refused_in_both_sections(self):
        self.make_file("Movies/a.mp4")
        self.make_file("Греция/b.jpg")
        self.start_server()

        status, data = self.post("/api/source-tree/excludes", {
            "root": str(self.src_dir),
            "skip_scan": ["Movies", "../outside", "C:/windows"],
            "skip_layout": ["Греция", "/etc", "..\\windows"],
        })

        self.assertEqual(status, 200)
        self.assertEqual(data["skip_scan"], ["Movies"])
        self.assertEqual(data["skip_layout"], ["Греция"])
        self.assertEqual(sorted(data["rejected"]),
                         ["../outside", "..\\windows", "/etc", "C:/windows"])

    def test_a_section_that_is_not_a_list_is_a_bad_request(self):
        self.start_server()

        for payload in ({"root": None, "skip_layout": ["Греция"]},
                        {"root": str(self.src_dir), "skip_layout": "Греция"},
                        {"root": str(self.src_dir), "skip_scan": {"a": 1}}):
            with self.subTest(payload=payload):
                status, _data = self.post("/api/source-tree/excludes", payload)
                self.assertEqual(status, 400)
        self.assertFalse(self.excludes_file.exists())


if __name__ == "__main__":
    unittest.main()
