"""F81: the source folder tree in the web app — "do not scan" chosen before indexing.

The endpoint only ever reads metadata (folders, counts, sizes) and writes one settings
file; nothing here touches a photo or the index. Everything the client sends — the
root and every list entry — is validated before it is used.
"""
from __future__ import annotations

import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import yaml

from sorta import ui
from sorta.config import Config
from sorta.db import connect
from sorta.indexer import load_excludes


class SourceTreeTestBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.src_dir = self.root / "photos"
        self.src_dir.mkdir()
        self.excludes_file = self.root / "excludes.yaml"
        self.cfg = Config(sources=[self.src_dir], database=self.root / "test.db",
                          raw={"index": {"excludes_file": str(self.excludes_file)}})
        self.conn = connect(self.cfg.database)
        self.server = None
        self.thread = None
        self.base_url = None

    def tearDown(self):
        if self.server is not None:
            self.server.shutdown()
            self.thread.join(timeout=5)
            self.server.server_close()
        self.conn.close()
        self.tmp.cleanup()

    def start_server(self) -> None:
        self.server = ui.build_server(self.cfg, self.conn, port=0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"

    def make_file(self, rel: str, size: int = 16) -> Path:
        p = self.src_dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x" * size)
        return p

    def get(self, path: str) -> tuple[int, object]:
        try:
            with urllib.request.urlopen(f"{self.base_url}{path}", timeout=5) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())

    def get_tree(self, root: Path | str) -> tuple[int, object]:
        return self.get("/api/source-tree?path=" + urllib.parse.quote(str(root)))

    def post(self, path: str, payload: dict) -> tuple[int, object]:
        req = urllib.request.Request(
            f"{self.base_url}{path}", data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())


class TestSourceTreeEndpoint(SourceTreeTestBase):
    def test_returns_folders_with_counts_and_sizes_and_no_files(self):
        self.make_file("Movies/a.mp4", size=100)
        self.make_file("Movies/deep/b.mp4", size=50)
        self.make_file("Trip/c.jpg", size=10)
        self.make_file("top.jpg", size=7)
        self.start_server()

        status, data = self.get_tree(self.src_dir)

        self.assertEqual(status, 200)
        tree = data["tree"]
        self.assertEqual(tree["files"], 4)          # the whole subtree
        self.assertEqual(tree["size"], 167)
        names = {c["name"]: c for c in tree["children"]}
        self.assertEqual(set(names), {"Movies", "Trip"})  # folders only, no top.jpg
        self.assertEqual((names["Movies"]["files"], names["Movies"]["size"]), (2, 150))
        self.assertEqual(names["Movies"]["rel"], "Movies")
        self.assertEqual([c["name"] for c in names["Movies"]["children"]], ["deep"])
        self.assertFalse(data["truncated"])
        # not a single file name reaches the response
        self.assertNotIn("a.mp4", json.dumps(data))
        self.assertNotIn("top.jpg", json.dumps(data))

    def test_saved_excludes_come_back_with_the_tree(self):
        self.make_file("Movies/a.mp4")
        self.excludes_file.write_text(
            yaml.safe_dump({self.src_dir.resolve().as_posix(): {"skip_scan": ["Movies"]}}),
            encoding="utf-8")
        self.start_server()

        _status, data = self.get_tree(self.src_dir)

        self.assertEqual(data["skip_scan"], ["Movies"])

    def test_response_size_is_limited_and_says_so(self):
        for i in range(12):
            self.make_file(f"dir{i:02d}/sub/file.jpg", size=3)
        self.start_server()
        root = self.src_dir.resolve()

        payload = ui._source_tree_payload(root, [], [], max_nodes=5, max_depth=3)

        emitted = self._count_nodes(payload["tree"])
        self.assertLessEqual(emitted, 6)  # 5 folders + the root node
        self.assertTrue(payload["truncated"], "обрезка не сообщена в ответе")
        # totals stay honest even for the part that did not fit
        self.assertEqual(payload["tree"]["files"], 12)
        self.assertEqual(payload["tree"]["size"], 36)

    def test_depth_limit_stops_the_nesting_but_keeps_the_totals(self):
        self.make_file("a/b/c/d/deep.jpg", size=5)
        root = self.src_dir.resolve()

        payload = ui._source_tree_payload(root, [], [], max_nodes=100, max_depth=2)

        node = payload["tree"]["children"][0]           # a
        self.assertEqual(node["name"], "a")
        self.assertEqual([c["name"] for c in node["children"]], ["b"])
        self.assertEqual(node["children"][0]["children"], [])  # c is over the depth limit
        self.assertTrue(payload["truncated"])
        self.assertEqual(payload["tree"]["files"], 1)
        self.assertEqual(payload["tree"]["size"], 5)

    def _count_nodes(self, node: dict) -> int:
        return 1 + sum(self._count_nodes(c) for c in node["children"])


class TestRootValidation(SourceTreeTestBase):
    def test_arbitrary_paths_are_rejected(self):
        self.make_file("Movies/a.mp4")
        self.start_server()
        missing = self.root / "does-not-exist"
        a_file = self.make_file("Trip/c.jpg")
        for raw in ("", "   ", "../..", "photos", str(missing), str(a_file)):
            with self.subTest(path=raw):
                status, data = self.get_tree(raw)
                self.assertEqual(status, 400)
                self.assertIn("error", data)
                self.assertNotIn("tree", data)

    def test_saving_with_an_arbitrary_root_is_rejected(self):
        self.start_server()
        status, data = self.post("/api/source-tree/excludes",
                                 {"root": "../elsewhere", "skip_scan": ["Movies"]})
        self.assertEqual(status, 400)
        self.assertIn("error", data)
        self.assertFalse(self.excludes_file.exists())

    def test_invalid_bodies_are_rejected(self):
        self.start_server()
        for payload in ({}, {"root": ""}, {"root": str(self.src_dir), "skip_scan": "Movies"},
                        {"skip_scan": ["Movies"]}):
            with self.subTest(payload=payload):
                status, _data = self.post("/api/source-tree/excludes", payload)
                self.assertEqual(status, 400)
        self.assertFalse(self.excludes_file.exists())

    def test_escaping_entries_are_refused_and_reported(self):
        self.make_file("Movies/a.mp4")
        self.start_server()

        status, data = self.post("/api/source-tree/excludes", {
            "root": str(self.src_dir),
            "skip_scan": ["Movies", "../outside", "/etc", "C:/windows"],
        })

        self.assertEqual(status, 200)
        self.assertEqual(data["skip_scan"], ["Movies"])
        self.assertEqual(sorted(data["rejected"]), ["../outside", "/etc", "C:/windows"])
        self.assertEqual(
            load_excludes(self.excludes_file).for_root(self.src_dir), {"Movies"})


class TestSavingExcludes(SourceTreeTestBase):
    def test_saved_file_is_yaml_keyed_by_the_chosen_root(self):
        self.make_file("Movies/a.mp4", size=40)
        self.make_file("Screenshots/s.png", size=2)
        self.start_server()

        status, data = self.post("/api/source-tree/excludes", {
            "root": str(self.src_dir), "skip_scan": ["Movies", "Screenshots"]})

        self.assertEqual(status, 200)
        raw = yaml.safe_load(self.excludes_file.read_text(encoding="utf-8"))
        self.assertEqual(raw, {self.src_dir.resolve().as_posix():
                               {"skip_scan": ["Movies", "Screenshots"]}})
        self.assertEqual((data["count"], data["files"], data["size"]), (2, 2, 42))

    def test_resaving_one_root_keeps_the_other_roots(self):
        other = self.root / "archive"
        (other / "temp").mkdir(parents=True)
        self.make_file("Movies/a.mp4")
        self.start_server()

        self.post("/api/source-tree/excludes",
                  {"root": str(self.src_dir), "skip_scan": ["Movies"]})
        self.post("/api/source-tree/excludes", {"root": str(other), "skip_scan": ["temp"]})
        self.post("/api/source-tree/excludes",
                  {"root": str(self.src_dir), "skip_scan": []})

        loaded = load_excludes(self.excludes_file)
        self.assertEqual(loaded.for_root(self.src_dir), frozenset())
        self.assertEqual(loaded.for_root(other), {"temp"})

    def test_get_reports_what_is_currently_not_scanned(self):
        self.make_file("Movies/a.mp4", size=64)
        self.make_file("keep.jpg", size=8)
        self.start_server()
        self.post("/api/source-tree/excludes",
                  {"root": str(self.src_dir), "skip_scan": ["Movies"]})

        status, data = self.get(
            "/api/source-tree/excludes?path=" + urllib.parse.quote(str(self.src_dir)))

        self.assertEqual(status, 200)
        self.assertEqual(data["skip_scan"], ["Movies"])
        self.assertEqual((data["count"], data["files"], data["size"]), (1, 1, 64))


class TestFirstTabLayout(unittest.TestCase):
    """§5: three blocks, a configured one collapses to a line, nothing is locked."""

    def setUp(self):
        self.html = ui._render_index_html("ru")

    def test_three_blocks_exist(self):
        for block in ("step-source", "step-options", "step-actions"):
            self.assertIn(f'id="{block}"', self.html)

    def test_configured_block_collapses_to_one_line(self):
        self.assertIn("function updateStepLayout()", self.html)
        self.assertIn('updateStepToggle("step-source", "step-source-edit", '
                      'stepSourceOpen, !!src)', self.html)
        self.assertIn('updateStepToggle("step-options", "step-options-edit", '
                      'stepOptionsOpen, !!src)', self.html)
        self.assertIn('id="step-source-summary"', self.html)
        self.assertIn('id="step-options-summary"', self.html)
        # a collapsed block hides its body and shows the summary
        self.assertIn(".step.collapsed .step-body { display: none; }", self.html)
        self.assertIn(".step.collapsed .step-summary { display: inline; }", self.html)

    def test_change_buttons_toggle_a_block_both_ways(self):
        """Opened a step, found nothing to change — the same button folds it back.

        It used to only ever open: `stepSourceOpen = true`, and the button was drawn
        by `.step.collapsed` alone, so an expanded step had no way back short of
        reloading the page.
        """
        for btn in ("step-source-edit", "step-options-edit"):
            self.assertIn(f'id="{btn}"', self.html)
        self.assertIn("stepSourceOpen = !stepSourceOpen;", self.html)
        self.assertIn("stepOptionsOpen = !stepOptionsOpen;", self.html)
        # the button is on screen whenever the step CAN be folded, not only when it is
        self.assertIn(".step.can-collapse .step-edit-btn { display: inline-flex; }",
                      self.html)
        self.assertIn('step.classList.toggle("can-collapse", canCollapse)', self.html)

    def test_the_button_says_what_it_will_do(self):
        self.assertIn("button.textContent = open ? I18N.step_collapse_button "
                      ": I18N.step_change_button;", self.html)
        self.assertIn('button.setAttribute("aria-expanded", open ? "true" : "false")',
                      self.html)

    def test_nothing_to_fold_before_a_source_is_picked(self):
        """canCollapse is `!!src`: with no source the step is open and the button gone
        — folding away the only field that has to be filled in would be a trap."""
        self.assertIn('updateStepToggle("step-source", "step-source-edit", '
                      'stepSourceOpen, !!src)', self.html)
        self.assertIn('step.classList.toggle("collapsed", canCollapse && !open)',
                      self.html)

    def test_folding_the_source_closes_the_exclusions_panel(self):
        """The tree belongs to that step — it must not stay on screen without it."""
        handler = self.html.split('getElementById("step-source-edit")')[1][:600]
        self.assertIn('getElementById("excludes-panel").style.display = "none"', handler)

    def test_next_blocks_are_dimmed_not_blocked(self):
        self.assertIn('options.classList.toggle("step-dimmed", !src)', self.html)
        self.assertIn('.step.step-dimmed { opacity: 0.65; }', self.html)
        # the start button is never disabled by the step state — only by a running job
        self.assertIn('<button type="button" id="process-start-btn" class="btn btn-primary">',
                      self.html)
        self.assertIn("startBtn.disabled = processRunning;", self.html)
        self.assertNotIn("startBtn.disabled = !src", self.html)
        self.assertNotIn('getElementById("process-start-btn").disabled = !', self.html)

    def test_the_existing_option_checkboxes_moved_into_the_second_block(self):
        options_block = self.html.split('id="step-options"')[1].split('id="step-actions"')[0]
        for box in ("process-deep-checkbox", "process-geo-online-checkbox",
                    "process-faces-checkbox", "process-events-checkbox"):
            self.assertIn(box, options_block)
        # F81 explicitly does not add a preview-cache toggle here. Scoped to the options
        # block, which is the claim: F117 put the preview-cache CEILING in the settings
        # column of the "Cities" tab, so the key now appears on the page — just not as a
        # switch among the run options, which is what this case is about.
        self.assertNotIn("preview_cache", options_block)

    def test_do_not_scan_is_worded_apart_from_do_not_sort(self):
        # F82 put the two side by side in one tree, so the difference now lives in the
        # state labels rather than in a warning attached to the panel.
        self.assertEqual(ui._t("tri_scan_label", "ru"), "не сканировать")
        self.assertEqual(ui._t("tri_layout_label", "ru"), "не раскладывать")
        self.assertEqual(ui._t("override_exclude_button", "ru"), "Не трогать")
        for key in ("excludes_button", "excludes_title", "excludes_hint",
                    "step_source_title", "step_options_title", "step_actions_title",
                    "excludes_summary", "excludes_summary_none"):
            with self.subTest(key=key):
                entry = ui._UI_STRINGS[key]
                self.assertEqual(set(entry), {"ru", "en", "ja"})
                self.assertTrue(all(entry.values()))

    def test_tree_panel_is_wired_to_the_endpoints(self):
        self.assertIn('fetch("/api/source-tree?path="', self.html)
        self.assertIn('postJson("/api/source-tree/excludes"', self.html)
        self.assertIn('id="excludes-panel"', self.html)
        self.assertIn("function collectExcludes()", self.html)
        # marking a parent marks its subtree
        self.assertIn("function setSubtreeState(ul, state)", self.html)


if __name__ == "__main__":
    unittest.main()


class TestSourcePickKeepsTheStepOpen(unittest.TestCase):
    """Picking a folder must not collapse the step that holds the folder tree.

    Reported live: choosing a directory through Explorer jumped straight past the
    exclusions, and reaching them meant going back via "change". Exclusions belong to
    a specific root and are part of the same step, so the step stays open — it starts
    collapsed only on page load with an already-remembered source, where there is
    genuinely nothing to do.
    """

    def setUp(self):
        from sorta import ui

        self.html = ui._render_index_html("ru")

    def _fn(self, name: str) -> str:
        start = self.html.index("function " + name + "(")
        depth, i = 0, self.html.index("{", start)
        for j in range(i, len(self.html)):
            if self.html[j] == "{":
                depth += 1
            elif self.html[j] == "}":
                depth -= 1
                if depth == 0:
                    return self.html[start:j + 1]
        raise AssertionError("не найдено тело " + name)

    def test_choosing_a_source_opens_the_step(self):
        body = self._fn("sourceDirChanged")
        self.assertIn("stepSourceOpen = true", body)
        self.assertNotIn("stepSourceOpen = false", body)

    def test_choosing_a_source_shows_the_tree(self):
        self.assertIn("loadSourceTree()", self._fn("sourceDirChanged"))

    def test_tree_loader_is_shared_with_the_button(self):
        """One loader, two entry points — the button and the pick."""
        self.assertIn("function loadSourceTree(", self.html)
        self.assertIn("loadSourceTree(true)", self.html)
