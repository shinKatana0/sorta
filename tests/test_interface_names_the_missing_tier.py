"""F217: the web app names the install tier that is missing, where it is missing.

A person installs Sorta with the installer and clears the "set it up at the end" box —
which is their right, the box is meant to be clearable. They then live in the web app and
never open a terminal. Until this feature the app never told them that anything was
missing: the word "tier" appears all over `sorta/ui/` and always means the OTHER thing
(the classification tier, fast CLIP against deep VLM), and the one place that did name a
way out named `uv tool install --force ".[gpu]"` — a command that works for nobody who
used the installer.

What is under test:

* the interface and `sorta doctor` cannot drift apart, because they read ONE probe. The
  precedent is F211: two check screens disagree within a release, which is why the wizard
  calls `doctor` instead of growing a screen of its own;
* the three answers, not two. Two of the four tiers install no packages at all — their
  weights are downloaded by the stage on first use — and reporting those as "not
  installed" would send a person to the wizard for something that happens by itself;
* the way out matches the platform, the pairing F213 established;
* a run that asked for the deep tier and went through on the fast one says so;
* and the boundary: the page NAMES the way out, the wizard does the work. Nothing in the
  web app installs a package, and that is a property rather than somebody's memory — the
  same shape as the F170 guard that keeps the cloud provider from coming back.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest import mock

from sorta import cli, i18n, install, tiers, ui, wizard
from sorta.ui import strings as ui_strings
from tests.test_ui_process import ProcessTestBase, _poll_until

_LANGS: tuple[i18n.Lang, ...] = ("ru", "en", "ja")
_ROOT = Path(__file__).resolve().parent.parent
_UI_FILES = tuple(p for p in (_ROOT / "sorta" / "ui").rglob("*.py")
                  if "__pycache__" not in p.parts)
_WEB_FILES = tuple(p for p in (_ROOT / "sorta" / "web").rglob("*")
                   if p.is_file() and p.suffix in (".html", ".css", ".js"))


def _state(key: str, *, packages: tuple[str, ...] = (),
           weights: tuple[str, ...] = ()) -> tiers.TierState:
    return tiers.TierState(key, missing_packages=packages, missing_weights=weights)


def _machine_with_nothing() -> list[tiers.TierState]:
    """The tier states of a machine that downloaded no weights and installed no extras."""
    return tiers.tier_states(package_present=lambda _name: False,
                             weights_cached=lambda _name: False)


def _by_key(states: list[tiers.TierState]) -> dict[str, tiers.TierState]:
    return {state.key: state for state in states}


def _doctor_marker(key: str, lang: i18n.Lang = "en") -> str:
    """The literal words that follow the tier's name in one of `doctor`'s sentences.

    Enough to tell the three apart in a printed line, and taken from the catalog rather
    than typed out here — the point of the test below is that the two screens agree, not
    that somebody kept a copy of the wording up to date.
    """
    template = i18n.cli_text(key, lang, name="{name}", missing="{missing}",
                             weights="{weights}", size="{size}")
    return template.split("{name}")[1].split("{")[0]


def _browser_note(key: str, info: dict, lang: i18n.Lang) -> str:
    """The sentence the script builds for one tier — `tierNote` of app.js, in Python.

    The template and the two constants are exactly what the page ships in `window.I18N`,
    so this renders what a person reads rather than an approximation of it.
    """
    strings = ui_strings._UI_STRINGS
    name = strings[f"tier_name_{key}"][lang]
    if info["state"] == "ready":
        return ""
    if info["state"] == "weights":
        return strings["tier_weights_note"][lang].format(
            name=name, weights=", ".join(info["missing"]),
            size=strings[f"tier_size_{key}"][lang])
    return (strings["tier_absent_note"][lang].format(name=name)
            + " " + strings["tier_add_hint"][lang])


class TestTheScreenAndTheDoctorReadOneProbe(unittest.TestCase):
    """The guard the whole feature hangs on: a second reading of "what is installed"
    would answer differently within a release, and nobody would find out from the code."""

    def test_all_three_names_are_the_same_function(self):
        self.assertIs(ui.process.tier_states, tiers.tier_states)
        self.assertIs(cli.tier_states, tiers.tier_states)

    def test_a_state_gets_the_same_verdict_on_both_screens(self):
        """`doctor` prints one of three sentences and the browser gets one of three
        names. This pairs them, state by state, so a screen that started calling a
        weights-only tier "missing" fails here rather than in somebody's install."""
        pairs = (
            (_state("faces"), "ready", "cli.doctor.tier_ready"),
            (_state("faces", weights=("buffalo_l",)), "weights",
             "cli.doctor.tier_weights"),
            (_state("deep", packages=("transformers",)), "absent",
             "cli.doctor.tier_absent"),
            # Both halves missing: the packages are what a person has to act on, and the
            # weights of a tier whose code is absent are never asked for.
            (_state("deep", packages=("transformers",), weights=("Qwen2.5-VL-3B",)),
             "absent", "cli.doctor.tier_absent"),
        )
        others = {"cli.doctor.tier_ready", "cli.doctor.tier_weights",
                  "cli.doctor.tier_absent"}
        for state, expected, key in pairs:
            with self.subTest(tier=state.key, expected=expected):
                payload = ui.process._tiers_payload([state])
                self.assertEqual(payload[state.key]["state"], expected)
                # ...and `doctor`, given the same state, prints the sentence that means
                # the same thing — and not one of the other two.
                line = cli._doctor_tier_lines("en", [state])[1]
                self.assertIn(_doctor_marker(key), line)
                for other in others - {key}:
                    self.assertNotIn(_doctor_marker(other), line)

    def test_the_interface_does_not_probe_the_install_by_itself(self):
        """The way a second copy comes back is not a rewritten module but a helper: a
        `find_spec` here, a cache directory there. The UI layer may ASK the probe and may
        not do any of it itself."""
        forbidden = ("insightface", "importlib.metadata", "PackageNotFoundError",
                     "tier_requirements", "hf_cache_dir")
        for path in _UI_FILES:
            text = path.read_text(encoding="utf-8")
            for needle in forbidden:
                with self.subTest(file=path.name, needle=needle):
                    self.assertNotIn(needle, text)


class TestATierThatDownloadsItselfIsNotCalledBroken(unittest.TestCase):
    """The mistake this feature could most easily make. `faces` and `search` install no
    packages: their weights arrive with the first run of the stage that needs them, so
    "not installed" would be a lie AND would send a person to the wizard for nothing.
    F216 built the middle state for exactly this; the screen has to use it."""

    def test_the_weights_only_tiers_report_the_middle_state(self):
        payload = ui.process._tiers_payload(_machine_with_nothing())
        for key in ("faces", "search"):
            with self.subTest(tier=key):
                self.assertEqual(payload[key]["state"], "weights")
                self.assertEqual(payload[key]["missing"],
                                 list(wizard.TIERS_BY_KEY[key].weights))

    def test_the_sentence_states_the_size_and_does_not_say_not_installed(self):
        payload = ui.process._tiers_payload(_machine_with_nothing())
        expected_size = {"faces": "400", "search": "1.4"}
        for key in ("faces", "search"):
            for lang in _LANGS:
                with self.subTest(tier=key, lang=lang):
                    note = _browser_note(key, payload[key], lang)
                    self.assertIn(expected_size[key], note)
                    self.assertIn(wizard.TIERS_BY_KEY[key].weights[0], note)
                    absent = ui_strings._UI_STRINGS["tier_absent_note"][lang]
                    self.assertNotIn(absent.split("{")[0].strip(), note)
                    # ...and no wizard is offered: nothing has to be added by hand.
                    self.assertNotIn("sorta-setup", note)

    def test_a_tier_that_is_in_place_is_not_mentioned_at_all(self):
        """The screen does not turn into a list of what is already there."""
        ready = [tiers.TierState(tier.key) for tier in wizard.TIERS]
        payload = ui.process._tiers_payload(ready)
        for key, info in payload.items():
            with self.subTest(tier=key):
                self.assertEqual(info["state"], "ready")
                self.assertEqual(_browser_note(key, info, "en"), "")

    def test_the_script_draws_nothing_for_a_tier_in_place(self):
        source = (_ROOT / "sorta" / "web" / "app" / "app.js").read_text(encoding="utf-8")
        self.assertIn('if (!info || info.state === "ready") return "";', source)


class TestTheWayOutMatchesTheMachine(unittest.TestCase):
    """F213's pairing, one screen further out: the Start menu belongs to the Windows
    installer, and an install that came from `uv tool install` has no such entry."""

    def test_the_install_decides_which_line_is_offered(self):
        """F230: it used to be the PLATFORM that decided, and that was the defect — a
        checkout on Windows was sent to a Start menu item it does not have. The page asks
        the same question the doctor does, one screen further out."""
        self.assertEqual(tiers._tier_hint_key(install.KIND_INSTALLED),
                         "cli.doctor.tier_hint.installed")
        self.assertEqual(tiers._tier_hint_key(install.KIND_TOOL),
                         "cli.doctor.tier_hint.tool")
        self.assertEqual(tiers._tier_hint_key(install.KIND_CHECKOUT),
                         "cli.doctor.tier_hint.checkout")

    def test_the_page_carries_the_line_of_this_install_and_not_another(self):
        for kind, other in ((install.KIND_INSTALLED, install.KIND_CHECKOUT),
                            (install.KIND_TOOL, install.KIND_INSTALLED),
                            (install.KIND_CHECKOUT, install.KIND_INSTALLED)):
            expected = install.advice_key("cli.doctor.tier_hint", kind)
            with mock.patch.object(tiers, "_tier_hint_key", lambda _key=expected: _key):
                generated = ui_strings._tier_strings()["tier_add_hint"]
            for lang in _LANGS:
                with self.subTest(kind=kind, lang=lang):
                    self.assertEqual(generated[lang],
                                     i18n.cli_text(expected, lang).strip())
                    self.assertNotEqual(
                        generated[lang],
                        i18n.cli_text(install.advice_key("cli.doctor.tier_hint", other),
                                      lang).strip())

    def test_only_the_installed_copy_names_the_menu_item(self):
        installed = i18n.cli_text("cli.doctor.tier_hint.installed", "en")
        tool = i18n.cli_text("cli.doctor.tier_hint.tool", "en")
        checkout = i18n.cli_text("cli.doctor.tier_hint.checkout", "en")
        self.assertIn("Start menu", installed)
        for line in (tool, checkout):
            self.assertNotIn("Start menu", line)
        for line in (installed, tool):
            self.assertIn("sorta-setup", line)
        # ...and the developer keeps the command that is true in a checkout.
        self.assertIn("uv sync --extra", checkout)

    def test_the_note_of_an_absent_tier_ends_in_that_line(self):
        payload = ui.process._tiers_payload(
            [_state("deep", packages=("transformers",))])
        for lang in _LANGS:
            with self.subTest(lang=lang):
                note = _browser_note("deep", payload["deep"], lang)
                self.assertTrue(note.endswith(
                    ui_strings._UI_STRINGS["tier_add_hint"][lang]), note)


class TestTheCaptionsAreTheProductsOwnWords(unittest.TestCase):
    """Every caption about a tier is taken from the catalog `sorta-setup` and
    `sorta doctor` answer from — a hand-written copy of one would drift, and drifting is
    what this feature is about."""

    def test_the_tier_captions_come_from_the_setup_catalog(self):
        for tier in wizard.TIERS:
            for lang in _LANGS:
                with self.subTest(tier=tier.key, lang=lang):
                    self.assertEqual(ui_strings._UI_STRINGS[f"tier_name_{tier.key}"][lang],
                                     tier.name(lang))
                    self.assertEqual(ui_strings._UI_STRINGS[f"tier_size_{tier.key}"][lang],
                                     wizard.human_size(tier.download_mb, lang))

    def test_the_middle_sentence_is_the_doctors_own(self):
        for lang in _LANGS:
            with self.subTest(lang=lang):
                template = ui_strings._UI_STRINGS["tier_weights_note"][lang]
                rendered = template.format(name="N", weights="W", size="S")
                self.assertEqual(rendered,
                                 i18n.cli_text("cli.doctor.tier_weights", lang, name="N",
                                               weights="W", size="S").strip())

    def test_the_two_captions_that_name_a_tier_in_prose_name_it_the_same_way(self):
        """Two sentences on this screen mention a tier inside prose rather than as a
        value, so the name is written out in them — and pinned here to the catalog, which
        is the only reason writing it out is allowed."""
        for caption, key, gone in (("env_cpu_warning", "gpu", "uv tool install"),
                                   ("process_deep_hint", "deep", "uv sync")):
            for lang in _LANGS:
                with self.subTest(caption=caption, lang=lang):
                    text = ui_strings._UI_STRINGS[caption][lang]
                    self.assertIn(wizard.TIERS_BY_KEY[key].name(lang), text)
                    # ...and neither of them offers a command for a source checkout.
                    self.assertNotIn(gone, text)

    def test_every_new_caption_exists_in_three_distinct_languages(self):
        keys = ("tier_absent_note", "tier_weights_note", "tier_add_hint",
                "process_deep_falls_back", "process_deep_fell_back", "env_cpu_warning",
                *(f"tier_name_{tier.key}" for tier in wizard.TIERS))
        for key in keys:
            entry = ui_strings._UI_STRINGS[key]
            with self.subTest(key=key):
                self.assertEqual(set(entry), set(_LANGS))
                for lang, value in entry.items():
                    self.assertTrue(value.strip(), f"{key}/{lang} is empty")
                self.assertEqual(len(set(entry.values())), 3, entry)


class TestNothingIsInstalledFromTheBrowser(unittest.TestCase):
    """The boundary, as a property. The page NAMES the way out and the wizard does the
    work — installing is minutes long, fails in ways that need a person reading the
    output, and the product stands on a local page that does not drive a package
    installer. The same shape as the F170 guard: read the shipped files as text, because
    what brings this back is a helper or a route, not a rewritten module."""

    FORBIDDEN = ("pip install", "uv pip", "install_command", "run_install",
                 "sorta.wizard", "_add_tiers")

    def test_no_file_of_the_interface_installs_anything(self):
        files = _UI_FILES + _WEB_FILES
        self.assertGreater(len(files), 5)
        for path in files:
            text = path.read_text(encoding="utf-8")
            for needle in self.FORBIDDEN:
                with self.subTest(file=path.name, needle=needle):
                    self.assertNotIn(needle, text)

    def test_no_route_of_the_server_is_about_installing(self):
        source = (_ROOT / "sorta" / "ui" / "__init__.py").read_text(encoding="utf-8")
        for needle in ('"/api/install', '"/api/setup', '"/api/tiers'):
            with self.subTest(route=needle):
                self.assertNotIn(needle, source)

    def test_the_env_route_answers_and_changes_nothing(self):
        """The one route that carries the tier states is a GET that reads a handful of
        answers — there is no verb on it.

        F222 added two of those answers (`parts`, `weights`): what each line of the run
        screen would download, out of the same probe. Still a read.
        """
        payload = ui.process._env_payload()
        self.assertEqual(set(payload),
                         {"gpu_profile", "gpu_present", "tiers", "parts", "weights"})


class TestASilentFallBackIsSaidOutLoud(ProcessTestBase):
    """The half that catches the person who has ALREADY run it. `junk.py` falls back to
    the fast tier without failing the run — deliberately, and it is not touched here —
    so the run finishes, the collection is unchanged, and the reason is in a log nobody
    opens. `media_class.tier` has recorded which tier handled a frame since schema v11;
    the run screen reads the same thing the Overview tab reads."""

    def _classify(self, tier: str) -> None:
        """A frame in the index, classified by `tier` — the DB state, without a model."""
        file_id, _path, _content = self.add_photo_file(f"{tier}.jpg")
        self.conn.execute(
            """INSERT INTO media_class (file_id, verdict, source, score, updated_at, tier)
               VALUES (?, 'photo', ?, NULL, '2026-08-07', ?)""",
            (file_id, tier, tier))
        self.conn.commit()

    def _run_with_deep(self) -> dict:
        self.patch_fast_stages()
        self.start_server()
        status, _resp = self.post("/api/process",
                                  {"source_dir": str(self.src_dir), "deep": True})
        self.assertEqual(status, 200)
        return _poll_until(self.status, lambda d: d["finished"])

    def test_asked_for_the_deep_tier_and_the_fast_one_ran(self):
        self._classify("clip")
        final = self._run_with_deep()
        self.assertTrue(final["deep_requested"])
        self.assertFalse(final["deep_ran"])

    def test_the_deep_tier_did_run_and_nothing_is_claimed(self):
        self._classify("vlm")
        final = self._run_with_deep()
        self.assertTrue(final["deep_requested"])
        self.assertTrue(final["deep_ran"])

    def test_a_run_that_never_asked_for_it_is_not_answered_for(self):
        self._classify("clip")
        self.patch_fast_stages()
        self.start_server()
        self.post("/api/process", {"source_dir": str(self.src_dir)})
        final = _poll_until(self.status, lambda d: d["finished"])
        self.assertFalse(final["deep_requested"])
        self.assertIsNone(final["deep_ran"])

    def test_the_screen_says_it_and_only_after_a_finished_run(self):
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        self.assertIn('id="process-deep-fallback"', html)
        self.assertIn("data.deep_requested && data.deep_ran === false", html)
        self.assertIn("!!data.finished && !data.error && !data.cancel_requested", html)
        self.assertIn("I18N.process_deep_fell_back", html)

    def test_the_reader_of_the_database_is_the_overviews_own_question(self):
        """One column, one meaning: `tier='vlm'` is written only when the model was
        actually raised — a fall back writes 'clip'."""
        self.assertFalse(ui.process._deep_tier_ran(self.conn))
        self._classify("clip")
        self.assertFalse(ui.process._deep_tier_ran(self.conn))
        self._classify("vlm")
        self.assertTrue(ui.process._deep_tier_ran(self.conn))


class TestTheNoteStandsAtTheOption(ProcessTestBase):
    """The general banner at the top of a screen is scrolled past — that mechanism is
    right for "everything is slower than it could be" and wrong for this. The line stands
    where the checkbox is, next to the price of the stage the screen already shows. And
    it does not block anything: a person may start the run and get an honest result from
    the fast tier."""

    def test_the_notes_are_slots_next_to_the_two_checkboxes(self):
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        for element in ("process-faces-tier-note", "process-deep-tier-note"):
            with self.subTest(element=element):
                self.assertIn(f'id="{element}"', html)
        self.assertIn('id="process-deep-tier-note" class="process-toggle-hint '
                      'process-toggle-warn" style="display:none"', html)

    def test_the_note_does_not_wait_for_the_checkbox(self):
        """The person this is written for never ticks it — a note that appears only on a
        tick is a note they never see."""
        source = (_ROOT / "sorta" / "web" / "app" / "app.js").read_text(encoding="utf-8")
        self.assertIn("function updateTierNotes()", source)
        self.assertNotIn("updateVlmMissingWarning", source)

    def test_the_note_itself_still_only_explains(self):
        """F217 wrote "we explain, we do not forbid" and asserted that the note function
        disables nothing. F222 §6b reversed the FORBIDDING half for one case — a tier
        whose packages are absent makes the checkbox useless, and a control that does
        nothing is what F211 rules out — but not this half: the sentence is still a
        sentence, and whatever goes dead is decided elsewhere
        (`updateOptionAvailability`), from the probe rather than from the wording.
        """
        source = (_ROOT / "sorta" / "web" / "app" / "app.js").read_text(encoding="utf-8")
        notes = source.split("function tierNote(")[1].split("function setPartNote")[0]
        self.assertNotIn("disabled", notes)

    def test_a_tier_that_only_lacks_its_weights_never_disables_anything(self):
        """The distinction F216 built the middle state for, now that a state CAN disable
        a control: weights arrive on first use, so an option waiting for them is a normal
        option — only absent packages make one impossible."""
        weights_only = [tiers.TierState("faces", missing_weights=("buffalo_l",))]
        parts = ui.process._parts_payload(tiers.run_parts(weights_only))
        self.assertTrue(parts["faces"]["available"])
        self.assertEqual(parts["faces"]["missing"], ["buffalo_l"])

    def test_the_tier_states_reach_the_browser_on_the_env_request(self):
        ui.process._gpu_present_cache_clear()
        self.addCleanup(ui.process._gpu_present_cache_clear)
        with mock.patch.object(ui.process, "nvidia_gpu_present", return_value=False):
            self.start_server()
            _status, body, _ctype = self.get("/api/env")
        data = json.loads(body)
        self.assertIn("deep", data["tiers"])
        self.assertIn("faces", data["tiers"])


if __name__ == "__main__":
    unittest.main()
