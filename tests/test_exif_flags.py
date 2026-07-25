"""F71: the exiftool read flag, and a fake exiftool that lets tests change what it reports.

The regression tests here are trivial on purpose — they are the guard against someone
putting `-fast2` back for speed and silently losing the metadata block of every HEIC.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import sorta.exif as exif

# A fake exiftool: speaks both protocols the code uses — the -stay_open argfile
# (stdin, response up to {ready}) and the one-shot command line. It answers from a JSON
# file that the test can rewrite between runs (the F71 situation: the metadata was on
# disk all along, only our reading of it changed) and records the arguments it was given,
# so a test can assert on the flags that actually reached the binary.
#
# F72 added what the session-pool tests need: every launch and every clean exit drops a
# marker file (counting processes, and proving none is left hanging), a per-file delay
# makes parallelism measurable, and `crash_on` kills a session mid-request — but only in
# -stay_open mode, so the one-shot fallback for that slice still answers.
_FAKE_EXIFTOOL = r'''
import json, os, sys, time
from pathlib import Path

META = Path(@META@)
ARGS = Path(@ARGS@)
CTL = Path(@CTL@)
MARKS = Path(@MARKS@)

ctl = json.loads(CTL.read_text(encoding="utf-8"))
stay_open = "-stay_open" in sys.argv
mark = "%d-%d" % (os.getpid(), time.time_ns())
(MARKS / ("start-" + mark)).write_text("", encoding="utf-8")


def respond(args):
    # atomic replace: several fake processes may report their arguments at once
    tmp = ARGS.with_name(ARGS.name + ".%s.tmp" % os.getpid())
    tmp.write_text(json.dumps(args), encoding="utf-8")
    os.replace(tmp, ARGS)
    meta = json.loads(META.read_text(encoding="utf-8"))
    recs = []
    for a in args:
        p = Path(a)
        if not p.is_absolute():
            continue
        if stay_open and p.name in ctl["crash_on"]:
            sys.exit(1)  # the session dies before {ready}
        if ctl["delay"]:
            time.sleep(ctl["delay"])
        if meta.get(p.name) is not None:
            recs.append({"SourceFile": a, **meta[p.name]})
    return recs


sys.stdout.reconfigure(encoding="utf-8")
if stay_open:
    sys.stdin.reconfigure(encoding="utf-8")
    args = []
    for line in sys.stdin:
        line = line.rstrip("\r\n")
        if line == "-execute":
            recs = respond(args)
            if recs:
                sys.stdout.write(json.dumps(recs) + "\n")
            sys.stdout.write("{ready}\n")
            sys.stdout.flush()
            args = []
        elif line == "-stay_open":
            break  # the next line is False — exit like the real exiftool
        else:
            args.append(line)
else:
    sys.stdout.write(json.dumps(respond(sys.argv[1:])))
(MARKS / ("exit-" + mark)).write_text("", encoding="utf-8")
'''


def _always_available() -> bool:
    return True


class FakeExifTool:
    """Substitutes the exiftool binary for the duration of a test.

    Patches `_EXIFTOOL_CMD` (the same hook the -stay_open protocol test uses),
    `exiftool_available` (so the Pillow fallback never hides the fake on a machine
    without exiftool) and the module-level session pool.
    """

    def __init__(self, tmpdir: Path, meta: dict | None = None, *,
                 crash_on: list[str] | None = None, delay: float = 0.0) -> None:
        self.meta_path = tmpdir / "fake_meta.json"
        self.args_path = tmpdir / "fake_args.json"
        self.ctl_path = tmpdir / "fake_ctl.json"
        self.marks_dir = tmpdir / "fake_marks"
        self.marks_dir.mkdir(exist_ok=True)
        self.args_path.write_text("[]", encoding="utf-8")
        self.ctl_path.write_text(
            json.dumps({"crash_on": crash_on or [], "delay": delay}), encoding="utf-8")
        self.set_meta(meta or {})
        self.script = tmpdir / "fake_exiftool.py"
        self.script.write_text(
            _FAKE_EXIFTOOL
            .replace("@META@", json.dumps(str(self.meta_path)))
            .replace("@ARGS@", json.dumps(str(self.args_path)))
            .replace("@CTL@", json.dumps(str(self.ctl_path)))
            .replace("@MARKS@", json.dumps(str(self.marks_dir))),
            encoding="utf-8",
        )
        self._orig = (exif._EXIFTOOL_CMD, exif.exiftool_available, exif._pool)
        exif._EXIFTOOL_CMD = [sys.executable, str(self.script)]
        exif.exiftool_available = _always_available
        exif._pool = exif.ExifToolPool()

    def set_meta(self, meta: dict) -> None:
        """What the "camera" reports, keyed by file name."""
        self.meta_path.write_text(json.dumps(meta), encoding="utf-8")

    def last_args(self) -> list[str]:
        return json.loads(self.args_path.read_text(encoding="utf-8"))

    def launches(self) -> int:
        """How many fake exiftool processes were started (sessions + one-shot calls)."""
        return len(list(self.marks_dir.glob("start-*")))

    def clean_exits(self) -> int:
        """How many of them finished on their own — the rest are still running or killed."""
        return len(list(self.marks_dir.glob("exit-*")))

    def restore(self) -> None:
        exif._pool.close()
        exif._EXIFTOOL_CMD, exif.exiftool_available, exif._pool = self._orig


class TestReadFlag(unittest.TestCase):
    """-fast2 stops reading before the HEIC metadata block — it must not come back."""

    def test_query_args_use_fast(self):
        self.assertIn("-fast", exif._QUERY_ARGS)
        self.assertNotIn("-fast2", exif._QUERY_ARGS)

    def test_session_args_inherit_the_flag(self):
        self.assertIn("-fast", exif._SESSION_ARGS)
        self.assertNotIn("-fast2", exif._SESSION_ARGS)


class TestFlagReachesExiftool(unittest.TestCase):
    """Both read paths (the -stay_open session and the one-shot fallback) pass the flag."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.path = self.root / "a.heic"
        self.fake = FakeExifTool(self.root, {"a.heic": {"Make": "samsung"}})

    def tearDown(self):
        self.fake.restore()
        self.tmp.cleanup()

    def test_session_passes_fast(self):
        out = exif._pool.sessions(1)[0].read([self.path])
        args = self.fake.last_args()
        self.assertIn("-fast", args)
        self.assertNotIn("-fast2", args)
        self.assertEqual(out[str(self.path.resolve())].make, "samsung")

    def test_one_shot_fallback_passes_fast(self):
        out = exif.read_batch_exiftool([self.path])
        args = self.fake.last_args()
        self.assertIn("-fast", args)
        self.assertNotIn("-fast2", args)
        self.assertEqual(out[str(self.path.resolve())].make, "samsung")

    def test_read_batch_uses_the_session(self):
        out = exif.read_batch([self.path])
        self.assertIn("-fast", self.fake.last_args())
        self.assertEqual(out[str(self.path.resolve())].make, "samsung")


if __name__ == "__main__":
    unittest.main()
