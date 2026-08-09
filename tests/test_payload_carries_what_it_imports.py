"""F218: the payload has to contain everything its own modules import.

The installer built before this shipped a payload that could not load `torch` on a clean
Windows — `c10.dll` died with WinError 126 — because three MSVC runtime libraries were
missing and the build machine happened to have them in System32. Nothing caught it: the
sources say nothing about DLL imports, and "does it start on the runner" answers whether
the RUNNER has the runtime.

So the check is about FILES: read what every `*.dll` and `*.pyd` of the payload imports,
and require every name to be either inside the payload or given by Windows. It needs no
clean machine, and this suite is where it is kept honest — a check nobody can see go red
is not a check (F182 and F216 both taught us that one the expensive way), so the first
test below builds a payload with a hole in it and reads the complaint.

The fixtures are real PE files, assembled here byte by byte. A parser tested only against
files a test wrote could agree with itself about a layout Microsoft never used, so one
test reads a module of the machine the suite is running on when that machine is Windows.
"""
from __future__ import annotations

import importlib.util
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _ROOT / "scripts" / "build_installer.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("build_installer_f218", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


builder = _load_script()

_IMAGE_BASE = 0x10000000
_SECTION_RVA = 0x1000
_RAW_AT = 0x200


def pe_module(imports: tuple[str, ...] = (), delayed: tuple[str, ...] = (),
              *, delay_uses_addresses: bool = False) -> bytes:
    """A minimal but structurally real PE32+ image importing the names given.

    One section holds both descriptor arrays and the strings they point at. `delayed`
    goes into data directory 13; `delay_uses_addresses` writes it the pre-2015 way, where
    the descriptor stores addresses the loaded image would have rather than offsets into
    it — the case the parser has to subtract the image base for.
    """
    blob = bytearray()
    placeholders: list[tuple[int, str]] = []

    descriptors = bytearray()
    for name in imports:
        placeholders.append((len(descriptors) + 12, name))
        descriptors += struct.pack("<IIIII", 0, 0, 0, 0, 0)
    descriptors += b"\0" * 20
    import_rva, import_size = _SECTION_RVA, len(descriptors)
    blob += descriptors

    delay_at = len(blob)
    delays = bytearray()
    for name in delayed:
        placeholders.append((delay_at + len(delays) + 4, name))
        delays += struct.pack("<IIIIIIII", 1 if not delay_uses_addresses else 0,
                              0, 0, 0, 0, 0, 0, 0)
    delays += b"\0" * 32
    blob += delays

    for where, name in placeholders:
        rva = _SECTION_RVA + len(blob)
        blob += name.encode("ascii") + b"\0"
        stored = rva + (_IMAGE_BASE if delay_uses_addresses and where >= delay_at else 0)
        struct.pack_into("<I", blob, where, stored)

    section_size = len(blob)
    raw = bytes(blob) + b"\0" * (-section_size % 512)

    optional = bytearray(240)
    struct.pack_into("<H", optional, 0, 0x20B)               # PE32+
    struct.pack_into("<Q", optional, 24, _IMAGE_BASE)
    struct.pack_into("<II", optional, 32, 0x1000, 0x200)     # section, file alignment
    struct.pack_into("<I", optional, 56, _SECTION_RVA + 0x1000)   # SizeOfImage
    struct.pack_into("<I", optional, 60, _RAW_AT)                 # SizeOfHeaders
    struct.pack_into("<I", optional, 108, 16)                     # NumberOfRvaAndSizes
    struct.pack_into("<II", optional, 112 + 8, import_rva, import_size)
    if delayed:
        struct.pack_into("<II", optional, 112 + 13 * 8,
                         _SECTION_RVA + delay_at, len(delays))

    section = bytearray(40)
    section[0:8] = b".rdata\0\0"
    struct.pack_into("<IIII", section, 8, section_size, _SECTION_RVA, len(raw), _RAW_AT)

    head = bytearray(_RAW_AT)
    head[0:2] = b"MZ"
    struct.pack_into("<I", head, 0x3C, 0x40)
    head[0x40:0x44] = b"PE\0\0"
    struct.pack_into("<HHIIIHH", head, 0x44, 0x8664, 1, 0, 0, 0, len(optional), 0x2022)
    head[0x58:0x58 + len(optional)] = optional
    head[0x148:0x148 + 40] = section
    return bytes(head) + raw


def fake_payload(root: Path, modules: dict[str, tuple[str, ...]],
                 carried: tuple[str, ...] = ()) -> Path:
    """A payload-shaped directory: `modules` import names, `carried` are files beside them."""
    for relative, imports in modules.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pe_module(imports))
    for name in carried:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(b"")
    return root


class TestReadingWhatAModuleImports(unittest.TestCase):
    """Sixty lines of `struct` instead of a dependency — and they have to be right."""

    def test_the_ordinary_import_directory_is_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c10.dll"
            path.write_bytes(pe_module(("MSVCP140.dll", "KERNEL32.dll")))
            self.assertEqual(builder.imported_dlls(path), ["MSVCP140.dll", "KERNEL32.dll"])

    def test_the_delay_loaded_directory_is_read_too(self):
        """A name reached only on the first call is still a name that has to be there."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "later.dll"
            path.write_bytes(pe_module(("KERNEL32.dll",), delayed=("msvcp140_1.dll",)))
            self.assertEqual(builder.imported_dlls(path),
                             ["KERNEL32.dll", "msvcp140_1.dll"])

    def test_a_delay_descriptor_written_the_old_way_is_read_too(self):
        """Before Visual Studio 2015 the descriptor stored addresses of the loaded image
        rather than offsets into it. Reading one as the other does not fail — it reads a
        name out of the wrong place, which is how a parser lies quietly."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "old.dll"
            path.write_bytes(pe_module(delayed=("msvcp140_atomic_wait.dll",),
                                       delay_uses_addresses=True))
            self.assertEqual(builder.imported_dlls(path), ["msvcp140_atomic_wait.dll"])

    def test_nothing_that_is_not_a_module_is_an_error(self):
        """This walks four hundred files; one unreadable one must not stop the other 399."""
        with tempfile.TemporaryDirectory() as tmp:
            for name, content in (("empty.dll", b""), ("text.dll", b"not a program"),
                                  ("stub.pyd", b"MZ" + b"\0" * 200)):
                with self.subTest(file=name):
                    path = Path(tmp) / name
                    path.write_bytes(content)
                    self.assertEqual(builder.imported_dlls(path), [])

    @unittest.skipUnless(sys.platform == "win32", "needs Windows' own modules to read")
    def test_a_module_this_machine_shipped_is_read_the_same_way(self):
        """The fixtures above were written here and would happily agree with a parser
        that has the layout wrong. This one was not: `shell32.dll` was built by Microsoft,
        it imports over a hundred names and it uses BOTH directories — `uiautomationcore`
        is delay-loaded, so reading it back is the proof that directory 13 is parsed
        against a real image and not only against one this file assembled."""
        path = Path(os.environ.get("SystemRoot", "C:\\Windows"))
        names = [name.lower() for name in
                 builder.imported_dlls(path / "System32" / "shell32.dll")]
        self.assertGreater(len(names), 100, names)
        for expected in ("ntdll.dll", "kernelbase.dll", "uiautomationcore.dll"):
            with self.subTest(name=expected):
                self.assertIn(expected, names)
        for name in names:
            with self.subTest(name=name):
                self.assertRegex(name, r"^[\w.+-]+\.(dll|drv|exe)$")


class TestTheWatchdogGoesRed(unittest.TestCase):
    """The point of the whole feature: it has to be able to fail, and be seen failing."""

    def test_a_module_importing_something_the_payload_has_not_is_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = fake_payload(Path(tmp), {
                "lib/onnxruntime/capi/onnxruntime_pybind11_state.pyd":
                    ("msvcp140_1.dll", "KERNEL32.dll"),
                "lib/torch/lib/torch_python.dll":
                    ("msvcp140_atomic_wait.dll", "c10.dll"),
            }, carried=("lib/torch/lib/c10.dll",))
            gaps = builder.payload_import_gaps(payload)
            self.assertEqual(
                sorted((str(module.as_posix()), name) for module, name in gaps),
                [("lib/onnxruntime/capi/onnxruntime_pybind11_state.pyd", "msvcp140_1.dll"),
                 ("lib/torch/lib/torch_python.dll", "msvcp140_atomic_wait.dll")])

    def test_a_payload_that_carries_everything_says_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = fake_payload(Path(tmp), {
                "lib/torch/lib/c10.dll": ("msvcp140.dll", "VCRUNTIME140.dll",
                                          "KERNEL32.dll"),
            }, carried=("python/msvcp140.dll", "python/vcruntime140.dll"))
            self.assertEqual(builder.payload_import_gaps(payload), [])

    def test_the_build_stops_before_compiling_an_incomplete_payload(self):
        """In the suite AND in the build (the brief asks for both): an installer must not
        be produced from a payload that cannot load itself."""
        with tempfile.TemporaryDirectory() as tmp:
            payload = fake_payload(Path(tmp), {
                "lib/torch/lib/torch_python.dll": ("msvcp140_atomic_wait.dll",)})
            with mock.patch.object(builder, "PAYLOAD", payload):
                code = builder.main(["--skip-payload", "--no-exiftool"])
            self.assertEqual(code, 1)

    def test_a_build_over_an_empty_directory_does_not_pass_for_a_complete_payload(self):
        """Nothing staged is not the same as nothing missing — the failure mode of every
        check that walks a directory and finds it empty."""
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(builder, "PAYLOAD", Path(tmp)):
                self.assertEqual(builder.main(["--skip-payload", "--no-exiftool"]), 1)


class TestWhatCountsAsCarried(unittest.TestCase):
    """The other way this check dies: going red on a payload that is perfectly fine."""

    def test_the_mangled_copies_numpy_and_shapely_carry_are_counted(self):
        """delvewheel gives a vendored library a name nothing else can collide with, and
        the importing module names that mangled copy. It is IN the payload — raising an
        alarm on it would make the check noise, and noise gets switched off."""
        mangled = "msvcp140-a4c2229b1e2f3c4d5e6f708192a3b4c5.dll"
        with tempfile.TemporaryDirectory() as tmp:
            payload = fake_payload(Path(tmp), {
                "lib/numpy/_core/_multiarray_umath.cp313-win_amd64.pyd": (mangled,),
                "lib/shapely/lib.cp313-win_amd64.pyd": (mangled.upper(),),
            }, carried=(f"lib/numpy.libs/{mangled}",))
            self.assertEqual(builder.payload_import_gaps(payload), [])

    def test_both_module_suffixes_are_walked(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = fake_payload(Path(tmp), {"lib/a.dll": ("gone.dll",),
                                               "lib/b.pyd": ("gone.dll",),
                                               "lib/c.txt": ()})
            (payload / "lib" / "c.txt").write_bytes(pe_module(("gone.dll",)))
            self.assertEqual([module.name for module, _name
                              in builder.payload_import_gaps(payload)], ["a.dll", "b.pyd"])


class TestWhatCountsAsSystem(unittest.TestCase):
    """The list this check lives or dies by: too soft and it passes on a broken payload,
    too strict and it goes red on every build until somebody switches it off."""

    def test_what_windows_provides_is_not_reported_as_missing(self):
        given = ("KERNEL32.dll", "ADVAPI32.dll", "ucrtbase.dll",
                 "api-ms-win-crt-runtime-l1-1-0.dll", "api-ms-win-core-synch-l1-2-0.dll",
                 "ext-ms-win-ntuser-window-l1-1-0.dll")
        for name in given:
            with self.subTest(name=name):
                self.assertTrue(builder.is_system_dll(name))

    def test_the_api_sets_are_a_family_and_the_reason_is_written_down(self):
        """`api-ms-win-crt-*` IS Windows 10 and 11 — it has to say so in words, or the
        next reader concludes they were forgotten and starts shipping them."""
        self.assertIn("api-ms-win-", builder.SYSTEM_DLL_PREFIXES)
        self.assertRegex(builder.SYSTEM_DLL_PREFIXES["api-ms-win-"],
                         r"(?i)windows 10")

    def test_the_three_libraries_of_this_feature_are_not_on_the_list(self):
        """The one assertion that keeps today's defect from passing tomorrow: call
        msvcp140 a system library and the payload is allowed to sail without it."""
        for name in builder.MSVC_RUNTIME_DLLS:
            with self.subTest(name=name):
                self.assertFalse(builder.is_system_dll(name))
                self.assertNotIn(name, builder.SYSTEM_DLLS)

    def test_nothing_of_ours_is_called_a_system_library(self):
        for name in ("torch_cpu.dll", "onnxruntime.dll", "c10.dll", "vcruntime140.dll",
                     "python313.dll", "opencv_world.dll", "libcrypto-3-x64.dll"):
            with self.subTest(name=name):
                self.assertFalse(builder.is_system_dll(name))

    def test_the_list_is_explicit_and_every_name_says_why_it_is_there(self):
        self.assertGreater(len(builder.SYSTEM_DLLS), 20)
        for name, reason in builder.SYSTEM_DLLS.items():
            with self.subTest(name=name):
                self.assertEqual(name, name.lower())
                self.assertGreater(len(reason), 10, reason)


class TestTheRuntimeTravelsAndIsRecorded(unittest.TestCase):
    """Three libraries, in the one directory the loader already looks in."""

    def test_the_plan_puts_them_beside_the_interpreter(self):
        plan = builder.payload_plan(None, Path("dist/msvc-runtime"))
        destinations = {str(destination.as_posix()) for _source, destination in plan}
        for name in ("msvcp140.dll", "msvcp140_1.dll", "msvcp140_atomic_wait.dll"):
            with self.subTest(name=name):
                self.assertIn(f"python/{name}", destinations)
        self.assertEqual(builder.PAYLOAD_PYTHON_EXE.parent, builder.PAYLOAD_PYTHON)

    def test_a_plan_without_a_runtime_directory_is_the_plan_it_always_was(self):
        destinations = {destination.as_posix() for _source, destination
                        in builder.payload_plan(None)}
        self.assertEqual(destinations,
                         {"config.example.yaml", "LICENSE", "NOTICE", "favicon.ico",
                          "python/Lib/site-packages/sitecustomize.py"})

    def test_the_manifest_names_the_version_of_the_runtime(self):
        """The same obligation exiftool's version travels for: a library bundled without
        its version recorded is one nobody can tell whether they have to update."""
        manifest = builder.build_manifest("1.2.3", exiftool=False)
        self.assertEqual(manifest["msvc_runtime_version"], builder.VC_REDIST_VERSION)
        self.assertRegex(manifest["msvc_runtime_version"], r"^\d+\.\d+\.\d+$")

    def test_the_source_is_pinned_by_version_and_by_checksum(self):
        """Not `aka.ms/vs/17/release`, which is whatever is newest today, and not a copy
        out of the System32 of whoever ran the build."""
        self.assertNotIn("aka.ms", builder.VC_REDIST_URL)
        self.assertRegex(builder.VC_REDIST_SHA256, r"^[0-9a-f]{64}$")
        # Microsoft's permanent download links carry the file's own sha256 in the path,
        # so the two halves of the pin can be checked against each other from here.
        self.assertIn(builder.VC_REDIST_SHA256, builder.VC_REDIST_URL.lower())

    def test_a_file_whose_checksum_is_wrong_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            cached = Path(tmp) / "VC_redist.x64.exe"
            cached.write_bytes(b"half a download")
            with self.assertRaises(SystemExit):
                builder.verified_download("https://example.invalid/x", cached, "0" * 64)

    def test_a_cached_file_that_matches_is_not_downloaded_again(self):
        with tempfile.TemporaryDirectory() as tmp:
            cached = Path(tmp) / "VC_redist.x64.exe"
            cached.write_bytes(b"the real thing")
            digest = builder.sha256(cached)
            # The URL is unreachable on purpose: reaching for it at all would be the bug.
            self.assertEqual(
                builder.verified_download("https://example.invalid/x", cached, digest),
                cached)


class TestTheBundleIsOpenedWithoutRunningIt(unittest.TestCase):
    """`vc_redist.x64.exe` is a WiX bundle: `/layout` only copies it and `expand` does not
    recognise it, so the cabinet behind the stub is found by reading the section the
    linker writes for the purpose."""

    @staticmethod
    def _bundle(stub_payload: bytes, attached: bytes) -> bytes:
        header = struct.pack("<II16sIIIIII", builder.BURN_MAGIC, 2, b"\0" * 16,
                             0x400, 0, 0, 0, 1, 2)
        header += struct.pack("<II", len(stub_payload), len(attached))
        section = bytearray(40)
        section[0:8] = b".wixburn"
        struct.pack_into("<IIII", section, 8, len(header), 0x1000, 512, 0x200)
        head = bytearray(0x400)
        head[0:2] = b"MZ"
        struct.pack_into("<I", head, 0x3C, 0x40)
        head[0x40:0x44] = b"PE\0\0"
        struct.pack_into("<HHIIIHH", head, 0x44, 0x8664, 1, 0, 0, 0, 240, 0x2022)
        head[0x200:0x200 + len(header)] = header
        head[0x148:0x148 + 40] = section
        return bytes(head) + stub_payload + attached + b"a signature"

    def test_the_cabinet_behind_the_stub_is_found_and_measured_by_its_own_header(self):
        """Its own header and not the container length: the authenticode signature sits
        after the containers, and it must not travel into the cabinet."""
        first = b"MSCF" + b"\0" * 4 + struct.pack("<I", 40) + b"\0" * 28
        second = b"MSCF" + b"\0" * 4 + struct.pack("<I", 60) + b"\0" * 48
        offset, length = builder.burn_attached_container(self._bundle(first, second))
        self.assertEqual(length, 60)
        bundle = self._bundle(first, second)
        self.assertEqual(bundle[offset:offset + 4], b"MSCF")
        self.assertEqual(bundle[offset:offset + length], second)

    def test_something_that_is_not_a_bundle_is_an_error_rather_than_a_guess(self):
        with self.assertRaises(SystemExit):
            builder.burn_attached_container(pe_module(("KERNEL32.dll",)))

    def test_the_unpacker_calls_windows_own_expand_by_full_path(self):
        """By name it would be found on PATH, and where the build runs from a POSIX shell
        that is a different program entirely — one that copies the cabinet and reports
        success."""
        # Separator-blind: what is being pinned is that the path is FULL, and on the
        # Linux runner `Path` joins the Windows root with a forward slash.
        path = builder.expand_binary().lower().replace("\\", "/")
        self.assertTrue(path.endswith("system32/expand.exe"), builder.expand_binary())
        self.assertNotEqual(path, "expand.exe")


if __name__ == "__main__":
    unittest.main()
