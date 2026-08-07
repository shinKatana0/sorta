"""F221: the payload carries its own trust, and an empty root store cannot take it away.

The installer built before this could not download anything on a clean Windows: the
weights go through `urllib.request.urlopen`, urllib asks Windows for the root
certificates, and a freshly installed Windows has almost none — it fetches them on
demand, and on a clean machine that regularly does not happen. Every tier was
unreachable, which made the whole tiered construction unreachable on the one machine it
was built for.

Nothing caught it, and the reason is the point of this file: **checking whether TLS works
on a machine where TLS works proves nothing.** The build machine's root store is full,
`windows-latest` is a developer image, and both would have gone green on the broken
payload — exactly the way they went green on the payload that was missing the MSVC
runtime (F218).

So the check here does not ask whether the network works. It gives an interpreter an
EMPTY root store — `ssl.enum_certificates` returning nothing is precisely a clean
machine's store — and asks what `ssl.create_default_context()` ends up trusting. With the
payload's `sitecustomize.py` on the path it trusts exactly the certificate set the payload
carries; without it, it trusts whatever the machine has, which on that clean machine was
nothing. The certificate set in the fixtures is ONE certificate on purpose, so "our set"
and "the machine's set" cannot be confused for one another on any machine this suite runs
on.

The subprocess is not decoration either. `sitecustomize` is a thing CPython imports while
it is starting up, and running the file by hand with `exec()` would prove the four `..`
hops and nothing about whether the interpreter ever reaches it.
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import ssl
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import certifi

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _ROOT / "scripts" / "build_installer.py"
_SITECUSTOMIZE = _ROOT / "packaging" / "windows" / "sitecustomize.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("build_installer_f221", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


builder = _load_script()

# What the probe below reports back: the two variables, and the serial number of every CA
# the default context ended up trusting. Serial numbers rather than whole certificates
# because that is the shortest thing that still says WHICH certificate, and "which" is the
# entire question — a context holding a hundred roots and a context holding ours are both
# "not empty".
_PROBE = """
import json, os, ssl, sys

if hasattr(ssl, "enum_certificates"):
    # A CLEAN Windows: the root store is empty. This one line is the whole fixture — on a
    # machine where Windows has already filled the store the defect is invisible, which is
    # why it survived a build machine, a CI runner and a release.
    ssl.enum_certificates = lambda storename: []

context = ssl.create_default_context()
json.dump({
    "file": os.environ.get("SSL_CERT_FILE"),
    "dir": os.environ.get("SSL_CERT_DIR"),
    "serials": sorted(cert.get("serialNumber", "") for cert in context.get_ca_certs()),
}, sys.stdout)
"""

_PEM = re.compile(r"-----BEGIN CERTIFICATE-----.*?-----END CERTIFICATE-----\n",
                  re.DOTALL)


def one_certificate(index: int = 0) -> str:
    """One real certificate out of certifi's set, as PEM.

    Real because a context has to be able to LOAD it, and one because the size of the set
    is what tells our bundle apart from the machine's on any machine at all.
    """
    found = _PEM.findall(Path(certifi.where()).read_text(encoding="utf-8"))
    assert len(found) > index, "certifi's bundle holds fewer certificates than expected"
    return found[index]


def probe_serials(certificate: str) -> list[str]:
    """The serial numbers a context holds after loading exactly this PEM.

    Read back through `ssl` rather than written down, so the comparisons below are against
    the certificate itself and not against a constant that would have to be updated
    whenever certifi reorders its set.
    """
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.load_verify_locations(cadata=certificate)
    return sorted(cert.get("serialNumber", "") for cert in context.get_ca_certs())


def install_tree(root: Path, *, certificate: str | None = None,
                 sitecustomize: bool = True) -> Path:
    """A directory shaped like an installation: the two files this feature is about.

    `python\\Lib\\site-packages\\sitecustomize.py` is the real one, copied rather than
    re-written, and `lib\\certifi\\cacert.pem` is where certifi lands when the base tier
    is installed into `lib\\`.
    """
    site_packages = root / builder.PAYLOAD_SITE_PACKAGES
    site_packages.mkdir(parents=True, exist_ok=True)
    if sitecustomize:
        shutil.copy2(_SITECUSTOMIZE, site_packages / "sitecustomize.py")
    bundle = root / builder.PAYLOAD_CA_BUNDLE
    bundle.parent.mkdir(parents=True, exist_ok=True)
    bundle.write_text(certificate if certificate is not None else one_certificate(),
                      encoding="ascii")
    return root


def probe(installation: Path | None, **environment: str) -> dict:
    """Start an interpreter with an empty root store and report what it trusts.

    `installation` is put on PYTHONPATH by its `site-packages`, which is how a
    `sitecustomize.py` sitting there gets imported at startup — the same mechanism that
    finds it in the payload, where the directory is the interpreter's own. Passing None is
    the fix taken away.
    """
    env = {key: value for key, value in os.environ.items()
           if key not in ("SSL_CERT_FILE", "SSL_CERT_DIR", "PYTHONPATH", "PYTHONHOME",
                          "PYTHONSTARTUP")}
    if installation is not None:
        env["PYTHONPATH"] = str(installation / builder.PAYLOAD_SITE_PACKAGES)
    env.update(environment)
    completed = subprocess.run([sys.executable, "-c", _PROBE], capture_output=True,
                               text=True, env=env, check=False)
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


class TestAnEmptyRootStoreIsNotAnObstacle(unittest.TestCase):
    """Requirement 1 of the brief: with nothing trusted by the machine, the shipped
    configuration still finds the set the payload carries."""

    def test_the_default_context_trusts_the_set_the_payload_carries(self):
        with tempfile.TemporaryDirectory() as tmp:
            installation = install_tree(Path(tmp))
            reported = probe(installation)
            self.assertEqual(reported["file"],
                             str(installation / builder.PAYLOAD_CA_BUNDLE))
            self.assertEqual(reported["dir"],
                             str((installation / builder.PAYLOAD_CA_BUNDLE).parent))
            # Exactly the one certificate that was put in the payload, and nothing else:
            # on a machine whose store is full this is also what says the store lost.
            self.assertEqual(reported["serials"], probe_serials(one_certificate()))

    def test_the_same_configuration_covers_urllib_requests_and_the_hub(self):
        """`requests` and `huggingface_hub` were never broken — they use certifi
        themselves — and the fix must not be a second, different set for them. It is not:
        SSL_CERT_FILE names the very file `certifi.where()` returns inside the payload,
        so all three download paths verify against one set of roots."""
        with tempfile.TemporaryDirectory() as tmp:
            installation = install_tree(Path(tmp))
            named = Path(probe(installation)["file"])
            self.assertEqual(named.name, Path(certifi.where()).name)
            self.assertEqual(named.parent.name, "certifi")
            self.assertEqual(named.parent.parent, installation / builder.PAYLOAD_LIB)


class TestTheWatchdogGoesRed(unittest.TestCase):
    """Requirement 4, and the reason the defect lived: a check that cannot fail is not a
    check. F182, F216 and F218 each taught this the expensive way."""

    def test_with_the_fix_taken_away_the_empty_store_is_all_there_is(self):
        """The proof recorded in the report: run the same probe with the payload's
        `sitecustomize.py` off the path and the interpreter no longer trusts our set —
        it is back to the machine's own store, which on the owner's clean machine held
        nothing and here holds something entirely different."""
        with tempfile.TemporaryDirectory() as tmp:
            install_tree(Path(tmp))
            reported = probe(None)
            self.assertIsNone(reported["file"])
            self.assertNotEqual(reported["serials"], probe_serials(one_certificate()))


class TestAPersonsOwnCertificatesWin(unittest.TestCase):
    """Requirement 2: a corporate proxy with a root of its own is an ordinary thing."""

    def test_an_ssl_cert_file_that_was_already_set_is_not_overridden(self):
        with tempfile.TemporaryDirectory() as tmp:
            installation = install_tree(Path(tmp))
            theirs = Path(tmp) / "corporate-roots.pem"
            theirs.write_text(one_certificate(1), encoding="ascii")
            reported = probe(installation, SSL_CERT_FILE=str(theirs))
            self.assertEqual(reported["file"], str(theirs))
            # And it is theirs that is actually trusted, not only theirs that is named.
            self.assertEqual(reported["serials"], probe_serials(one_certificate(1)))

    def test_an_ssl_cert_dir_that_was_already_set_is_not_overridden_either(self):
        with tempfile.TemporaryDirectory() as tmp:
            installation = install_tree(Path(tmp))
            theirs = Path(tmp) / "roots.d"
            theirs.mkdir()
            reported = probe(installation, SSL_CERT_DIR=str(theirs))
            self.assertEqual(reported["dir"], str(theirs))
            # The half they did NOT name is still ours — the two are set independently.
            self.assertEqual(reported["file"],
                             str(installation / builder.PAYLOAD_CA_BUNDLE))

    def test_nothing_outside_this_process_is_touched(self):
        """`os.environ.setdefault` in a starting interpreter is this process and its
        children. The parent running the suite must come out of all of the above with the
        variables it had, which here is none of them."""
        self.assertNotIn("SSL_CERT_FILE", os.environ)


class TestThePathIsRelativeToTheInstallation(unittest.TestCase):
    """Requirement 3: the payload is built here and copied to somebody else's disk — the
    same reason `lib\\` is found by a relative `.pth`."""

    def test_a_payload_moved_to_another_directory_points_at_its_own_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = install_tree(Path(tmp) / "Programs" / "Sorta")
            second = Path(tmp) / "elsewhere" / "Sorta"
            shutil.copytree(first, second)
            self.assertEqual(probe(first)["file"],
                             str(first / builder.PAYLOAD_CA_BUNDLE))
            self.assertEqual(probe(second)["file"],
                             str(second / builder.PAYLOAD_CA_BUNDLE))

    def test_the_file_names_no_path_of_the_machine_that_built_it(self):
        source = _SITECUSTOMIZE.read_text(encoding="utf-8")
        self.assertIn("__file__", source)
        self.assertNotIn(str(_ROOT), source)

    def test_a_payload_with_no_certificate_set_changes_nothing(self):
        """It runs at the start of every process, including the tray, which has no
        console to complain into — so a missing set is silence here and a refusal in the
        build (below), not a traceback on somebody's screen."""
        with tempfile.TemporaryDirectory() as tmp:
            installation = install_tree(Path(tmp))
            (installation / builder.PAYLOAD_CA_BUNDLE).unlink()
            self.assertIsNone(probe(installation)["file"])


class TestVerificationIsNotWeakenedAnywhere(unittest.TestCase):
    """A boundary of the brief, pinned rather than promised: this feature is allowed to
    say WHERE the roots are and nothing else. A product that downloads and then RUNS
    model weights has to know where they came from."""

    def test_nothing_shipped_or_built_switches_verification_off(self):
        forbidden = ("_create_unverified_context", "_create_default_https_context",
                     "CERT_NONE", "verify=False", "PYTHONHTTPSVERIFY", "check_hostname")
        for path in (_SITECUSTOMIZE, _SCRIPT):
            source = path.read_text(encoding="utf-8")
            for needle in forbidden:
                with self.subTest(file=path.name, needle=needle):
                    self.assertNotIn(needle, source)

    def test_the_set_is_certifis_and_not_one_assembled_by_hand(self):
        """The other boundary: no roots of our own travel. The file the payload points at
        is the one certifi installs, in the directory certifi installs it in."""
        self.assertEqual(builder.PAYLOAD_CA_BUNDLE,
                         builder.PAYLOAD_LIB / "certifi" / "cacert.pem")
        self.assertEqual(Path(certifi.where()).name, builder.PAYLOAD_CA_BUNDLE.name)
        self.assertNotIn("BEGIN CERTIFICATE",
                         _SITECUSTOMIZE.read_text(encoding="utf-8"))


class TestThePlanNamesTheFile(unittest.TestCase):
    """Requirement 5: checkable without building, like the rest of the plan."""

    def test_the_payload_plan_copies_it_beside_the_pth(self):
        plan = builder.payload_plan(None)
        destinations = {destination.as_posix() for _source, destination in plan}
        self.assertIn("python/Lib/site-packages/sitecustomize.py", destinations)
        # Beside `_sorta_lib.pth`, which is not a coincidence: both are found relative to
        # that directory and both climb the same four levels out of it.
        self.assertEqual(builder.PAYLOAD_SITECUSTOMIZE.parent,
                         builder.PAYLOAD_SITE_PACKAGES)
        self.assertEqual(builder.PTH_LINE.count(".."), 3)

    def test_the_source_it_names_is_a_file_that_exists(self):
        sources = {source for source, _destination in builder.payload_plan(None)}
        self.assertIn(_ROOT / builder.SITECUSTOMIZE_SOURCE, sources)
        for source in sources:
            with self.subTest(source=source.name):
                self.assertTrue(source.is_file(), source)


class TestTheBuildRefusesAPayloadWithoutTrust(unittest.TestCase):
    """The loud half of a file that is deliberately silent: an installer is not compiled
    from a payload whose interpreter would be pointed at nothing."""

    def test_a_complete_payload_has_no_gap(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(builder.payload_trust_gap(install_tree(Path(tmp))))

    def test_a_payload_without_the_sitecustomize_is_named_as_such(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = install_tree(Path(tmp), sitecustomize=False)
            gap = builder.payload_trust_gap(payload)
            self.assertIsNotNone(gap)
            self.assertIn("sitecustomize.py", str(gap))

    def test_a_payload_without_the_certificate_set_is_named_as_such(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = install_tree(Path(tmp))
            (payload / builder.PAYLOAD_CA_BUNDLE).unlink()
            gap = builder.payload_trust_gap(payload)
            self.assertIsNotNone(gap)
            self.assertIn("certifi", str(gap))

    def test_the_build_stops_before_compiling_such_a_payload(self):
        """In the build and not only in the suite, the way F218's watchdog runs in both.
        The module below is a file with a module's suffix and no import table, so the
        completeness check passes and this one is what the build stops on."""
        with tempfile.TemporaryDirectory() as tmp:
            payload = install_tree(Path(tmp), sitecustomize=False)
            (payload / "lib" / "torch" / "lib").mkdir(parents=True)
            (payload / "lib" / "torch" / "lib" / "c10.dll").write_bytes(b"")
            with mock.patch.object(builder, "PAYLOAD", payload):
                code = builder.main(["--skip-payload", "--no-exiftool"])
            self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
