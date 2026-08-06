#!/usr/bin/env python3
"""F211 — build the Windows installer: the base tier whole, everything else offered.

    python scripts/build_installer.py                 # payload + installer
    python scripts/build_installer.py --dry-run       # print every step, run none
    python scripts/build_installer.py --no-exiftool   # build the Pillow-fallback variant
    python scripts/build_installer.py --sign          # ...and the signing step (opt-in)

What it makes, and why it looks like this
-----------------------------------------
The payload is a directory that is COPIED to wherever somebody installs the program:

    python\\      a standalone CPython, fetched by `uv python install`
    lib\\         the packages, from `uv pip install --target` — a plain tree
    uv.exe       the same resolver, kept for the tiers the wizard offers later
    exiftool\\    the metadata reader (see the decision in packaging/windows/README.md)
    favicon.ico, config.example.yaml, LICENSE, NOTICE
    sorta-install.json   what was shipped, in paths relative to itself

`--target` and not a virtualenv, deliberately: a venv records the absolute path of the
interpreter it was created from, so it cannot be built here and installed there, while a
target tree records nothing at all and is found by one `.pth` file. `uv` does the
resolving here and again on the target machine when a tier is added — ONE mechanism, which
is the boundary the brief draws around not re-packaging our dependencies by hand.

What is checkable without building
----------------------------------
Everything this module returns rather than runs: the tier manifest, the commands, the
payload plan, the paths, and the pairing of the tier list with `pyproject.toml`'s extras
(`unaccounted_extras` — the watchdog, run both here and in the suite, so an extra added to
the project and forgotten in the installer fails the gate). Building the real thing needs
a network, `uv`, Inno Setup and a Windows machine, and pretending otherwise in a test would
prove nothing — the manual checklist is in packaging/windows/README.md.

Signing: OFF by default and not in the path at all (owner's decision, 2026-08-06 — the
installer ships unsigned and SmartScreen is warned about in the README and the guides
instead). `--sign`, or SORTA_SIGN_INSTALLER=1, adds one step at the end that runs
`signtool` with what SORTA_SIGN_* names. When a certificate appears it plugs in; nothing
above it changes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sorta import wizard  # noqa: E402 — after the path insert, so a checkout works

ROOT = Path(__file__).resolve().parent.parent
PACKAGING = ROOT / "packaging" / "windows"
ISS = PACKAGING / "sorta.iss"
DIST = ROOT / "dist" / "windows"
PAYLOAD = DIST / "payload"

# The interpreter that ships. Inside `requires-python` of pyproject.toml (checked by the
# suite), and pinned rather than "latest": the version somebody installs is the version
# the wheels of the base tier were resolved for.
PYTHON_VERSION = "3.13"

# Where things sit INSIDE the payload. These strings are what the manifest carries, so
# they are relative — the build cannot know where a person will install the program.
PAYLOAD_PYTHON = Path("python")
PAYLOAD_PYTHON_EXE = PAYLOAD_PYTHON / "python.exe"
PAYLOAD_LIB = Path("lib")
PAYLOAD_UV = Path("uv.exe")
PAYLOAD_EXIFTOOL = Path("exiftool") / "exiftool.exe"

# The one line that puts `lib\` on the path of the shipped interpreter. Relative to the
# .pth file's own directory (`python\Lib\site-packages`), which is what makes the whole
# payload movable.
PTH_NAME = "_sorta_lib.pth"
PTH_LINE = "..\\..\\..\\lib"

# The environment the optional signing step reads. Nothing here is consulted unless
# signing was asked for.
ENV_SIGN = "SORTA_SIGN_INSTALLER"
ENV_SIGN_TOOL = "SORTA_SIGN_TOOL"
ENV_SIGN_CERT = "SORTA_SIGN_CERT"
ENV_SIGN_PASSWORD = "SORTA_SIGN_PASSWORD"
ENV_SIGN_TIMESTAMP = "SORTA_SIGN_TIMESTAMP"
DEFAULT_TIMESTAMP_URL = "http://timestamp.digicert.com"

# The files copied into the payload as they are: source relative to the repository root,
# destination relative to the payload.
STATIC_PAYLOAD: tuple[tuple[str, str], ...] = (
    ("config.example.yaml", "config.example.yaml"),
    ("LICENSE", "LICENSE"),
    ("NOTICE", "NOTICE"),
    ("sorta/web/favicon.ico", "favicon.ico"),
)


# --- the watchdog: the tiers against pyproject.toml ----------------------------------


def project_extras(pyproject_text: str) -> set[str]:
    """Every extra declared by the project."""
    data = tomllib.loads(pyproject_text)
    return set(data.get("project", {}).get("optional-dependencies", {}))


def project_version(pyproject_text: str) -> str:
    match = re.search(r'^version\s*=\s*"([^"]+)"', pyproject_text, re.M)
    if not match:
        raise ValueError("pyproject.toml: no `version = \"X.Y.Z\"`")
    return match.group(1)


def unaccounted_extras(pyproject_text: str) -> tuple[set[str], set[str]]:
    """(extras the installer never heard of, extras it names that no longer exist).

    Both directions matter and they fail differently: the first is a tier somebody
    added to the project and forgot here — the wizard would never offer it — and the
    second is the installer promising something the project cannot install any more.
    """
    declared = wizard.declared_extras()
    actual = project_extras(pyproject_text)
    return actual - declared, declared - actual


# --- the commands (returned, not run — this is the checkable half) -------------------


def python_install_command(uv: str, destination: Path,
                           version: str = PYTHON_VERSION) -> list[str]:
    """Fetch the standalone CPython that ships inside the payload."""
    return [uv, "python", "install", "--install-dir", str(destination), version]


def base_install_command(uv: str, python: Path, target: Path,
                         project: Path = ROOT) -> list[str]:
    """Install the base tier into the payload — the project with the base tier's extras.

    The extras come from the tier catalog, so the installer carries exactly what the
    wizard calls the base tier: the hardware profile (`cpu`) and the tray icon the
    shortcut starts.
    """
    extras = ",".join(wizard.BASE_TIER.extras)
    return [uv, "pip", "install", "--python", str(python), "--target", str(target),
            f"{project}[{extras}]"]


def iscc_command(iscc: str, version: str, *, payload: Path = PAYLOAD,
                 output: Path = DIST, script: Path = ISS) -> list[str]:
    """Compile the installer. Every variable part is a /D define — the .iss holds none."""
    return [iscc, f"/DVersion={version}", f"/DPayloadDir={payload}",
            f"/DOutputDir={output}", str(script)]


def installer_path(version: str, output: Path = DIST) -> Path:
    """Where Inno writes it — the name `OutputBaseFilename` of the .iss builds."""
    return output / f"sorta-{version}-setup.exe"


def signing_requested(env: dict[str, str] | None = None, *, flag: bool = False) -> bool:
    """Is the signing step on? Off unless asked, and asking is a flag or one variable."""
    environment = os.environ if env is None else env
    return bool(flag or environment.get(ENV_SIGN, "").strip() not in ("", "0", "false"))


def sign_command(target: Path, env: dict[str, str] | None = None) -> list[str]:
    """`signtool sign …` for the certificate the environment names.

    Deliberately a step of its own, outside the build path: today there is no
    certificate (owner's decision, 2026-08-06) and the installer ships unsigned. When one
    appears, this is where it plugs in — nothing above has to be rewritten.
    """
    environment = os.environ if env is None else env
    certificate = environment.get(ENV_SIGN_CERT)
    if not certificate:
        raise ValueError(f"signing was asked for but {ENV_SIGN_CERT} names no certificate")
    command = [environment.get(ENV_SIGN_TOOL) or "signtool", "sign", "/f", certificate]
    password = environment.get(ENV_SIGN_PASSWORD)
    if password:
        command += ["/p", password]
    command += ["/fd", "sha256", "/tr",
                environment.get(ENV_SIGN_TIMESTAMP) or DEFAULT_TIMESTAMP_URL,
                "/td", "sha256", str(target)]
    return command


# --- the payload ---------------------------------------------------------------------


def payload_plan(exiftool: Path | None, root: Path = ROOT) -> list[tuple[Path, Path]]:
    """(source, destination-inside-the-payload) for everything copied as it is."""
    plan = [(root / source, Path(destination)) for source, destination in STATIC_PAYLOAD]
    if exiftool is not None:
        plan.append((exiftool, PAYLOAD_EXIFTOOL))
    return plan


def tier_summary() -> list[dict]:
    """The tiers as the download page and the release notes state them."""
    return [
        {"key": tier.key, "extras": list(tier.extras), "weights": list(tier.weights),
         "download_mb": tier.download_mb, "optional": tier.optional}
        for tier in wizard.TIERS
    ]


def directory_size(path: Path) -> int:
    """Bytes under `path`; 0 if it is not there (a dry run has staged nothing)."""
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def exiftool_version(binary: Path | None) -> str | None:
    """`exiftool -ver`, or None if it cannot be asked.

    Bundling a binary means owing an update to whoever installed it, and an obligation
    nobody can see the state of is not one that gets met — so the version travels in the
    manifest (NOTICE §3 promises exactly this).
    """
    if binary is None:
        return None
    try:
        completed = subprocess.run([str(binary), "-ver"], capture_output=True, text=True,
                                   timeout=15, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return (completed.stdout or "").strip() or None


def build_manifest(version: str, *, exiftool: bool, tool_version: str | None = None,
                   payload: Path = PAYLOAD,
                   python_version: str = PYTHON_VERSION) -> dict:
    """What the installer leaves next to the program for the wizard to read.

    Paths are relative to this file's own directory: the payload is built here and
    installed somewhere else, and a relative path needs no install-time rewriting to
    survive that (`wizard.manifest_path_of` resolves them).
    """
    return {
        "version": version,
        "python_version": python_version,
        "python": str(PAYLOAD_PYTHON_EXE),
        "lib": str(PAYLOAD_LIB),
        "uv": str(PAYLOAD_UV),
        # The exiftool decision, recorded rather than assumed: the wizard says one
        # sentence or another by this value, so a build without the binary cannot leave
        # a person guessing why HEIC has no dates.
        "exiftool": str(PAYLOAD_EXIFTOOL) if exiftool else None,
        "exiftool_version": tool_version,
        "tiers": tier_summary(),
        "payload_bytes": directory_size(payload),
    }


def sha256(path: Path) -> str:
    """The checksum published beside the installer — the answer to an unsigned download."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --- running it ----------------------------------------------------------------------


class Builder:
    """The steps, with `--dry-run` as the one switch between printing and doing."""

    def __init__(self, *, dry_run: bool = False) -> None:
        self.dry_run = dry_run

    def run(self, command: list[str]) -> None:
        print("$ " + " ".join(command))
        if self.dry_run:
            return
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            raise SystemExit(f"failed ({completed.returncode}): {' '.join(command)}")

    def copy(self, source: Path, destination: Path) -> None:
        print(f"  copy {source} -> {destination}")
        if self.dry_run:
            return
        if not source.exists():
            raise SystemExit(f"missing payload file: {source}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(source, destination)

    def write(self, path: Path, text: str) -> None:
        print(f"  write {path}")
        if self.dry_run:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def find_exiftool(explicit: str | None) -> Path | None:
    """The exiftool binary to bundle: what was named, or what is on PATH."""
    if explicit:
        candidate = Path(explicit)
        if candidate.is_dir():
            candidate = candidate / "exiftool.exe"
        return candidate
    found = shutil.which("exiftool")
    return Path(found) if found else None


def build(args: argparse.Namespace) -> int:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    missing, stale = unaccounted_extras(pyproject)
    if missing or stale:
        # The same watchdog the suite runs, on the build itself: an installer that does
        # not know about an extra of the project must not be produced at all.
        print(f"tier manifest out of date — extras missing from it: {sorted(missing)}; "
              f"extras it names that the project has not: {sorted(stale)}")
        return 1
    version = project_version(pyproject)
    builder = Builder(dry_run=args.dry_run)
    uv = shutil.which("uv") or "uv"

    exiftool = None if args.no_exiftool else find_exiftool(args.exiftool)
    if exiftool is None and not args.no_exiftool:
        print("exiftool was not found. Bundle it (--exiftool <dir>) or build the "
              "Pillow-fallback variant on purpose (--no-exiftool) — the wizard says "
              "which of the two this build is, so the choice has to be made here.")
        return 1

    if not args.skip_payload:
        if not args.dry_run and PAYLOAD.exists():
            shutil.rmtree(PAYLOAD)
        builder.run(python_install_command(uv, PAYLOAD / PAYLOAD_PYTHON))
        builder.run(base_install_command(uv, PAYLOAD / PAYLOAD_PYTHON_EXE,
                                         PAYLOAD / PAYLOAD_LIB))
        builder.write(PAYLOAD / PAYLOAD_PYTHON / "Lib" / "site-packages" / PTH_NAME,
                      PTH_LINE + "\n")
        builder.copy(Path(uv), PAYLOAD / PAYLOAD_UV)
        for source, destination in payload_plan(exiftool):
            builder.copy(source, PAYLOAD / destination)

    manifest = build_manifest(version, exiftool=exiftool is not None,
                              tool_version=None if args.dry_run
                              else exiftool_version(exiftool))
    builder.write(PAYLOAD / wizard.MANIFEST_NAME,
                  json.dumps(manifest, indent=2, ensure_ascii=False))

    iscc = shutil.which("ISCC") or shutil.which("iscc") or args.iscc
    if not iscc:
        print("Inno Setup (ISCC.exe) was not found — the payload is staged at "
              f"{PAYLOAD}, compile it with: ISCC "
              f"{' '.join(iscc_command('ISCC', version)[1:])}")
        return 0 if args.dry_run else 1
    builder.run(iscc_command(iscc, version))

    target = installer_path(version)
    if signing_requested(flag=args.sign):
        builder.run(sign_command(target))
    if not args.dry_run:
        checksum = sha256(target)
        builder.write(target.with_suffix(".exe.sha256"), f"{checksum}  {target.name}\n")
        print(f"\n{target} ({target.stat().st_size / 1e6:.1f} MB)\nsha256: {checksum}")
        print("Unsigned on purpose — the download page must warn about SmartScreen and "
              "publish this checksum.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dry-run", action="store_true",
                        help="print every step and run none")
    parser.add_argument("--skip-payload", action="store_true",
                        help="reuse the staged payload, only recompile the installer")
    parser.add_argument("--exiftool", default=None,
                        help="the exiftool.exe (or its directory) to bundle; "
                             "default — the one on PATH")
    parser.add_argument("--no-exiftool", action="store_true",
                        help="build without it: metadata falls back to Pillow and the "
                             "wizard says so in as many words")
    parser.add_argument("--iscc", default=None,
                        help="path to Inno Setup's ISCC.exe if it is not on PATH")
    parser.add_argument("--sign", action="store_true",
                        help=f"run the signing step ({ENV_SIGN_CERT} and friends); "
                             "off by default — the release is unsigned")
    return parser


def main(argv: list[str] | None = None) -> int:
    return build(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
