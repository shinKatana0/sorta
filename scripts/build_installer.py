#!/usr/bin/env python3
"""F211 — build the Windows installer: the base tier whole, everything else offered.

    python scripts/build_installer.py                 # payload + installer
    python scripts/build_installer.py --dry-run       # print every step, run none
    python scripts/build_installer.py --no-exiftool   # build the Pillow-fallback variant
    python scripts/build_installer.py --sign          # ...and the signing step (opt-in)

What it makes, and why it looks like this
-----------------------------------------
The payload is a directory that is COPIED to wherever somebody installs the program:

    python\\      a standalone CPython, fetched by `uv python install`, plus the three
                 MSVC runtime libraries that torch and onnxruntime import (F218), plus a
                 `sitecustomize.py` that points it at the certifi set in lib\\ (F221)
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

The second watchdog, `payload_import_gaps`, is about FILES and not about running anything
(F218): it reads what every `*.dll` and `*.pyd` of the payload imports and requires each
name to be either in the payload or given by Windows. That is what catches a payload which
works on the build machine only because the machine happens to have something installed —
which is exactly how the MSVC runtime went missing for three releases.

The third, `payload_trust_gap`, is the same idea about the same class of defect (F221):
the payload has to carry the certificate set its interpreter is pointed at, or nothing
downloads on a machine whose root store Windows has not filled in yet. Whether TLS works
cannot be answered here — the build machine's store IS filled — so the proof is a suite
test against an empty root store, and last a clean virtual machine.

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
import mmap
import os
import re
import shutil
import stat
import struct
import subprocess
import sys
import tomllib
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sorta import wizard  # noqa: E402 — after the path insert, so a checkout works

ROOT = Path(__file__).resolve().parent.parent
PACKAGING = ROOT / "packaging" / "windows"
ISS = PACKAGING / "sorta.iss"
DIST = ROOT / "dist" / "windows"
PAYLOAD = DIST / "payload"
# Downloads that survive `--skip-payload` and a rebuilt payload: the redistributable is
# 25 MB and its version is pinned, so fetching it once per machine is the whole point.
CACHE = DIST / "cache"
# Where the three runtime libraries are unpacked before they are copied into the payload.
RUNTIME_STAGE = DIST / "msvc-runtime"

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
# F226: the other half of that binary. The Windows build of exiftool is the .exe PLUS a
# directory beside it holding `perl5xx.dll` and the whole Image::ExifTool tree; on its own
# the .exe prints `Could not find ...\exiftool_files\perl5*.dll` and exits 1. Only the
# .exe was ever copied, so the 25 MB the installer carried could not start even once it
# was found — which is why this travels, and why the reader's probe (`sorta.exif._starts`)
# asks a candidate for its version instead of asking whether the file is there.
EXIFTOOL_FILES_DIR = "exiftool_files"
PAYLOAD_EXIFTOOL_FILES = PAYLOAD_EXIFTOOL.with_name(EXIFTOOL_FILES_DIR)

# The directory of the shipped interpreter that CPython reads at startup: both the `.pth`
# below and the `sitecustomize.py` further down are found there, and both climb out of it
# by the same four levels.
PAYLOAD_SITE_PACKAGES = PAYLOAD_PYTHON / "Lib" / "site-packages"

# The one line that puts `lib\` on the path of the shipped interpreter. Relative to the
# .pth file's own directory (`python\Lib\site-packages`), which is what makes the whole
# payload movable.
PTH_NAME = "_sorta_lib.pth"
PTH_LINE = "..\\..\\..\\lib"

# F221 — the trust the shipped interpreter uses, and where it comes from. `sitecustomize`
# is imported by CPython out of `site-packages` before any of our code runs, on every one
# of the five ways this program starts, so it is the one place that can point OpenSSL at
# the certifi set the payload already carries. The file itself says why at length; what
# matters here is that BOTH halves have to be in the payload, which is what
# `payload_trust_gap` below checks before an installer is compiled.
SITECUSTOMIZE_SOURCE = "packaging/windows/sitecustomize.py"
PAYLOAD_SITECUSTOMIZE = PAYLOAD_SITE_PACKAGES / "sitecustomize.py"
PAYLOAD_CA_BUNDLE = PAYLOAD_LIB / "certifi" / "cacert.pem"

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
    # F221: a real file in the repository rather than a string generated here, for the
    # same reason `config.example.yaml` is one — a second copy written out by hand is the
    # copy that drifts, and this one is read by people trying to understand what the
    # shipped interpreter trusts.
    (SITECUSTOMIZE_SOURCE, str(PAYLOAD_SITECUSTOMIZE)),
)

# --- the MSVC runtime the payload has to carry (F218) --------------------------------
#
# The installer built before this shipped a payload that could not load torch on a clean
# Windows: `c10.dll` failed with WinError 126 and everything behind it — faces, CLIP,
# the whole machine half of the product — was dead. Reading the import table of all 439
# modules named three libraries:
#
#   msvcp140.dll               25 modules want it. It IS in the payload — sklearn ships
#                              a copy in lib\sklearn\.libs\ — but nothing looks there.
#   msvcp140_1.dll             onnxruntime wants it. Not in the payload at all.
#   msvcp140_atomic_wait.dll   torch_python wants it. Not in the payload at all.
#
# `vcruntime140.dll` and `vcruntime140_1.dll` were already fine, and the reason is the
# whole fix: standalone CPython puts them in `payload\python\`, the directory of the
# executable, which the Windows loader searches. So the three go there too — app-local
# deployment of the runtime, which Microsoft supports explicitly.
#
# NOT `vc_redist.x64.exe` run from the installer, the way almost everybody does it: that
# needs an administrator, and `PrivilegesRequired=lowest` is a promise the installer is
# built around (F211), not a decoration to drop over a missing file.
MSVC_RUNTIME_DLLS: tuple[str, ...] = (
    "msvcp140.dll", "msvcp140_1.dll", "msvcp140_atomic_wait.dll",
)

# Where they come from, and why this source and not another. A copy from the System32 of
# whoever ran the build is NOT a source: it pins the release to that machine's patch
# level and the next build is made by somebody else. This is the official
# redistributable, at a PERMANENT versioned URL (the `aka.ms/vs/17/release` shortcut
# always points at the newest one, so it is not reproducible), verified by checksum
# before it is opened. The version is what `sorta-install.json` records — the same
# obligation exiftool's version travels for.
VC_REDIST_VERSION = "14.44.35211"
VC_REDIST_URL = (
    "https://download.visualstudio.microsoft.com/download/pr/"
    "9b0d1fa5-c16d-4ee8-97f0-c2734086ece8/"
    "CC0FF0EB1DC3F5188AE6300FAEF32BF5BEEBA4BDD6E8E445A9184072096B713B/VC_redist.x64.exe"
)
VC_REDIST_SHA256 = "cc0ff0eb1dc3f5188ae6300faef32bf5beeba4bdd6e8e445a9184072096b713b"
# Inside the redistributable's cabinets every file carries the architecture it is for.
VC_REDIST_ARCH_SUFFIX = "_amd64"
# The signature and header of a WiX "burn" bundle, which vc_redist.x64.exe is.
BURN_SECTION = b".wixburn"
BURN_MAGIC = 0x00F14300
CAB_MAGIC = b"MSCF"


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


# The extra the desktop shortcut needs (F207). Carried by the build, not by a tier.
SHORTCUT_EXTRA = "tray"


def unaccounted_extras(pyproject_text: str) -> tuple[set[str], set[str]]:
    """(extras the installer never heard of, extras it names that no longer exist).

    Both directions matter and they fail differently: the first is a tier somebody
    added to the project and forgot here — the wizard would never offer it — and the
    second is the installer promising something the project cannot install any more.

    `SHORTCUT_EXTRA` is accounted for by this build rather than by a tier: the shortcut
    starts an icon and the payload carries pystray for it, while a machine without it has
    a whole base tier all the same.
    """
    declared = wizard.declared_extras() | {SHORTCUT_EXTRA}
    actual = project_extras(pyproject_text)
    return actual - declared, declared - actual


# --- the watchdog: the payload carries what it imports (F218) ------------------------
#
# Why a check about FILES rather than one about running the thing. On the build machine —
# and on `windows-latest`, which is a developer image with build tools on it — the MSVC
# runtime is in System32, put there by any one of a dozen unrelated programs. So "does it
# start on this machine" answers "does this machine have the runtime", not "is the payload
# complete", and F216's workflow would have gone green on the broken payload too. Reading
# the import tables needs no clean machine, no virtual machine and no runner, and it would
# have caught this before the first install.
#
# The PE parsing is about sixty lines of `struct` on purpose: `pefile` is a dependency
# this repository does not need, and reading the FILE rather than loading it means the
# check gives the same answer on Linux, which is where most of the suite runs.

MODULE_SUFFIXES = (".dll", ".pyd")

# What a module may import without the payload carrying it: Windows itself. The list is
# explicit and every entry says why it is here, because this list is where the check
# lives or dies — too soft and it passes on a broken payload, too strict and it goes red
# on every build until somebody switches it off.
SYSTEM_DLLS: dict[str, str] = {
    "ntdll.dll": "the native API; mapped into every process before it starts",
    "kernel32.dll": "the Win32 base API — processes, files, memory",
    "kernelbase.dll": "the lower half of kernel32, imported directly by newer toolchains",
    "advapi32.dll": "the registry and the security API",
    "sechost.dll": "the half of advapi32 that moved out of it in Windows 8",
    "rpcrt4.dll": "RPC, which COM is built on",
    "user32.dll": "windows and messages",
    "gdi32.dll": "the drawing API",
    "gdiplus.dll": "the drawing API Windows has shipped since XP",
    "ole32.dll": "the COM runtime",
    "combase.dll": "the half of COM that moved out of ole32",
    "oleaut32.dll": "the COM automation types (BSTR, VARIANT)",
    "oleacc.dll": "the accessibility API",
    "propsys.dll": "the shell property system",
    "shell32.dll": "the shell API — known folders",
    "shlwapi.dll": "the shell path helpers",
    "comdlg32.dll": "the common dialogs",
    "comctl32.dll": "the common controls",
    "version.dll": "the file version resource API",
    "psapi.dll": "process information",
    "userenv.dll": "the user profile API",
    "powrprof.dll": "the power API — what a thread pool asks about cores",
    "pdh.dll": "the performance counters",
    "setupapi.dll": "device enumeration",
    "cfgmgr32.dll": "the configuration manager, the newer half of setupapi",
    "cabinet.dll": "the cabinet reader Windows uses itself",
    "imm32.dll": "the input method API",
    "winmm.dll": "the multimedia timers",
    "winspool.drv": "the print spooler API",
    "mpr.dll": "the network share API",
    "ws2_32.dll": "the sockets library",
    "wsock32.dll": "the older sockets library, still shipped",
    "mswsock.dll": "the Winsock service provider",
    "iphlpapi.dll": "the IP helper API — interfaces and addresses",
    "dnsapi.dll": "the resolver",
    "netapi32.dll": "the network management API",
    "wldap32.dll": "the LDAP client",
    "winhttp.dll": "the HTTP client Windows ships",
    "wininet.dll": "the older HTTP client Windows ships",
    "urlmon.dll": "URL monikers",
    "normaliz.dll": "Unicode normalisation",
    "crypt32.dll": "certificates",
    "bcrypt.dll": "the CNG primitives — hashes and random",
    "bcryptprimitives.dll": "the implementation half of CNG",
    "ncrypt.dll": "CNG key storage",
    "secur32.dll": "SSPI, the authentication interface",
    "wintrust.dll": "signature verification",
    "dbghelp.dll": "stack walking and symbols — what a crash handler calls",
    "imagehlp.dll": "the older name of the same, and still a real Windows DLL",
    # The two C runtimes that ARE Windows, as opposed to the three this feature adds.
    # `ucrtbase.dll` is the universal C runtime: part of the operating system since
    # Windows 10, which is why nothing has to carry it. `msvcrt.dll` is the OS's own
    # copy, used by Windows itself and kept for compatibility.
    "ucrtbase.dll": "the universal C runtime — part of Windows since 10",
    "msvcrt.dll": "the OS's own C runtime, kept for compatibility",
    "dxgi.dll": "the DirectX device enumeration",
    "d3d11.dll": "Direct3D 11",
    "d3d12.dll": "Direct3D 12",
    "dxcore.dll": "the newer adapter enumeration DirectML asks for",
    "d3dcompiler_47.dll": "the shader compiler shipped with Windows",
    "opengl32.dll": "the OpenGL entry point Windows ships",
    "glu32.dll": "its companion utility library",
    "dwmapi.dll": "the desktop window manager",
    "uxtheme.dll": "the theming API",
    "avicap32.dll": "Video for Windows capture — opencv's cameras",
    "avifil32.dll": "Video for Windows files — opencv again",
    "msvfw32.dll": "the Video for Windows compressor interface",
    "mf.dll": "Media Foundation",
    "mfplat.dll": "the Media Foundation platform",
    "mfreadwrite.dll": "the Media Foundation source reader",
    "hid.dll": "the human interface device API",
    "winusb.dll": "the generic USB driver interface",
}

# Whole families rather than names, and this needs saying in words or the next reader
# decides somebody forgot to ship them: `api-ms-win-*` are the API SETS of Windows 10 and
# 11 — virtual DLL names the loader resolves to the real implementation inside the OS.
# `api-ms-win-crt-*` in particular is the universal C runtime, part of Windows since
# Windows 10, and a payload is not supposed to carry it. `ext-ms-win-*` are the same idea
# for extensions that are present on some editions and absent on others.
SYSTEM_DLL_PREFIXES: dict[str, str] = {
    "api-ms-win-": "a Windows API set; `api-ms-win-crt-*` is the universal C runtime, "
                   "which is part of Windows 10 and 11",
    "ext-ms-win-": "a Windows extension API set, resolved by the loader when present",
}


def is_system_dll(name: str) -> bool:
    """Does Windows provide this one, so that the payload need not carry it?"""
    lowered = name.lower()
    if lowered in SYSTEM_DLLS:
        return True
    return any(lowered.startswith(prefix) for prefix in SYSTEM_DLL_PREFIXES)


def _cstring(view, offset: int) -> str:
    end = view.find(b"\0", offset)
    return bytes(view[offset:end if end != -1 else offset]).decode("ascii", "replace")


def _section_table(view, first: int, count: int) -> list[tuple[int, int, int, int]]:
    """(virtual address, virtual size, raw offset, raw size) per section."""
    table = []
    for index in range(count):
        header = first + index * 40
        virtual_size, virtual_address, raw_size, raw_offset = struct.unpack_from(
            "<IIII", view, header + 8)
        table.append((virtual_address, virtual_size, raw_offset, raw_size))
    return table


def _file_offset(sections: list[tuple[int, int, int, int]], rva: int) -> int | None:
    """Where an address the loader would use sits in the file on disk."""
    for virtual_address, virtual_size, raw_offset, raw_size in sections:
        if virtual_address <= rva < virtual_address + max(virtual_size, raw_size):
            inside = rva - virtual_address
            if inside < raw_size:
                return raw_offset + inside
    return None


def _descriptor_names(view, sections, first: int, size: int, stride: int,
                      name_field: int, base: int, delayed: bool) -> list[str]:
    """Walk an array of import descriptors and collect the DLL names it points at."""
    names: list[str] = []
    offset = _file_offset(sections, first)
    if offset is None:
        return names
    while offset + stride <= size:
        fields = struct.unpack_from(f"<{stride // 4}I", view, offset)
        if not any(fields):
            break
        rva = fields[name_field]
        # A delay-load descriptor written before Visual Studio 2015 stores addresses the
        # image would have once loaded, not addresses relative to it — the low bit of its
        # attributes says which, and getting this backwards would read a name out of thin
        # air rather than out of the file.
        if delayed and not fields[0] & 1:
            rva -= base
        located = _file_offset(sections, rva) if rva > 0 else None
        if located is not None:
            names.append(_cstring(view, located))
        offset += stride
    return names


def imported_dlls(path: Path) -> list[str]:
    """Every DLL named by the import directory of a PE file, in the order they appear.

    Both directories are read: the ordinary one and the delay-load one, because a name
    that is only reached on the first call is still a name that has to be there when the
    call comes. Anything that is not a PE file — a stray text file with a `.dll` name, an
    empty stub — is not an error and returns nothing: this walks a whole payload, and one
    unreadable file must not stop it from checking the other four hundred.
    """
    if path.stat().st_size < 0x40:
        return []
    with path.open("rb") as handle:
        with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as view:
            size = len(view)
            if view[:2] != b"MZ":
                return []
            pe = struct.unpack_from("<I", view, 0x3C)[0]
            if pe + 24 > size or view[pe:pe + 4] != b"PE\0\0":
                return []
            coff = pe + 4
            section_count = struct.unpack_from("<H", view, coff + 2)[0]
            optional_size = struct.unpack_from("<H", view, coff + 16)[0]
            optional = coff + 20
            if optional + optional_size > size:
                return []
            magic = struct.unpack_from("<H", view, optional)[0]
            if magic == 0x20B:  # PE32+, which everything x64 is
                image_base = struct.unpack_from("<Q", view, optional + 24)[0]
                directory_count = optional + 108
            elif magic == 0x10B:  # PE32, still here for the 32-bit modules of old wheels
                image_base = struct.unpack_from("<I", view, optional + 28)[0]
                directory_count = optional + 92
            else:
                return []
            count = struct.unpack_from("<I", view, directory_count)[0]
            directories = directory_count + 4
            sections = _section_table(view, optional + optional_size, section_count)
            names: list[str] = []
            if count > 1:  # directory 1 — the ordinary imports
                rva = struct.unpack_from("<I", view, directories + 8)[0]
                if rva:
                    names += _descriptor_names(view, sections, rva, size, 20, 3,
                                               image_base, delayed=False)
            if count > 13:  # directory 13 — the delay-loaded ones
                rva = struct.unpack_from("<I", view, directories + 13 * 8)[0]
                if rva:
                    names += _descriptor_names(view, sections, rva, size, 32, 1,
                                               image_base, delayed=True)
            return names


def payload_modules(payload: Path) -> list[Path]:
    """Every module of the payload whose imports have to be accounted for."""
    return sorted(item for item in payload.rglob("*")
                  if item.is_file() and item.suffix.lower() in MODULE_SUFFIXES)


def payload_import_gaps(payload: Path) -> list[tuple[Path, str]]:
    """(module, name) for every import the payload neither carries nor gets from Windows.

    "Carries" is by file name, anywhere in the payload, and that is deliberate rather
    than lax: numpy and shapely ship their own copies of the runtime under mangled names
    (`msvcp140-a4c2229b….dll`), those copies are found through the directory their own
    package adds at import time, and a check that insisted on one canonical location
    would go red on every build because of them. Where a library sits so that the loader
    finds it is fixed by putting the runtime in `python\\` — the directory of the
    executable; what this answers is the other half, whether the payload contains it at
    all. That is the half nobody was checking.
    """
    carried = {item.name.lower() for item in payload.rglob("*") if item.is_file()}
    gaps: list[tuple[Path, str]] = []
    for module in payload_modules(payload):
        for name in imported_dlls(module):
            if name.lower() not in carried and not is_system_dll(name):
                gaps.append((module.relative_to(payload), name))
    return gaps


# --- the watchdog: the payload carries the trust it points at (F221) ------------------
#
# The same class of defect as F218 and checked the same way, about FILES: the product
# relied on the state of the host machine — there, on the Visual C++ runtime being in
# System32; here, on Windows' root certificate store being filled — while promising not
# to. `sitecustomize.py` points OpenSSL at `lib\certifi\cacert.pem`, and it does so
# silently when the file is absent, because it runs at the start of every process
# including the tray, which has no console. So the loud half is here: an installer is not
# compiled from a payload where either end of that pair is missing.
#
# What this canNOT answer is whether TLS works, and nothing on this machine can: the build
# machine's root store is full, so a successful download here proves only that the machine
# is not the owner's clean one. That half is the suite (an empty root store, in
# tests/test_payload_carries_its_trust.py) and, last, a clean virtual machine.


def payload_trust_gap(payload: Path) -> str | None:
    """Why the payload's TLS trust would not work, in one sentence — or None if it would.

    Both ends are named because they fail identically and are fixed differently: without
    `sitecustomize.py` nothing points the interpreter anywhere, and without the certifi
    set it points at a file that is not there — and in both cases every download falls
    back to the machine's own root store, which is the defect.
    """
    if not (payload / PAYLOAD_SITECUSTOMIZE).is_file():
        return (f"{PAYLOAD_SITECUSTOMIZE} is missing — nothing would point the shipped "
                f"interpreter at the certificate set the payload carries, and every "
                f"download would fall back to the machine's own root store")
    if not (payload / PAYLOAD_CA_BUNDLE).is_file():
        return (f"{PAYLOAD_CA_BUNDLE} is missing — certifi travels as a dependency of "
                f"requests and huggingface_hub, so a payload without it means the base "
                f"tier was not installed the way this build expects")
    return None


# --- the commands (returned, not run — this is the checkable half) -------------------


def python_install_command(uv: str, destination: Path,
                           version: str = PYTHON_VERSION) -> list[str]:
    """Fetch the standalone CPython that ships inside the payload.

    `--no-bin` and `--no-registry` because this is a PAYLOAD and not an installation:
    the build machine must not end up with a shim on its PATH or an entry in its registry
    pointing into `dist/`.
    """
    return [uv, "python", "install", "--install-dir", str(destination),
            "--no-bin", "--no-registry", version]


def _is_alias(path: Path) -> bool:
    """Is this another NAME for a directory rather than the directory itself?

    A junction is what uv leaves beside the interpreter it installed, and Windows
    junctions are not symlinks: `is_symlink()` says no to them. `Path.is_junction` says
    yes and arrived in 3.12, so under 3.11 the reparse tag is read directly — the
    earlier `getattr` probe answered "not an alias" there, which made the build refuse a
    perfectly good download with "expected exactly one python installation" (caught by
    the py3.11 runner, 2026-08-09; this project supports 3.11).
    """
    if path.is_symlink():
        return True
    probe = getattr(path, "is_junction", None)
    if probe is not None:
        return bool(probe())
    return _has_mount_point_tag(path)


def _has_mount_point_tag(path: Path) -> bool:
    """The junction test `Path.is_junction` performs, for interpreters without it."""
    try:
        tag = getattr(os.lstat(path), "st_reparse_tag", 0)
    except OSError:
        return False
    return tag == getattr(stat, "IO_REPARSE_TAG_MOUNT_POINT", None)


def flatten_python_install(install_dir: Path) -> Path:
    """Lift uv's versioned directory up one level, and return the interpreter.

    `uv python install --install-dir X` writes `X/cpython-3.13.x-windows-x86_64-none/`,
    and the payload needs ONE fixed path: the manifest, the `.pth` and the shortcuts all
    name `python\\python.exe`, and a path with a patch version in it would have to be
    rewritten by three different files on every upgrade.

    uv 0.11 puts TWO names in there: the real `cpython-3.13.14-...` directory and a
    `cpython-3.13-...` junction beside it, so that a minor version can be named without
    knowing its patch. Both answer `is_dir()` and both hold a `python.exe`, so the
    aliases are separated out rather than counted — two names for one installation are
    one installation, and an alias is removed before its target is emptied so that the
    payload never carries a link pointing at nothing.
    """
    direct = install_dir / "python.exe"
    if direct.is_file():
        return direct
    candidates = [child for child in sorted(install_dir.iterdir())
                  if child.is_dir() and (child / "python.exe").is_file()]
    aliases = [child for child in candidates if _is_alias(child)]
    staged = [child for child in candidates if child not in aliases]
    if len(staged) != 1:
        raise SystemExit(f"expected exactly one python installation in {install_dir}, "
                         f"found {[child.name for child in staged]} "
                         f"(aliases: {[child.name for child in aliases]})")
    for alias in aliases:
        # A junction comes off with rmdir; a POSIX symlink to a directory does not, and
        # answers NotADirectoryError. The payload is Windows-only, but the tests are not
        # — this ran green here and red on the Linux runner (2026-08-09).
        alias.unlink() if alias.is_symlink() else alias.rmdir()
    for item in list(staged[0].iterdir()):
        shutil.move(str(item), str(install_dir / item.name))
    staged[0].rmdir()
    return direct


def base_install_command(uv: str, python: Path, target: Path,
                         project: Path = ROOT) -> list[str]:
    """Install the base tier into the payload — the project with the base tier's extras.

    The hardware profile comes from the tier catalog; `tray` is added here because the
    shortcut starts an icon and this build has to carry it. It is deliberately NOT part
    of the base tier: a machine without pystray still has a working base tier, and saying
    otherwise is what made a correct Linux install report the base tier as missing.
    """
    extras = ",".join((*wizard.BASE_TIER.extras, SHORTCUT_EXTRA))
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


# --- the MSVC runtime, fetched reproducibly ------------------------------------------


def verified_download(url: str, destination: Path, checksum: str) -> Path:
    """Download once, and refuse to use anything whose checksum is not the pinned one.

    Cached by file name: the version is in the constant above, so a cached copy is the
    same copy. The checksum is verified on the CACHED file too — a truncated download
    left behind by an interrupted build is exactly the case this must not wave through.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.is_file():
        print(f"  download {url}\n        -> {destination}")
        with urllib.request.urlopen(url, timeout=300) as response:  # noqa: S310 — pinned
            destination.write_bytes(response.read())
    actual = sha256(destination)
    if actual != checksum.lower():
        raise SystemExit(f"{destination}: sha256 is {actual}, expected {checksum.lower()} "
                         f"— delete it and let the build fetch it again")
    return destination


def burn_attached_container(data: bytes) -> tuple[int, int]:
    """Where the payload cabinet of a WiX bundle starts, and how long it is.

    `vc_redist.x64.exe` is a "burn" bundle: a small executable with two cabinets glued
    behind it — the user interface, then the packages. `/layout` only copies the bundle,
    `expand` does not recognise it, and the offset is not guessable, so it is read from
    the `.wixburn` section the linker writes for exactly this purpose: the size of the
    stub, then the size of each container. The cabinet's own header carries its length,
    which is what is returned rather than the container size — the authenticode signature
    is appended after the containers and must not travel into the cabinet.
    """
    pe = struct.unpack_from("<I", data, 0x3C)[0]
    coff = pe + 4
    section_count = struct.unpack_from("<H", data, coff + 2)[0]
    sections = coff + 20 + struct.unpack_from("<H", data, coff + 16)[0]
    for index in range(section_count):
        header = sections + index * 40
        if data[header:header + 8].rstrip(b"\0") != BURN_SECTION.rstrip(b"\0"):
            continue
        start = struct.unpack_from("<I", data, header + 20)[0]
        magic, _version = struct.unpack_from("<II", data, start)
        if magic != BURN_MAGIC:
            raise SystemExit("the .wixburn section does not begin with the burn magic — "
                             "this is not the redistributable this build expects")
        # After the magic, the version and the bundle's GUID come the stub size, the
        # original checksum, the signature's offset and size, the format, and then one
        # length per container.
        fixed = start + 8 + 16
        stub, _checksum, _offset, _size, _format, containers = struct.unpack_from(
            "<IIIIII", data, fixed)
        lengths = struct.unpack_from(f"<{containers}I", data, fixed + 24)
        # Container 0 is the user interface; the packages are the one after it.
        cabinet = data.find(CAB_MAGIC, stub + lengths[0])
        if cabinet == -1:
            raise SystemExit("no cabinet behind the bundle's stub")
        return cabinet, struct.unpack_from("<I", data, cabinet + 8)[0]
    raise SystemExit("no .wixburn section — this is not a WiX bundle")


def expand_binary() -> str:
    """Windows' own `expand.exe`, by full path.

    By name it would be found on PATH, and on a machine where the build runs from a
    POSIX shell that is a completely different program (`expand` turns tabs into spaces)
    which cheerfully copies the cabinet and reports success.
    """
    return str(Path(os.environ.get("SystemRoot", "C:\\Windows")) / "System32" / "expand.exe")


def extract_msvc_runtime(redist: Path, stage: Path,
                         names: tuple[str, ...] = MSVC_RUNTIME_DLLS) -> Path:
    """Unpack the three runtime libraries out of the redistributable; return their directory.

    Two cabinets deep: the bundle's attached container holds one package per
    architecture, and the x64 runtime package holds the libraries under names with the
    architecture appended (`msvcp140.dll_amd64`).
    """
    if stage.exists():
        shutil.rmtree(stage)
    packages = stage / "packages"
    packages.mkdir(parents=True)
    offset, length = burn_attached_container(redist.read_bytes())
    container = stage / "packages.cab"
    with redist.open("rb") as handle:
        handle.seek(offset)
        container.write_bytes(handle.read(length))
    subprocess.run([expand_binary(), str(container), "-F:*", str(packages)],
                   capture_output=True, check=True)
    for package in sorted(packages.iterdir()):
        with package.open("rb") as handle:  # these run to eleven megabytes; read four
            if handle.read(4) != CAB_MAGIC:
                continue  # the MSIs travel beside the cabinets and hold nothing we want
        for name in names:
            if (stage / name).is_file():
                continue
            packed = name + VC_REDIST_ARCH_SUFFIX
            subprocess.run([expand_binary(), str(package), f"-F:{packed}", str(stage)],
                           capture_output=True, check=False)
            if (stage / packed).is_file():
                (stage / packed).replace(stage / name)
    missing = [name for name in names if not (stage / name).is_file()]
    if missing:
        raise SystemExit(f"the redistributable {redist} does not hold {missing} — the "
                         f"pinned version or the naming inside it has changed")
    shutil.rmtree(packages)
    container.unlink()
    return stage


def msvc_runtime(cache: Path = CACHE, stage: Path = RUNTIME_STAGE) -> Path:
    """The three libraries on disk, downloaded and unpacked if they are not there yet."""
    redist = verified_download(VC_REDIST_URL,
                               cache / f"VC_redist.x64-{VC_REDIST_VERSION}.exe",
                               VC_REDIST_SHA256)
    return extract_msvc_runtime(redist, stage)


# --- the payload ---------------------------------------------------------------------


def payload_plan(exiftool: Path | None, runtime: Path | None = None,
                 root: Path = ROOT) -> list[tuple[Path, Path]]:
    """(source, destination-inside-the-payload) for everything copied as it is."""
    plan = [(root / source, Path(destination)) for source, destination in STATIC_PAYLOAD]
    if exiftool is not None:
        plan.append((exiftool, PAYLOAD_EXIFTOOL))
        # Unconditionally, and not "if it happens to be there": a build machine whose
        # exiftool has no `exiftool_files\` beside it would otherwise ship the same
        # non-starting binary again, silently. `Builder.copy` refuses a source that is
        # missing, so that build stops instead.
        plan.append((exiftool.parent / EXIFTOOL_FILES_DIR, PAYLOAD_EXIFTOOL_FILES))
    if runtime is not None:
        # Beside `vcruntime140.dll`, which standalone CPython put there and which IS
        # found from there — the directory of the executable is on the loader's path.
        plan += [(runtime / name, PAYLOAD_PYTHON / name) for name in MSVC_RUNTIME_DLLS]
    return plan


def tier_summary() -> list[dict]:
    """The tiers as the download page and the release notes state them."""
    return [
        {"key": tier.key, "extras": list(tier.extras), "weights": list(tier.weights),
         "download_mb": tier.download_mb, "optional": tier.optional,
         # F223: what a tier does not work without, and what Enter answers for it. A
         # summary that states the price but not those two describes a catalog the wizard
         # no longer has — and this file is what the download page reads. (F225 dropped
         # `preload` with the field: every tier that carries weights downloads them at
         # the screen now, so there was nothing left for it to distinguish.)
         "requires": list(tier.requires), "default_yes": tier.default_yes}
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
                   python_version: str = PYTHON_VERSION,
                   msvc_runtime_version: str = VC_REDIST_VERSION) -> dict:
    """What the installer leaves next to the program for the wizard to read.

    Paths are relative to this file's own directory: the payload is built here and
    installed somewhere else, and a relative path needs no install-time rewriting to
    survive that (`wizard.manifest_path_of` resolves them).
    """
    return {
        "version": version,
        "python_version": python_version,
        # Which release of the MSVC runtime travels in `python\` — the same obligation
        # exiftool's version travels for: a library bundled without its version recorded
        # is one nobody can tell whether they have to update.
        "msvc_runtime_version": msvc_runtime_version,
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


def exiftool_home(candidate: Path) -> Path | None:
    """The directory holding both the binary and `exiftool_files\\`, or None.

    Not always the directory the binary is in. Chocolatey puts a shim on PATH and the
    real thing under `lib\\exiftool\\tools`, so `shutil.which` answers with a wrapper
    that has no `exiftool_files\\` beside it — bundling that ships a binary which cannot
    start. Caught 2026-08-09 by the installer workflow, the first machine to build here
    that was not the developer's.
    """
    if (candidate.parent / EXIFTOOL_FILES_DIR).is_dir():
        return candidate.parent
    # Searched for by name rather than at a guessed depth: the runner keeps it at
    # `lib\exiftool\tools\exiftool-13.59_64\`, and a version in a path is a thing that
    # moves. Scoped to the package directory, so this is a handful of stat calls.
    for package in sorted(candidate.parent.parent.glob("lib/exiftool*")):
        for files in sorted(package.rglob(EXIFTOOL_FILES_DIR)):
            if files.is_dir() and any(files.parent.glob("exiftool*.exe")):
                return files.parent
    return None


def find_exiftool(explicit: str | None) -> Path | None:
    """The exiftool binary to bundle: what was named, or what is on PATH."""
    if explicit:
        candidate = Path(explicit)
        if candidate.is_dir():
            candidate = candidate / "exiftool.exe"
    else:
        found = shutil.which("exiftool")
        if not found:
            return None
        candidate = Path(found)
    home = exiftool_home(candidate)
    if home is None or home == candidate.parent:
        return candidate
    named = home / candidate.name
    return named if named.exists() else next(iter(sorted(home.glob("exiftool*.exe"))))


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
    if exiftool is not None and exiftool_home(exiftool) is None:
        print(f"{exiftool} has no {EXIFTOOL_FILES_DIR}\\ beside it, and that directory is "
              "most of exiftool: the executable alone starts and immediately says it "
              "cannot find its Perl library. Point --exiftool at the directory holding "
              "both (a Chocolatey shim on PATH is not it).")
        return 1

    if not args.skip_payload:
        if not args.dry_run and PAYLOAD.exists():
            shutil.rmtree(PAYLOAD)
        builder.run(python_install_command(uv, PAYLOAD / PAYLOAD_PYTHON))
        if not args.dry_run:
            flatten_python_install(PAYLOAD / PAYLOAD_PYTHON)
        builder.run(base_install_command(uv, PAYLOAD / PAYLOAD_PYTHON_EXE,
                                         PAYLOAD / PAYLOAD_LIB))
        builder.write(PAYLOAD / PAYLOAD_SITE_PACKAGES / PTH_NAME, PTH_LINE + "\n")
        builder.copy(Path(uv), PAYLOAD / PAYLOAD_UV)
        # The MSVC runtime, from the pinned redistributable rather than from the System32
        # of whoever is building (F218). A dry run says what would be fetched and fetches
        # nothing, so the staging directory is named but not filled.
        print(f"  msvc runtime {VC_REDIST_VERSION} <- {VC_REDIST_URL}")
        runtime = RUNTIME_STAGE if args.dry_run else msvc_runtime()
        for source, destination in payload_plan(exiftool, runtime):
            builder.copy(source, PAYLOAD / destination)

    manifest = build_manifest(version, exiftool=exiftool is not None,
                              tool_version=None if args.dry_run
                              else exiftool_version(exiftool))
    builder.write(PAYLOAD / wizard.MANIFEST_NAME,
                  json.dumps(manifest, indent=2, ensure_ascii=False))

    if not args.dry_run:
        # The second watchdog, before a byte is compiled: an incomplete payload must not
        # reach Inno Setup. This is the check that would have caught the missing runtime
        # before the first install rather than in somebody's clean virtual machine.
        modules = payload_modules(PAYLOAD)
        if not modules:
            print(f"no modules under {PAYLOAD} — nothing is staged there, and a "
                  f"completeness check over nothing proves nothing")
            return 1
        gaps = payload_import_gaps(PAYLOAD)
        if gaps:
            print(f"\nthe payload does not carry what it imports "
                  f"({len(gaps)} unresolved import(s)):")
            for module, name in gaps:
                print(f"  {module} imports {name}")
            print("Every name above is neither inside the payload nor provided by "
                  "Windows. Either it has to travel, or — if Windows really does give "
                  "it — SYSTEM_DLLS in this script has to say so, and say why.")
            return 1
        print(f"payload complete: {len(modules)} modules, every import either carried "
              f"or provided by Windows")
        # The third watchdog (F221): the payload has to carry its own TLS trust, or
        # nothing downloads on a machine whose root store Windows has not filled yet.
        trust = payload_trust_gap(PAYLOAD)
        if trust:
            print(f"\nthe payload does not carry the trust it points at:\n  {trust}")
            return 1
        print(f"payload trust: {PAYLOAD_SITECUSTOMIZE} points at {PAYLOAD_CA_BUNDLE}")

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
