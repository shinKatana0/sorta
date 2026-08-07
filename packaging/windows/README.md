# The Windows installer

`sorta-<version>-setup.exe` carries the **base tier** whole and offers the heavier ones
once, by name and by price. This directory holds the Inno Setup script; everything that
can be decided before a byte is downloaded lives in `scripts/build_installer.py` and is
checked by `tests/test_build_installer.py`.

## Building it

```powershell
# what the build needs on the machine: uv, Inno Setup 6 (ISCC.exe), and exiftool
python scripts/build_installer.py --dry-run          # print every step, run none
python scripts/build_installer.py --exiftool C:\tools\exiftool
python scripts/build_installer.py --no-exiftool      # the Pillow-fallback variant
```

The payload is a directory that is COPIED to wherever somebody installs the program:

```
python\      a standalone CPython, fetched by `uv python install`, and beside it the
             three MSVC runtime libraries torch and onnxruntime import; in its
             `Lib\site-packages\` the `.pth` that finds `lib\` and the `sitecustomize.py`
             that points TLS at the certificates in it
lib\         the packages, from `uv pip install --target` — a plain tree, not a venv
uv.exe       the same resolver, kept for the tiers the wizard offers later
exiftool\    the metadata reader and its `exiftool_files\`
favicon.ico, config.example.yaml, LICENSE, NOTICE
sorta-install.json   what was shipped, in paths relative to itself
```

`--target` and not a virtualenv, deliberately: a venv records the absolute path of the
interpreter it was created from, so it cannot be built here and installed there, while a
target tree records nothing at all and is found by one `.pth` file.

## exiftool is bundled, and why

Without it the metadata reader falls back to Pillow, which reads JPEG/PNG/TIFF/WEBP and
no video — an iPhone library then arrives with no dates and no GPS, which is precisely
the trimmed install the tiers exist to avoid. So the binary travels inside the payload
(~25 MB, free licence, attributed in `NOTICE` §3), together with its `exiftool_files\`
directory: the Windows build of exiftool is an executable **plus** that directory, and
the executable alone does not start.

Bundling a binary means owing an update to whoever installed it, and an obligation
nobody can see the state of is not one that gets met — so `exiftool -ver` is recorded in
`sorta-install.json` at build time. A build made on purpose without it (`--no-exiftool`)
is not a broken build: the manifest says so and the wizard then SAYS which formats will
not be read.

## The MSVC runtime travels in `python\`, and a watchdog says whether it does

The installer built before F218 could not load `torch` on a clean Windows 11. The first
install into a fresh virtual machine got:

```
WinError 126: the specified module could not be found.
Error loading ...\Sorta\lib\torch\lib\c10.dll or one of its dependencies
```

Reading the import table of all 439 modules of the payload named three libraries of the
Microsoft Visual C++ runtime:

```
msvcp140.dll                25 modules want it. It WAS in the payload — sklearn ships a
                            copy in lib\sklearn\.libs\ — but nothing looks there.
msvcp140_1.dll              onnxruntime wants it. Not in the payload at all.
msvcp140_atomic_wait.dll    torch_python wants it. Not in the payload at all.
```

The defect is wider than torch: `msvcp140_1.dll` is onnxruntime's, so faces and CLIP were
dead too. What survived was everything that imports neither — index, EXIF, geo,
duplicates, `doctor`. The installer shipped a product whose machine half was silently
gone, and nobody noticed because the Visual C++ Redistributable is on every development
machine, put there by a dozen unrelated programs.

**The fix is where the loader already looks.** `vcruntime140.dll` and `vcruntime140_1.dll`
were fine all along, because standalone CPython puts them in `payload\python\` — the
directory of the executable, which the Windows loader searches. So the three go there
too. That is app-local deployment of the runtime, which Microsoft supports explicitly.

**Not `vc_redist.x64.exe` run from the installer**, the way almost everybody does it: it
needs an administrator, and `PrivilegesRequired=lowest` is a promise this installer is
built around (F211), not a decoration to drop over a missing file.

**And not a copy out of the build machine's `System32`**, which would pin the release to
that machine's patch level and be unreproducible the moment somebody else builds. The
build downloads the official `vc_redist.x64.exe` from a permanent, versioned Microsoft
URL (`aka.ms/vs/17/release/...` is always the newest one and therefore not a pin),
verifies its SHA-256, reads the offset of the packages cabinet out of the bundle's
`.wixburn` section, and unpacks the three files with Windows' own `expand.exe`. Nothing
is executed, nothing needs administrator, and the version lands in `sorta-install.json`
beside exiftool's. The redistributable is cached in `dist\windows\cache\` — an offline
build machine can be given the file by hand, and the checksum is checked either way.
Attribution is in `NOTICE` §4.

### The watchdog: the payload carries what it imports

`build_installer.py` reads the import directory of every `*.dll` and `*.pyd` of the
staged payload and requires each imported name to be either **inside the payload** or
**provided by Windows**. It runs in the build — an incomplete payload never reaches Inno
Setup — and again in the test suite, so that an edit to the list of system libraries
cannot pass in silence.

This is deliberately a check about FILES rather than about starting the program, and that
is the whole lesson of this feature. The build machine has the runtime in System32, and so
does `windows-latest` — it is a developer image. "Does it run on the runner" answers
whether the RUNNER has the runtime, not whether the payload is complete, so the F216
workflow would have gone green on the broken payload too. Reading the files needs no
clean machine, no virtual machine and no runner, and it would have caught this before the
first install.

Two things it is worth knowing about the check:

- **The list of system libraries is the check.** Too soft and it passes on a broken
  payload; too strict and it goes red on every build until somebody switches it off. It
  is kept explicit in `SYSTEM_DLLS`, with one line per name saying why it is there, and
  the families `api-ms-win-*` and `ext-ms-win-*` are Windows API sets —
  `api-ms-win-crt-*` in particular is the universal C runtime, which is **part of Windows
  10 and 11** and is not something a payload should carry. `msvcp140.dll` is deliberately
  NOT on that list, and a test pins that: put it there and today's defect passes again.
- **"Carried" means the name is in the payload, anywhere.** numpy and shapely ship their
  own copies of the runtime under mangled names (`msvcp140-a4c2229b….dll`) that only
  their own package's directory makes findable, and a check insisting on one canonical
  location would go red on every build because of them. WHERE a library sits so the
  loader finds it is fixed by putting the runtime in `python\`; what the watchdog answers
  is the other half — whether the payload contains it at all. That half was unchecked.

The proof that it can go red is in `tests/test_payload_carries_what_it_imports.py`, which
assembles a payload with a module importing a name nothing carries and reads the
complaint back. We have shipped checks that pass no matter what before (F182, F216).

## The payload carries its own trust, not the machine's

The installer built before F221 could not download anything at all on a clean Windows.
The first run in a fresh virtual machine stopped at the verdicts stage:

```
stage failed: verdicts
Processing error: <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED]
  certificate verify failed: unable to get local issuer certificate (_ssl.c:1032)>
```

The weights go through `urllib.request.urlopen`, and on Windows urllib verifies the chain
against the **system root certificate store**. On a freshly installed Windows that store
is nearly empty — Windows fetches roots on demand, and on a clean machine that regularly
does not happen. The stage that failed is the one that builds CLIP, but the defect is
wider than any stage: faces, search by words and the deep tier download the same way, so
**no tier could be added at all** and the whole tiered construction was unreachable on
the machine it was built for.

`certifi` was in the payload already — `requests` and `huggingface_hub` depend on it and
use it by default, which is why those two paths always worked. Plain `urllib` had simply
never heard of it. So the payload now points its own interpreter at that same set:

```
python\Lib\site-packages\sitecustomize.py   sets SSL_CERT_FILE and SSL_CERT_DIR to
                                            ..\..\..\lib\certifi\cacert.pem
```

Four things about that choice, because each of them is a way the fix could have been
narrower than the defect:

- **certifi and not `truststore`, and not filling the system store.** certifi's set is
  self-contained and versioned with the delivery, so it does not depend on the state of
  the machine at all — the same principle the missing MSVC runtime was fixed with.
  `truststore` does the opposite: it defers to the operating system's store, and the
  operating system's store is exactly what is empty here.
- **`sitecustomize` and not a shortcut, a launcher or an entry point.** There are five
  ways this program starts — the two Start-menu shortcuts, `sorta-setup`, the console
  command and the tray — and a variable set in one of them fixes one route out of five.
  CPython imports `sitecustomize` out of `site-packages` while it is still starting up,
  before a line of our code runs, on every one of the five. It also touches nothing
  outside the process: `os.environ` there is that interpreter's own, and nothing is
  written to the machine.
- **`setdefault` and not assignment.** A corporate proxy with a root of its own is an
  ordinary thing, and somebody who has already named their own set keeps it.
- **A path derived from `__file__`.** The payload is built here and copied to somebody
  else's disk — the same reason `lib\` is found by a relative `.pth`.

**Certificate verification is not weakened anywhere.** No `verify=False`, no
`ssl._create_unverified_context`, not even "just for the download": a program that
downloads and then RUNS model weights has to know where they came from. A test pins the
absence of every one of those names.

### The guard: an empty root store, not a working network

The defect lived because **checking whether TLS works on a machine where TLS works proves
nothing.** The build machine's root store is full, and `windows-latest` is a developer
image — both would have gone green on the broken payload, exactly the way they did for
the missing MSVC runtime.

So `tests/test_payload_carries_its_trust.py` does not check the network. It starts an
interpreter whose root store is **empty** (`ssl.enum_certificates` returning nothing is
precisely a clean machine's store), puts one known certificate in a payload-shaped
directory, and reads back what `ssl.create_default_context()` ends up trusting: with the
shipped `sitecustomize.py` on the path it is exactly that one certificate, and with it
taken away it is whatever the machine has. It also pins that a set the person configured
themselves survives, and that a payload moved to another directory points at its own copy.

Beside it, the build refuses to compile an installer whose payload is missing either half
of the pair (`payload_trust_gap`), and the workflow checks after installing that
`SSL_CERT_FILE` points inside the installation. **Neither of those is a test of TLS** —
the runner is not a clean machine either. The last word is a clean virtual machine, step 6
of the checklist below.

### Until a build with this in it exists

The workaround the owner verified, for an installation that already has the defect:

```powershell
$env:SSL_CERT_FILE = "$env:LOCALAPPDATA\Programs\Sorta\lib\certifi\cacert.pem"
```

It is per-shell and therefore per-route, which is the whole reason it is a workaround and
not the fix.

## Unsigned, on purpose

The owner's decision of 2026-08-06: this release ships without a code-signing
certificate. Windows SmartScreen greets an unsigned installer with "Windows protected
your PC", and a person who was warned beforehand reads that as *the author has no
certificate* while one who was not reads it as *this program is dangerous*. So the
README, the guides and the release page say it before the download, and publish the
`sha256` the build writes beside the file (`Get-FileHash sorta-<version>-setup.exe`).

Signing is a separate, opt-in step (`--sign`, or `SORTA_SIGN_INSTALLER=1`): when a
certificate appears it plugs in and nothing above it changes.

## What the machine checks: `.github/workflows/installer.yml`

An installer is the one part of the product that cannot be looked at from the sources,
so it is not enough for it to compile. On `windows-latest`, on changes to `packaging/**`,
`scripts/build_installer.py`, `scripts/verify_installation.py` and `sorta/wizard.py`, by
hand (`workflow_dispatch`) and **always on a tag**, the workflow:

1. builds the payload and compiles the installer;
2. installs it with `/VERYSILENT` into a per-user directory, with no administrator;
3. makes the product **do work** — `scripts/verify_installation.py` writes a handful of
   synthetic frames and drives the installed copy through `index`, `stats` and `doctor`,
   checking the OUTPUT and not only the exit code (a command that found nothing also
   exits zero);
4. proves the checker can go red, by running it against an empty install directory
   (`--expect-failure`);
5. adds two tiers for real through the shipped `uv`: **faces** (the buffalo_l weights,
   ~400 MB, downloaded by the stage that needs them) and the **packages** of the deep
   tier, and then runs the stage that needs each of them.

Run the same checker by hand against an install you made yourself:

```powershell
python scripts\verify_installation.py --install-dir "$env:LOCALAPPDATA\Programs\Sorta"
python scripts\verify_installation.py --install-dir "$env:LOCALAPPDATA\Programs\Sorta" --tier faces
```

### What that green tick does NOT cover

Named here so nobody reads it wider than it is:

- **The deep tier's weights.** Qwen2.5-VL-3B is 7 GB — it fits neither the disk nor the
  time of a runner. The workflow installs the deep tier's PACKAGES and stops there; that
  the 7 GB download and the VLM stage work is not checked by any machine.
- **The GPU tier.** A runner has no NVIDIA card, so the CUDA profile is never installed
  and never exercised.
- **The tray icon as a picture.** A runner has no desktop. That the module comes up, that
  the menu is built from two items and that a failing backend still leaves the app
  serving is checked by F207 against a fake backend; that a person SEES the icon in the
  notification area is the last metre, and it is a human one.
- **SmartScreen.** What an unsigned installer looks like to somebody downloading it can
  only be seen by downloading it.
- **A machine without the Visual C++ Redistributable.** `windows-latest` is a developer
  image and almost certainly has it, so the runner cannot tell whether the payload's own
  copy is doing the work or System32's is. What the workflow proves about F218 is that
  the completeness watchdog passed; that the three libraries are enough is proven in a
  clean virtual machine, by hand, at step 5 of the checklist below.
- **A machine whose root certificate store Windows has not filled in.** Same shape of
  gap, for F221: the runner downloads 400 MB of weights successfully, which says the
  RUNNER's store is populated and nothing about a clean one. The workflow checks the
  wiring — that `SSL_CERT_FILE` points at a file inside the installation — and the suite
  checks the behaviour against an empty store; the machine itself is step 6 below.

## The manual checklist (a clean machine, once per release)

The things above, plus everything the workflow cannot judge:

1. On a **clean machine** (or a fresh VM snapshot), download the installer and go
   through SmartScreen the way a person does: *More info* → *Run anyway*.
2. Install without administrator rights. Check `%LOCALAPPDATA%\Programs\Sorta` and
   `%APPDATA%\sorta\config.yaml`.
3. The wizard opens at the end. **Refuse every tier** and read what it says: what is on
   the machine has to be described as a working product, not a stub.
4. The Start menu **shortcut** opens the web app with an icon in the notification area
   and no console window behind it. Right-click → Quit closes the program.
5. On a machine that has **never had the Visual C++ Redistributable** (a fresh Windows
   snapshot, before anything else is installed on it), import the two libraries whose
   absence F218 was about — this is the check no runner can stand in for:

   ```powershell
   & "$env:LOCALAPPDATA\Programs\Sorta\python\python.exe" -c "import torch, onnxruntime; print(torch.__version__, onnxruntime.__version__)"
   ```

   A `WinError 126` here means the payload is missing a runtime library again; the
   watchdog in the build will name which one.
6. On that **same never-touched machine**, before installing anything else on it, add one
   tier and let it download — this is the check no runner can stand in for either, because
   a runner's root certificate store is full:

   ```powershell
   & "$env:LOCALAPPDATA\Programs\Sorta\python\python.exe" -X utf8 -m sorta.wizard --tiers faces
   & "$env:LOCALAPPDATA\Programs\Sorta\python\python.exe" -X utf8 -m sorta.cli faces
   ```

   A `CERTIFICATE_VERIFY_FAILED` here means the interpreter is not seeing the payload's
   own certificates (F221). What it should be seeing:

   ```powershell
   & "$env:LOCALAPPDATA\Programs\Sorta\python\python.exe" -c "import os; print(os.environ['SSL_CERT_FILE'])"
   # ...\Programs\Sorta\lib\certifi\cacert.pem
   ```

7. Sort a folder of real photographs: dates, GPS and cities have to be there without a
   single download.
8. Run **Sorta setup** again and accept one tier; the refusal in step 3 must not have
   been final.
9. Uninstall, and check that `%APPDATA%\sorta\config.yaml`, the run log and the preview
   cache are still there — uninstalling a program is not a request to delete data.
