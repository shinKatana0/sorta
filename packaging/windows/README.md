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
python\      a standalone CPython, fetched by `uv python install`
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
5. Sort a folder of real photographs: dates, GPS and cities have to be there without a
   single download.
6. Run **Sorta setup** again and accept one tier; the refusal in step 3 must not have
   been final.
7. Uninstall, and check that `%APPDATA%\sorta\config.yaml`, the run log and the preview
   cache are still there — uninstalling a program is not a request to delete data.
