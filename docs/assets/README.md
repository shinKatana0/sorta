# Screenshot assets

`hero.gif`, `process.png` and `layout.png` (referenced from the top‑level README) are
captured on a **synthetic** demo collection — generated JPEGs with fake EXIF/GPS labelled
"SORTA DEMO", no real photos or people. `icon.png` is the brand mark the header draws.
Regenerate them the same way if the UI changes.

**One trap when recapturing.** The web app remembers the last source folder in the
browser's `localStorage`, and a demo instance served from the same `127.0.0.1:8756` will
restore the path of a REAL session into the Source field. Nothing leaks from the server —
the path is in the browser — but it lands in the screenshot all the same. Capture in a
private window, or process the demo folder once so the remembered path is the demo's.

## The guides ship without screenshots, and that is a decision

The user guides carry their walkthrough in **real command output** instead — 22 worked
commands and 68 blocks of what they print, on a synthetic collection.

The reason is that text does not go stale quietly. The README pictures showed a "Cities"
tab for three months after F133 replaced that arrangement, and nobody noticed; a block of
command output either matches what the program prints or diverges visibly on the first
run, and a test can check it. A screenshot cannot be checked by anything but an eye that
happens to look.

**When to revisit**: after the installers land. A person who installs Sorta and never
opens a terminal is the reader for whom pictures start earning their upkeep, and there is
no such reader yet.

## If screenshots are added later

1. Build the synthetic collection — generated JPEGs with embedded EXIF/GPS. Make it wide
   enough to show what the product does now: several cities across years, byte-identical
   copies, one picture stored twice with different bytes, a burst of similar frames, a few
   screenshots, and a folder with no GPS at all. Thin data makes an empty-looking screen.
   **Never use real personal photos or images of identifiable people.**
2. Point a **separate** `config.yaml` at it, with its own `database:` — never the working
   one — then `sorta index` … `sorta phash`, and `sorta ui`.
3. Capture in a **private browser window** (see the trap above), and name the tabs as they
   are called today: **Overview**, **Review**, **Layout**, **Slices**, **Moves**.
4. Reference them from the guides with relative links, e.g.
   `![The Layout tab](../assets/layout.png)`.

Keep screenshots on synthetic/generated data only — never a real personal photo
collection or images of real identifiable people.
