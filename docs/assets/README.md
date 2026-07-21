# Screenshot assets

The user guides currently ship **without** UI screenshots (text worked‑examples
with real command output carry the walkthrough instead). To add real screenshots
later:

1. Build the synthetic mini‑collection from the guide's "Worked example"
   (`docs/guide/user-guide.en.md` §7): a handful of generated JPEGs with embedded
   EXIF/GPS. **Never use real personal photos or images of identifiable people.**
2. Point a `config.yaml` at that folder, run `sorta index` + `sorta run`, then
   `sorta ui` and open `http://127.0.0.1:8756`.
3. Capture the **Process** tab (source input + the "Detect faces"/"Detect events"
   checkboxes) as `screenshot-process.png`, and the **Cities** tab (the proposed
   `Country/City/Year/District` tree) as `screenshot-cities.png`. Crop to ~1000×620.
4. Add them into this folder and reference them from the guides' §6 with relative
   links, e.g. `![Process tab](../assets/screenshot-process.png)`.

Keep screenshots on synthetic/generated data only — never a real personal photo
collection or images of real identifiable people.
