# Decisions

Why Sorta is the way it is — the answers that cost something to get, kept apart from the
code that happens to implement them today.

This file exists for three readers. Somebody deciding whether to trust the tool. Somebody
about to re-open a question that was already settled. And whoever one day builds this on
another platform, where the models and the file system will be different but the answers
below will not.

**It is not a changelog and not a roadmap.** Each entry is a decision, the measurement
behind it, and — as importantly — what it does **not** claim.

---

## 1. What the product is

### 1.1 The index is separate from the sorting

The pipeline fills a SQLite index — metadata, geography, face embeddings, clusters, events.
Sorting is the application of a *view* of that index to the file system. Changing the mode
(`--by city` / `person` / `event`) never requires re-scanning.

**Why it matters beyond tidiness:** it makes every expensive answer reusable. A collection
is scanned once and can be laid out five ways, and a new question can be asked of old data
without touching a single file.

### 1.2 The product ranks; the person decides

Every list is **ordered, not cut by a threshold**. A slice says "these look most like X,
starting with the most confident" and never "these are X".

This is not modesty. It is what the numbers support: see §3.1 and §3.4 — the thresholds we
measured were either far less complete than the ranking that replaced them, or no better
than chance. **Depth of list is the only lever of completeness we have ever confirmed**
(§3.2), and only a person can decide how deep to go.

### 1.3 Every slice states what it measured

A caption reads *"the model calls these on-screen; about one in three is an ordinary
photograph, check before deleting"* rather than showing a count.

The alternative is worse than useless. On one live run the screenshot rescue moved **181 of
the owner's own photographs** into the screenshot bin, and a caption promising a
classification is what stops a person from looking again.

### 1.4 The terminal computes; the interface decides

Both entry points are first-class, and they do different work. The CLI runs **every stage
of the pipeline** and applies a layout — `index`, `geo`, `landmarks`, `phash`, `junk`,
`classify`, `faces`, `events`, `run`, `sort`, `album`, `search`, `dupes`, `stats`,
`doctor`, `cache`, `reset`, `undo`, and per-run overrides for every option a feature ever
added.

What it deliberately does **not** carry is the set of actions where a person looks at a
photograph and decides:

```
resolving a duplicate group        marking an animal by hand
correcting a place                 pinning a query as a slice
restoring a soft frame             sending a frame to the trash
```

`sorta dupes` lists them; it does not resolve them.

**This follows from §1.2.** Resolving duplicates from a command line means choosing without
seeing the frames — and §3.4 says that even with the frames in front of it, no rule we have
beats chance. An action whose whole content is human judgement has no business having a
flag.

The consequence for a script: everything reproducible is scriptable, and nothing that
needs an opinion pretends to be.

### 1.5 Safety of moves comes before everything

- `sort` is dry-run by default; real movement needs `--apply`.
- Every move is journaled **before** it happens; `undo` replays the journal backwards.
- blake3 is verified before a move and the destination is checked after.
- A name conflict never overwrites: `_1`, `_2`.

### 1.6 Local by construction, not by default

No code in the product sends an image anywhere. The cloud naming provider was removed
together with its upload, and a test refuses to let the string return unnoticed. The single
outbound path is optional Nominatim geocoding, which receives **rounded coordinates**, and
the config key says so in its name. The web app binds to `127.0.0.1`.

"By default" would be a weaker promise, and a weaker one than the code actually keeps.

---

## 2. Which instrument answers which question

Three regimes, and the boundary between them is measured rather than assumed. Getting this
wrong is the most expensive mistake available here, because "is there a product in this
frame" and "is this the best frame of the burst" sound like the same kind of question and
are not.

| the question is about | what wins | measured |
|---|---|---|
| a **physical property** of the frame | arithmetic | eyes by geometry 62%/48% vs a VLM's 60%/9%; blur by ranking 53% vs 8% by threshold |
| **what is depicted** | a model | products 78%/94%, screenshot rescue, animals 82%/64% |
| **human judgement** | nobody | the best frame of a burst: model 32%, arithmetic 28%, **random 30.4%** |

**A general-purpose VLM is useless when asked in words about small properties of an
image** — focus, open eyes, "interestingness". It earns its cost on questions about
CONTENT. Every question of the first kind we tried died: 5% precision for "was this shot
by accident", no discrimination at all for "does this have a subject", 60%/9% for "are the
eyes open".

**Before choosing an instrument, decide which of the three regimes the question belongs
to.**

---

## 3. Findings that survive a change of models

These are the results a re-implementation should inherit. The thresholds will need
re-measuring on other models; the shape of the answers will not.

### 3.1 Precision is cheap to measure, completeness is not — and only completeness matters

For months every filter was evaluated on the top of its own list, which measures only how
good the confident cases are. A random sample of **500 hand-labelled frames** showed what
that had hidden: a third of the archive falls into no class at all, and the blur filter
that looked fine reached **8%** of the blurred frames.

**A sweep over a gate, measured only inside the gate, measures its own assumption.** The
sample must include what the hypothesis does not expect.

### 3.2 Depth of the list is the only confirmed lever of completeness

Doubling the list adds about **25 points**. Nothing else did: refining a query with
examples (PRF and hand-picked alike) moved it by −2…−3 points with an undetermined sign,
an ensemble of phrasings did nothing, and `genderage` gave 100% precision at 25% recall.

### 3.3 Naming what is missing beats refining what is there

Products by query at depth 3200 recovered 65% of what the model missed. **Asking about "an
item held in a hand"** — the shape a human labelling of the misses named — recovered the
same frames three times shallower, and widening the model's own question moved recall
**80% → 94%** for three points of precision.

This is not §3.2 contradicted: the depth lever is about how far you look, this one is
about looking in the right place. The difference is that the wording came from labelled
causes rather than from guessing.

### 3.4 Nobody can pick the best frame of a burst

111 groups labelled blind: sharpness 27%, arithmetic 28%, a cascade 28%, the VLM 32% —
against **30.4% for choosing at random**. The VLM cost 451 seconds of a run for that.

It does **not** follow that the frames are alike: the owner chose confidently in 88 of the
111 groups. The difference is real and none of the signals available catch it. So the
product shows the group, pre-selects nothing, and lets several be kept.

### 3.5 Enlargement helps small frames and nothing else

Blind pairs, 80 of them: on frames under ~1280 px a super-resolution model beats plain
enlargement **62% against 10%**, and the gain grows as the frame shrinks (66% below 640 px,
52% at 1024–1280). Above the ceiling the frame is squeezed and rebuilt from a quarter of
itself, and on *fidelity* that came back 35/35/30 — indistinguishable. So above the ceiling
the product **refuses** and says why (F198): doing it anyway spends a run of the model and
leaves a near-duplicate file in the archive to buy a result nobody can tell from the
original.

A 1:1 deblurring model, the obvious answer for the other half, came back **21% against the
original's 36%** on fidelity: also nothing. The structural argument for it was correct —
it keeps 100% of the pixels where the ×4 path keeps 54–64% — and the kept pixels did not
make the frame truer. **Correct mechanics are not an argument for a result.**

### 3.6 Three tiers of sameness, and the cost of an error differs by orders of magnitude

| tier | what a mistake costs | judgement needed |
|---|---|---|
| identical bytes | **nothing** — it is the same file | none |
| same picture, other bytes | a better or worse copy | a rule: size, and it is checkable |
| similar frames | **a photograph, for good** | **nobody can** (§3.4) |

Half the archive is the first tier. One word — "duplicate" — was doing all three jobs.

---

## 4. How to measure, so that the answer is about the world

The most expensive lesson here, and the most portable: over three days, **eight
measurements answered a question that was not the one being asked**. None of them was a
coding error.

### 4.1 An instrument that agrees with the hypothesis by construction proves nothing

- A super-resolution model was judged by **laplacian variance** — a number any upscaler
  raises by drawing detail in. We measured sharpness and called it fidelity.
- Face crops were taken with coordinates from the full frame against a downscaled preview:
  39 of 68 fell outside the image, and the surviving 29 reported "100% recall" instead of
  62%. **A broken crop flatters the result rather than failing.**
- A class-recall figure was drawn from a sample filtered to `verdict='photo'` — that is,
  with everything the classifier had already caught removed. Recall on it was zero by
  construction.

### 4.2 A blind comparison does not save you if the question cannot come out otherwise

The pairs for the restore measurement were properly blind and still gave the wrong answer,
because the question was *"which is closer to the original"* on a population where **the
original is the defect**. One half literally was the original.

Ask not only "would the judge agree with the hypothesis" but **"could this question produce
any other answer"**.

### 4.3 A comparison of methods needs the method "at random"

Three rules and a model came in at 27–32% agreement with a person, and the report ranked
them. The random baseline is **30.4%**. Without it, "A beats B by four points" reads as "A
works".

### 4.4 A stratified sample must be weighted by the population

999 frames were drawn in three deliberately disproportionate layers, and a report computed
recall over the raw list: **94%**. Weighted by the share of each layer in the collection it
is **79%**. The unweighted number describes the draw, not the archive.

### 4.5 Include a control band the hypothesis does not expect

The hypothesis was that restoration helps *blurred* small frames. A control band of small
**sharp** frames was drawn deliberately — and it won too, which is how we learned the
driver is SIZE and not blur. Without the control we would have shipped the wrong reason.

### 4.6 Extrapolate from a sample, not from a handful

"570 downloaded frames" came from thirteen. The real population was ~143 — a fourfold
error. Two detector figures moved ~20 points each between a 200-frame and a 500-frame
sample.

### 4.7 A decision deferred to a measurement needs a way back into the code

A feature that ships an honest "the measurement will decide this" leaves a hole, and the
hole does not close by itself. The enlargement ceiling had one for a day: the answer came
back on 2026-08-04 and reached a document, the code kept doing the work it had been left
doing, and the owner found out by pressing the button and receiving a useless file. **The
verdict has to return to the place that was waiting for it** — a brief is where the number
lives, not where it acts.

---

## 5. What the schema encodes on purpose

- **One writer per table.** A module's interface is the set of tables it reads and writes;
  modules do not import each other. The single exception is `events.name`, and the
  predicate `name_is_manual = 0` guards it.
- **NULL means NOT ASKED**, never "no". A consumer reading a defaulted `False` would
  conclude that a frame nothing has ever looked at has its eyes closed.
- **Answers carry the fingerprint of the question.** `frame_quality.source` holds a hash of
  the prompts (`clip#abc12345`), so editing the wording invalidates exactly the rows it
  produced. This is what lets a prompt change take effect on an indexed collection — and
  what lets a measurement be trusted to describe the question it was actually asked.
- **Retired questions keep their columns, NULL.** Dropping them would need a table rebuild,
  and a documented empty column is cheaper than an excised one.
- **A recommendation is not a decision.** `group_keeper` is what the machine thinks;
  `dedup_choice`, `manual_places`, `manual_pet`, `manual_overrides` are what the person
  decided, and no stage writes those.
- **Paths are stored as absolute POSIX strings**, normalized on Windows.

---

## 6. What a port to another platform inherits — and what it does not

Roughly 30% of this project carries over, and it is the expensive 30%.

**Carries over:** everything in §1–§5. The design, every verdict, the measurement protocol,
and the schema — `schema.sql` is commented with the reasoning beside each field and can be
read as a specification.

**Does not carry over:** Python, the model formats, and the deep tier at all — 0.78 s per
frame on a desktop GPU becomes minutes on a phone.

**And one constraint changes the product rather than the implementation.** Sorta is built
on physically moving files into a new folder tree. On Android that is not permitted:
scoped storage, MediaStore, SAF. There, a layout has to become a *view*.

Which is where this project arrived anyway: a slice is a saved query, and a query is a
view. The port is not a smaller Sorta — it is the same idea with the last assumption
removed.

---

### Linux installs with one line, and gets no packaging

**Decided 2026-08-06.** `uv tool install` is the only supported way in. No AppImage, no
deb/rpm, no snap or flatpak.

The exact form is from a checkout, not from PyPI: `uv tool install ".[cpu]"`. The package
is not published to PyPI, and that is not decided here — this only records what the
install looks like today, so that "one line" is not read as a promise of
`uv tool install sorta`.

The Windows installer exists because there a person has **no other way**: no package
manager by default, and no habit of the terminal. Linux has both, and carrying the
decision across out of symmetry would create an obligation lasting years — every package
format is a build, a repository, a signature and a promise to keep them current.
Packaging nobody maintains becomes, within six months, a stale version published under
the author's name, which is worse than none.

The user here is known: someone with hundreds of gigabytes of photographs and a graphics
card. One line in a terminal is not a barrier for them.

**Re-open only with new data about WHO specifically cannot run that line** — not from a
feeling that everyone deserves an installer.

### The Windows installer ships unsigned, and says so out loud

**Decided 2026-08-06.** A code-signing certificate costs money and time, and neither is
being spent at this stage. The installer is released unsigned.

The consequence is accepted deliberately: SmartScreen shows its red screen. So the duty
moves to the text — the download page and the guide warn about it BEFOREHAND, explain the
way through ("More info" → "Run anyway"), and publish a checksum. A silent red screen
reads as "this program is dangerous"; a forewarned one reads as "this author has no
certificate".

The build keeps a place for a signature as a separate, switched-off step (`--sign`, or
`SORTA_SIGN_INSTALLER=1`). When a certificate appears it plugs in, rather than rewriting
the packaging. Nothing in the repository signs anything today, and no release has been
signed.

### The two heaviest tiers are off by default

**Decided 2026-08-07/08.** `vlm.enabled` and `features.landmarks` both default to false,
and every run screen prices them before it starts.

The reason is not caution, it is what each buys. The deep VLM tier produces exactly one
thing the fast tier cannot produce at all — the `product` class — for the largest single
block of time in a run, and it wants a 24 GB card (20.5 GB measured peak). Landmarks
recognise places by sight for a collection whose places are mostly already known from
GPS. A default that costs hours has to be worth hours to everybody, and neither of these
is; both are incremental, so switching one on later costs the same as switching it on now.

**What this does not claim:** not that they are unimportant. It claims that the price is
visible before it is paid and that a person who never opens the settings still gets a
finished layout.

### macOS is postponed, and not promised in the meantime

**Decided 2026-08-08.** There is no Mac here. `accel.py` carries the Metal and CoreML
rungs because writing them was the cheap part of removing a duplicated CUDA branch, but
nothing on that platform has been run against a real collection, and the `macos-latest`
job in CI is marked advisory (`continue-on-error`) for exactly that reason.

The rule that follows is about documentation: **an untested platform is not listed as a
supported one.** A runner that imports the package answers a narrower question than a
reader takes "macOS" in a requirements table to mean — F105 moved 7–11 verdicts out of 300
by changing an attention kernel, and a different compute device is the same class of
change, so even a green run there would not say the verdicts match.

**Reopen when there is a Mac to run a collection on**, not when the advisory job goes
green.

### A comment carries a measurement, a decision or a trap — and a ratchet holds it there

**Decided 2026-08-08.** A comment or a docstring stays only if it states a MEASUREMENT (a
number with a date), a DECISION that reads against what the code seems to say, or a TRAP
the next edit would spring. Retelling the code, rhetoric around a decision already made,
and the second account of a defect in a neighbouring module all go.

The rule was written down and broken an hour later, by its author, on six comments in a
row — which is the finding, not an anecdote. So the check is mechanical:
`tests/prose_budgets.txt` and `tests/prose_budgets_tests.txt` hold one number per file,
and the gate goes red on a file that grew. Under the number is free; **raising** one is an
edit of that file, so growth arrives in a diff with a reason beside it.

**What this does not claim:** not that less prose is better prose. The budgets were fixed
as the fact stood, as a ceiling and never as a target, and there is deliberately no limit
on a single block — a trap sometimes needs six lines, and a per-block limit would get it
split in two rather than deleted.

## 7. Questions closed by measurement — do not re-open without new data

| question | verdict |
|---|---|
| "was this shot by accident" (VLM) | 5% precision, anti-correlated |
| "does this frame have a subject" (VLM) | separates nothing |
| "are the eyes open" (VLM) | 60%/9%; geometry gives 62%/48% for free |
| the best frame of a burst (any method) | indistinguishable from random |
| refining a query with examples, PRF | −2…−3 points, sign undetermined |
| an ensemble of phrasings | no effect |
| `genderage` for the children slice | 100% precision at 25% recall |
| widening the deep-tier gate | 3.6 hours for a hundred more finds |
| StreetCLIP for places | measured and rejected (F85b) |
| place from text in the frame | measured and rejected (F96) |
| float16 for embeddings, VLM batching | no gain |
| a 1:1 deblurring model | not truer: 21% against the original's 36% |

Each of these has a brief with the numbers. Re-opening one without new data is a session
spent to reach the same place.
