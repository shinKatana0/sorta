#!/usr/bin/env python3
"""F214: ask a machine what it can compute on, and which tiers fit on it.

There is no Mac here. Everything F214 decides — the preference order, the retreat to
the CPU, the untouched CUDA path — is checked by tests that fake the hardware, and a
faked accelerator proves nothing about a real one. This script is what a machine that
HAS one runs: it prints what the machine offers and then tries the tiers on it, in
order of size, reporting each as fitted or skipped instead of failing.

    python scripts/probe_accelerator.py              what the machine offers (seconds)
    python scripts/probe_accelerator.py --faces      + buffalo_l, ~0.3 GB
    python scripts/probe_accelerator.py --clip       + CLIP ViT-L, ~1.6 GB
    python scripts/probe_accelerator.py --all --strict   every tier, and a miss fails

The deep tier (Qwen2.5-VL-3B, ~7 GB) has no flag on purpose: it does not fit a CI
runner by disk or by time, and a flag that nobody can run reads like a check somebody
performed. On a real Mac the same question is `sorta doctor` plus an ordinary run.

**`--clip` is the part that measures rather than reports.** A device is the same class
of change as an attention kernel, and the attention change moved 7-11 verdicts out of
300 — so the same frames are scored twice, once on the CPU and once on the accelerator,
and the gap between the two runs is printed. Synthetic frames are not a photo
collection and this is not the comparison the feature owes; it is the part of it a
runner with no photographs can do.

Never fatal without `--strict`: a tier that does not fit is the ANSWER, not an error.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sorta import accel  # noqa: E402 — after the path insert, so a checkout runs as-is

FITTED = "FITTED"
SKIPPED = "SKIPPED"


def free_gb(path: str = ".") -> float:
    return shutil.disk_usage(path).free / 1024 ** 3


def report(tier: str, status: str, detail: str) -> str:
    """One line per tier, in a shape a person can read out of a runner's log."""
    return f"tier {tier}: {status} — {detail}"


def probe_faces() -> tuple[str, str]:
    """buffalo_l through insightface: the providers a real onnxruntime hands back.

    What is being asked is not whether faces are found — it is whether the session
    builds at all with the providers this machine offers, and which one it settled on.
    """
    from insightface.app import FaceAnalysis

    providers = accel.onnx_providers()
    started = time.perf_counter()
    app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition"],
                       providers=providers)
    app.prepare(ctx_id=0, det_size=(640, 640))
    got = getattr(app.models.get("detection"), "session", None)
    settled = list(got.get_providers()) if got is not None else []
    return FITTED, (f"asked for {providers}, running on {settled or 'unknown'} "
                    f"({time.perf_counter() - started:.1f}s to load)")


def probe_clip(frames: int = 8) -> tuple[str, str]:
    """CLIP ViT-L on the CPU and on the accelerator, over the SAME synthetic frames.

    The point is the last number: how far the two devices' probabilities drift apart.
    Zero would be a pleasant surprise; a gap wide enough to move an argmax is what the
    brief calls a reason to look, not the price of speed.
    """
    import numpy as np
    import open_clip
    import torch
    from PIL import Image

    device = accel.torch_device(torch)
    if device == accel.CPU:
        return SKIPPED, "no accelerator on this machine — nothing to compare the CPU to"

    prompts = ["a photograph of a landscape", "a screenshot of a document",
               "a photograph of a person"]
    images = [Image.effect_mandelbrot((224, 224), (-2, -1.5, 1, 1.5), 8 + i).convert("RGB")
              for i in range(frames)]

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14-quickgelu", pretrained="openai", device=accel.CPU)
    tokenizer = open_clip.get_tokenizer("ViT-L-14-quickgelu")
    model.eval()
    batch = torch.stack([preprocess(im) for im in images])
    tokens = tokenizer(prompts)

    def scores(on_device: str) -> list[float]:
        model.to(on_device)
        with torch.no_grad():
            feats = model.encode_image(batch.to(on_device))
            feats /= feats.norm(dim=-1, keepdim=True)
            text = model.encode_text(tokens.to(on_device))
            text /= text.norm(dim=-1, keepdim=True)
            probs = (100.0 * feats @ text.T).softmax(dim=-1)
        return [float(v) for v in np.asarray(probs.float().cpu()).ravel()]

    on_cpu = scores(accel.CPU)
    on_accelerator = scores(device)
    agree, gap = accel.verdicts_agree(on_cpu, on_accelerator)
    labels_cpu = np.asarray(on_cpu).reshape(len(images), -1).argmax(axis=1)
    labels_acc = np.asarray(on_accelerator).reshape(len(images), -1).argmax(axis=1)
    moved = int((labels_cpu != labels_acc).sum())
    return FITTED, (f"cpu vs {device}: largest probability gap {gap:.2e} "
                    f"({'within' if agree else 'PAST'} tolerance), "
                    f"{moved} of {len(images)} labels moved")


def attempt(run: Callable[[], tuple[str, str]]) -> tuple[str, str]:
    """Run one tier's probe; anything it raises is the answer, not a crash.

    A runner that has no room for a 1.6 GB download says so in a line of output. It
    must not say it in a traceback, because then the job is red for a reason that is
    the point of the job rather than a fault in it.
    """
    try:
        return run()
    except Exception as exc:
        return SKIPPED, f"{type(exc).__name__}: {exc}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--faces", action="store_true", help="buffalo_l, ~0.3 GB")
    parser.add_argument("--clip", action="store_true", help="CLIP ViT-L, ~1.6 GB")
    parser.add_argument("--all", action="store_true", help="every tier that has a flag")
    parser.add_argument("--strict", action="store_true",
                        help="a tier that did not fit fails this command")
    args = parser.parse_args(argv)

    print(f"platform: {sys.platform}, python {sys.version.split()[0]}")
    print(f"free disk: {free_gb():.1f} GB")
    print(accel.describe())

    tiers: list[tuple[str, Callable[[], tuple[str, str]]]] = []
    if args.faces or args.all:
        tiers.append(("faces (buffalo_l)", probe_faces))
    if args.clip or args.all:
        tiers.append(("clip (ViT-L-14)", probe_clip))

    missed = 0
    for tier, run in tiers:
        status, detail = attempt(run)
        print(report(tier, status, detail))
        missed += status == SKIPPED
    if missed and args.strict:
        print(f"{missed} tier(s) did not fit — see the lines above")
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
