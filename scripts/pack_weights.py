#!/usr/bin/env python3
"""Split the monolithic checkpoints into shareable parts, and put them back together.

Each detector checkpoint carries a full copy of the DINOv3 trunk (85.7M parameters, 343 MB)
next to a 4.2M-parameter keypoint head. Four detectors ship per release, so three quarters of
the download is the same trunk four times over. The trunks are not quite identical -- detector
continue-training updated the last blocks of some of them -- but they differ in a minority of
tensors, so a release can carry one trunk plus a small override per detector.

    python scripts/pack_weights.py pack       # checkpoints/ -> release_weights/
    python scripts/pack_weights.py assemble   # release_weights/ -> checkpoints/
    python scripts/pack_weights.py verify     # assembled == original, tensor by tensor

`assemble` reconstructs every tensor of the original file, in the original key order, so the
evaluators load exactly what they loaded before; `verify` is what proves it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(ROOT, "checkpoints")
PARTS = os.path.join(ROOT, "release_weights")
ROBOTS = ("panda", "kuka")
STAGES = ("pass1", "pass2")
# The trunk every detector is a variation of; panda/pass1 carries it untouched.
TRUNK_SOURCE = ("panda", "pass1")
TRUNK_FILE = "dinov3_vitb16_trunk.pth"
MANIFEST = "manifest.json"


def load_sd(path: str) -> dict:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        for wrapper in ("model", "state_dict"):
            if wrapper in obj and isinstance(obj[wrapper], dict):
                return obj[wrapper]
    return obj


def split_backbone(sd: dict) -> tuple[dict, dict]:
    bb = {k: v for k, v in sd.items() if k.startswith("backbone.")}
    head = {k: v for k, v in sd.items() if not k.startswith("backbone.")}
    return bb, head


def mb(path: str) -> float:
    return os.path.getsize(path) / 1e6


def pack() -> int:
    os.makedirs(PARTS, exist_ok=True)
    trunk_sd = load_sd(os.path.join(CKPT, TRUNK_SOURCE[0], f"{TRUNK_SOURCE[1]}_detector.pth"))
    trunk, _ = split_backbone(trunk_sd)
    torch.save(trunk, os.path.join(PARTS, TRUNK_FILE))
    print(f"[trunk] {TRUNK_FILE}  {len(trunk)} tensors  "
          f"{mb(os.path.join(PARTS, TRUNK_FILE)):.1f} MB")

    manifest: dict = {"trunk": TRUNK_FILE, "robots": {}}
    for robot in ROBOTS:
        entry: dict = {}
        for stage in STAGES:
            src = os.path.join(CKPT, robot, f"{stage}_detector.pth")
            sd = load_sd(src)
            bb, head = split_backbone(sd)
            delta = {k: v for k, v in bb.items()
                     if k not in trunk or not torch.equal(v, trunk[k])}
            d_dir = os.path.join(PARTS, robot)
            os.makedirs(d_dir, exist_ok=True)
            head_name = f"{stage}_kphead.pth"
            torch.save(head, os.path.join(d_dir, head_name))
            part = {"keys": list(sd.keys()), "head": head_name}
            if delta:
                delta_name = f"{stage}_trunk_delta.pth"
                torch.save(delta, os.path.join(d_dir, delta_name))
                part["delta"] = delta_name
            entry[f"{stage}_detector"] = part
            print(f"[{robot}/{stage}] delta {len(delta):3d}/{len(bb)} tensors"
                  f"  head {mb(os.path.join(d_dir, head_name)):.1f} MB"
                  + (f"  delta {mb(os.path.join(d_dir, delta_name)):.1f} MB" if delta else "  (trunk verbatim)"))
            # The angle and rotation heads carry no trunk, so they ship unchanged.
            for kind in ("angle", "rotation"):
                name = f"{stage}_{kind}.pth"
                s = os.path.join(CKPT, robot, name)
                if os.path.isfile(s):
                    torch.save(load_sd(s), os.path.join(d_dir, name))
                    entry[f"{stage}_{kind}"] = {"file": name}
        manifest["robots"][robot] = entry

    with open(os.path.join(PARTS, MANIFEST), "w") as fh:
        json.dump(manifest, fh, indent=2)

    before = sum(os.path.getsize(os.path.join(CKPT, r, f))
                 for r in ROBOTS for f in os.listdir(os.path.join(CKPT, r)) if f.endswith(".pth"))
    after = sum(os.path.getsize(os.path.join(dp, f))
                for dp, _, fs in os.walk(PARTS) for f in fs if f.endswith((".pth", ".json")))
    print(f"\n{before/1e6:.0f} MB -> {after/1e6:.0f} MB "
          f"({100 * (1 - after / before):.0f}% smaller)")
    return 0


def assemble(robots=ROBOTS, dest=CKPT) -> int:
    with open(os.path.join(PARTS, MANIFEST)) as fh:
        manifest = json.load(fh)
    trunk = torch.load(os.path.join(PARTS, manifest["trunk"]), map_location="cpu",
                       weights_only=False)
    for robot in robots:
        entry = manifest["robots"][robot]
        out_dir = os.path.join(dest, robot)
        os.makedirs(out_dir, exist_ok=True)
        for name, part in entry.items():
            target = os.path.join(out_dir, f"{name}.pth")
            if "file" in part:            # angle / rotation: a straight copy
                torch.save(torch.load(os.path.join(PARTS, robot, part["file"]),
                                      map_location="cpu", weights_only=False), target)
                continue
            head = torch.load(os.path.join(PARTS, robot, part["head"]), map_location="cpu",
                              weights_only=False)
            merged = dict(trunk)
            if "delta" in part:
                merged.update(torch.load(os.path.join(PARTS, robot, part["delta"]),
                                         map_location="cpu", weights_only=False))
            merged.update(head)
            # Rebuild in the recorded order so the file matches what was packed.
            torch.save({k: merged[k] for k in part["keys"]}, target)
            print(f"[assembled] {robot}/{name}.pth  {mb(target):.1f} MB")
    return 0


def verify() -> int:
    tmp = os.path.join(ROOT, "outputs", "assembled_check")
    os.makedirs(tmp, exist_ok=True)
    assemble(dest=tmp)
    bad = 0
    for robot in ROBOTS:
        for f in sorted(os.listdir(os.path.join(CKPT, robot))):
            if not f.endswith(".pth"):
                continue
            a, b = load_sd(os.path.join(CKPT, robot, f)), load_sd(os.path.join(tmp, robot, f))
            if list(a.keys()) != list(b.keys()):
                print(f"  MISMATCH {robot}/{f}: key order or set differs")
                bad += 1
                continue
            off = [k for k in a if not torch.equal(a[k], b[k])]
            print(f"  {'OK  ' if not off else 'DIFF'} {robot}/{f}  {len(a)} tensors"
                  + (f"  {len(off)} differ" if off else ""))
            bad += bool(off)
    print("\nassembled checkpoints are identical to the originals" if not bad
          else f"\n{bad} file(s) did not round-trip")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["pack", "assemble", "verify"])
    ap.add_argument("--robot", choices=[*ROBOTS, "both"], default="both")
    args = ap.parse_args()
    if args.action == "pack":
        return pack()
    if args.action == "verify":
        return verify()
    return assemble(ROBOTS if args.robot == "both" else (args.robot,))


if __name__ == "__main__":
    sys.exit(main())
