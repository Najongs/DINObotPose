#!/usr/bin/env python3
"""Fetch the released weights from the HuggingFace Hub and assemble checkpoints/.

    python scripts/download_weights.py              # both robots
    python scripts/download_weights.py --robot panda

The release ships the DINOv3 trunk once (343 MB) rather than once per detector, with a small
override for the detectors whose continue-training moved their last blocks. Assembly rebuilds
the six checkpoints per robot exactly as they were trained, so nothing about inference changes:

    both robots   741 MB downloaded, 1486 MB on disk
    panda only    517 MB
    kuka only     573 MB

`scripts/pack_weights.py verify` is the tensor-by-tensor proof that assembly round-trips.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARTS = os.path.join(ROOT, "release_weights")
REPO_ID = os.environ.get("DINOBOTPOSE_HF_REPO", "Najongs/dinobotpose")
MANIFEST = "manifest.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot", choices=["panda", "kuka", "both"], default="both")
    ap.add_argument("--repo-id", default=REPO_ID)
    ap.add_argument("--parts-only", action="store_true",
                    help="download without assembling (assemble later with pack_weights.py)")
    args = ap.parse_args()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        sys.exit("huggingface_hub is required: pip install huggingface_hub")

    def get(name: str) -> str:
        target = os.path.join(PARTS, name)
        if os.path.isfile(target):
            print(f"[skip] {name}")
            return target
        print(f"[get ] {name}", flush=True)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        import shutil
        # Copy out of the HF cache so the tree survives a cache clear.
        shutil.copyfile(hf_hub_download(repo_id=args.repo_id, filename=name), target)
        return target

    manifest = json.load(open(get(MANIFEST)))
    get(manifest["trunk"])
    robots = ["panda", "kuka"] if args.robot == "both" else [args.robot]
    for robot in robots:
        for part in manifest["robots"][robot].values():
            for key in ("head", "delta", "file"):
                if key in part:
                    get(f"{robot}/{part[key]}")

    if args.parts_only:
        print("\nparts downloaded — assemble with: python scripts/pack_weights.py assemble")
        return 0

    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from pack_weights import assemble
    assemble(robots)
    print("\ncheckpoints ready — verify with: python scripts/doctor.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
