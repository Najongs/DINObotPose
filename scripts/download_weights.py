#!/usr/bin/env python3
"""Fetch the released checkpoints from the HuggingFace Hub into checkpoints/.

    python scripts/download_weights.py              # both robots
    python scripts/download_weights.py --robot panda

The weights are not in git: the six files per robot total 743 MB, and the detector checkpoints
carry the frozen DINOv3 backbone inside them (223 of 261 tensors), which is what lets this
package run without any HuggingFace model access at inference time.
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ID = os.environ.get("DINOBOTPOSE_HF_REPO", "Najongs/dinobotpose")
FILES = ["pass1_detector.pth", "pass1_angle.pth", "pass1_rotation.pth",
         "pass2_detector.pth", "pass2_angle.pth", "pass2_rotation.pth"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot", choices=["panda", "kuka", "both"], default="both")
    ap.add_argument("--repo-id", default=REPO_ID)
    args = ap.parse_args()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        sys.exit("huggingface_hub is required: pip install huggingface_hub")

    robots = ["panda", "kuka"] if args.robot == "both" else [args.robot]
    for robot in robots:
        dest = os.path.join(ROOT, "checkpoints", robot)
        os.makedirs(dest, exist_ok=True)
        for name in FILES:
            target = os.path.join(dest, name)
            if os.path.isfile(target):
                print(f"[skip] {robot}/{name} already present")
                continue
            print(f"[get ] {robot}/{name}", flush=True)
            path = hf_hub_download(repo_id=args.repo_id, filename=f"{robot}/{name}")
            # Copy rather than symlink into the HF cache, so the tree stays self-contained
            # if the cache is later cleared.
            import shutil
            shutil.copyfile(path, target)
    print("\ncheckpoints ready — verify with: python scripts/doctor.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
