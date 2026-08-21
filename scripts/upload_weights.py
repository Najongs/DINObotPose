#!/usr/bin/env python3
"""Publish release_weights/ to the HuggingFace Hub, with the model card the license requires.

Run `pack_weights.py pack` first, then:

    huggingface-cli login          # a token with WRITE scope; the read token cannot upload
    python scripts/upload_weights.py --dry-run
    python scripts/upload_weights.py

The weights are a derivative of the DINOv3 Materials, so the card carries the "Built with DINOv3"
notice and points at the license copy that must travel with them. `--dry-run` lists exactly what
would be uploaded and writes the card to stdout without touching the Hub.
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARTS = os.path.join(ROOT, "release_weights")

CARD = """---
license: other
license_name: dinov3-license
license_link: https://ai.meta.com/resources/models-and-libraries/dinov3-license/
tags:
  - robotics
  - pose-estimation
  - robot-pose
  - dinov3
---

# DINObotPose weights

Monocular robot pose and joint-angle estimation from a single RGB image, with no encoder readings
and no ground-truth bounding box. These are the released checkpoints for
[{repo}](https://github.com/{gh}); the code, the evaluation entry points, and the reproduction
script live there.

## Built with DINOv3

These weights contain the DINOv3 ViT-B/16 trunk and are a derivative of the DINOv3 Materials. They
are distributed under the [DINOv3 License Agreement]\
(https://ai.meta.com/resources/models-and-libraries/dinov3-license/), not under the MIT license
that covers the code. Downloading them is acceptance of that agreement; redistributing them onward
is permitted only under the same terms, with a copy of the agreement and the "Built with DINOv3"
notice included. Publications reporting results obtained with them should acknowledge DINOv3
(Siméoni et al., 2025).

## Layout

The trunk ships once rather than once per detector, with a small override for the detectors whose
continue-training moved their last blocks: 741 MB downloaded against 1486 MB assembled, or 517 MB
for Panda alone.

```
manifest.json               what to fetch and how to reassemble it
dinov3_vitb16_trunk.pth     the shared trunk, 211 tensors
panda/, kuka/               per-robot keypoint heads, trunk overrides, angle and rotation heads
```

Fetch and assemble with the repository's own script, which reproduces the six checkpoints per
robot exactly:

```bash
python scripts/download_weights.py --robot panda
python scripts/pack_weights.py verify     # tensor-by-tensor proof that assembly round-trips
```
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-id", default=os.environ.get("DINOBOTPOSE_HF_REPO", "Najongs/dinobotpose"))
    ap.add_argument("--github", default="Najongs/dinobotpose")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not os.path.isdir(PARTS):
        sys.exit(f"{PARTS} does not exist — run: python scripts/pack_weights.py pack")
    files = sorted(os.path.relpath(os.path.join(dp, f), PARTS)
                   for dp, _, fs in os.walk(PARTS) for f in fs)
    total = sum(os.path.getsize(os.path.join(PARTS, f)) for f in files)
    card = CARD.format(repo=args.repo_id.split("/")[-1], gh=args.github)

    print(f"repo   {args.repo_id}")
    print(f"upload {len(files)} files, {total / 1e6:.0f} MB")
    for f in files:
        print(f"  {os.path.getsize(os.path.join(PARTS, f)) / 1e6:8.1f} MB  {f}")
    if args.dry_run:
        print("\n--- README.md (model card) ---\n")
        print(card)
        return 0

    try:
        from huggingface_hub import HfApi
    except ImportError:
        sys.exit("huggingface_hub is required: pip install huggingface_hub")
    api = HfApi()
    who = api.whoami()          # fails loudly if the token is missing or read-only
    print(f"\nauthenticated as {who.get('name')}")
    api.create_repo(args.repo_id, repo_type="model", exist_ok=True)
    card_path = os.path.join(PARTS, "README.md")
    with open(card_path, "w") as fh:
        fh.write(card)
    # The license must travel with the weights, not only with the code.
    lic = os.path.join(ROOT, "docs", "DINOV3_LICENSE.md")
    if os.path.isfile(lic):
        import shutil
        shutil.copyfile(lic, os.path.join(PARTS, "LICENSE.md"))
    api.upload_folder(folder_path=PARTS, repo_id=args.repo_id, repo_type="model",
                      commit_message="DINObotPose release weights (shared trunk + per-detector parts)")
    print(f"\nhttps://huggingface.co/{args.repo_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
