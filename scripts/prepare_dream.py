#!/usr/bin/env python3
"""Build the evaluation index from a downloaded DREAM release.

DREAM ships each split as a directory of `NNNNNN.json` annotations beside `NNNNNN.rgb.jpg`
images, with the camera intrinsics factored out into a per-split `_camera_settings.json`. The
loader here wants one json per frame that carries its own intrinsics and a pointer to its image,
so this script writes a parallel index: the original `objects` and `sim_state` blocks are copied
through unchanged, and a `meta` block is added holding the 3x3 intrinsic matrix and a relative
path to the RGB file. No pixels are copied, so the index costs a few hundred MB at most.

    # after running DREAM's own data/DOWNLOAD.sh
    python scripts/prepare_dream.py --dream-root /path/to/DREAM/data --out /path/to/Converted_dataset

Verify a converted split before a long job:

    python scripts/doctor.py --val-dir /path/to/Converted_dataset/DREAM_real/panda-3cam_realsense

Only the Panda splits are indexed, and deliberately. DREAM keeps the KUKA frames next to their
images with the camera settings alongside, so the loader reads that tree directly; point
`--val-dir` at the downloaded `kuka_synth_test_dr` and nothing else is needed. Indexing it
anyway would change what the KUKA model is fed: without `meta.K` the loader hands the network an
identity K, which is what the released KUKA weights were trained and measured under, while the
solver reconstructs the true intrinsics separately from `_camera_settings.json`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# DREAM's own directory names, and where the index should land.
REAL_SPLITS = ["panda-3cam_azure", "panda-3cam_kinect360", "panda-3cam_realsense", "panda-orb"]
# Panda only. The KUKA splits are used exactly as downloaded -- see the note below.
SYN_SPLITS = ["panda_synth_test_dr", "panda_synth_test_photo",
              "panda_synth_train_dr"]   # the train split is needed only to retrain


def find_camera_settings(start: str, stop_after: int = 4) -> str | None:
    """Walk up from a split directory looking for _camera_settings.json (DREAM puts it one or
    two levels above the frames, depending on the split)."""
    cur = os.path.abspath(start)
    for _ in range(stop_after):
        cand = os.path.join(cur, "_camera_settings.json")
        if os.path.isfile(cand):
            return cand
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return None


def intrinsics_from(settings_path: str) -> list[list[float]]:
    with open(settings_path) as fh:
        cam = json.load(fh)["camera_settings"][0]["intrinsic_settings"]
    return [[cam["fx"], cam.get("s", 0.0), cam["cx"]],
            [0.0, cam["fy"], cam["cy"]],
            [0.0, 0.0, 1.0]]


def find_frame_dir(split_root: str) -> str | None:
    """The frames sometimes sit directly in the split dir and sometimes one level deeper in a
    directory repeating the split name (DREAM is inconsistent between real and synthetic)."""
    if any(f.endswith(".rgb.jpg") for f in os.listdir(split_root)):
        return split_root
    nested = os.path.join(split_root, os.path.basename(split_root))
    if os.path.isdir(nested) and any(f.endswith(".rgb.jpg") for f in os.listdir(nested)):
        return nested
    return None


def convert_split(split_root: str, out_dir: str) -> tuple[int, int]:
    frame_dir = find_frame_dir(split_root)
    if frame_dir is None:
        print(f"  [skip] no .rgb.jpg frames under {split_root}")
        return 0, 0
    settings = find_camera_settings(frame_dir)
    if settings is None:
        print(f"  [skip] no _camera_settings.json above {frame_dir}")
        return 0, 0
    K = intrinsics_from(settings)

    os.makedirs(out_dir, exist_ok=True)
    names = sorted(f for f in os.listdir(frame_dir)
                   if f.endswith(".json") and not f.startswith("_"))
    written = skipped = 0
    for name in names:
        image = os.path.join(frame_dir, name.replace(".json", ".rgb.jpg"))
        if not os.path.isfile(image):
            skipped += 1
            continue
        with open(os.path.join(frame_dir, name)) as fh:
            ann = json.load(fh)
        ann["meta"] = {"K": K, "image_path": os.path.relpath(image, out_dir)}
        with open(os.path.join(out_dir, name), "w") as fh:
            json.dump(ann, fh, indent=4)
        written += 1
    return written, skipped


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dream-root", required=True,
                    help="the directory DOWNLOAD.sh populated (holds real/ and synthetic/)")
    ap.add_argument("--out", required=True, help="where to write the index")
    args = ap.parse_args()

    real_src = os.path.join(args.dream_root, "real")
    syn_src = os.path.join(args.dream_root, "synthetic")
    if not os.path.isdir(real_src) and not os.path.isdir(syn_src):
        sys.exit(f"neither {real_src} nor {syn_src} exists — is --dream-root correct?")

    total = 0
    for src_root, splits, out_name in ((real_src, REAL_SPLITS, "DREAM_real"),
                                       (syn_src, SYN_SPLITS, "DREAM_to_DREAM_syn")):
        if not os.path.isdir(src_root):
            continue
        for split in splits:
            split_root = os.path.join(src_root, split)
            if not os.path.isdir(split_root):
                continue
            out_dir = os.path.join(args.out, out_name, split)
            print(f"[{out_name}/{split}]")
            written, skipped = convert_split(split_root, out_dir)
            if written:
                print(f"  {written} frames indexed" + (f", {skipped} without an image" if skipped else ""))
            total += written

    if total == 0:
        sys.exit("nothing was indexed — check --dream-root")
    print(f"\n{total} frames indexed under {args.out}")
    print("verify one split:  python scripts/doctor.py --val-dir "
          f"{os.path.join(args.out, 'DREAM_real', 'panda-3cam_realsense')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
