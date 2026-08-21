#!/usr/bin/env python3
"""Check that this clone can actually run: packages, GPU, checkpoints, dataset.

Exits non-zero on the first blocking problem, so it is safe to chain:

    python scripts/doctor.py && bash scripts/reproduce_paper.sh
"""
from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (import name, pip name, minimum version or None)
PACKAGES = [
    ("torch", "torch", "2.0"),
    ("torchvision", "torchvision", None),
    ("numpy", "numpy", None),
    ("cv2", "opencv-python-headless", None),
    ("PIL", "pillow", None),
    ("tqdm", "tqdm", None),
    ("transformers", "transformers", "4.56"),   # DINOv3ViTModel
    ("albumentations", "albumentations", None),
]

CHECKPOINTS = {
    "panda": ["pass1_detector", "pass1_angle", "pass1_rotation",
              "pass2_detector", "pass2_angle", "pass2_rotation"],
    "kuka": ["pass1_detector", "pass1_angle", "pass1_rotation",
             "pass2_detector", "pass2_angle", "pass2_rotation"],
}

OK, BAD, WARN = "  ok  ", " FAIL ", " warn "
_failed = False


def report(status: str, what: str, detail: str = "") -> None:
    global _failed
    if status is BAD:
        _failed = True
    print(f"[{status}] {what}" + (f"  {detail}" if detail else ""))


def _version_at_least(have: str, want: str) -> bool:
    def parts(v):
        out = []
        for chunk in v.split(".")[:3]:
            digits = "".join(c for c in chunk if c.isdigit())
            out.append(int(digits) if digits else 0)
        return out
    return parts(have) >= parts(want)


def check_packages() -> None:
    print("\npackages")
    for mod, pip_name, minimum in PACKAGES:
        try:
            m = importlib.import_module(mod)
        except ImportError:
            report(BAD, pip_name, "not installed  (uv sync  |  pip install -r requirements.txt)")
            continue
        version = getattr(m, "__version__", "?")
        if minimum and version != "?" and not _version_at_least(version, minimum):
            report(BAD, pip_name, f"{version}  (needs >= {minimum})")
        else:
            report(OK, pip_name, version)


def check_gpu() -> None:
    print("\ngpu")
    try:
        import torch
    except ImportError:
        report(BAD, "torch", "not installed, cannot probe GPUs")
        return
    if not torch.cuda.is_available():
        report(BAD, "cuda", "torch reports no CUDA device")
        return
    report(OK, "cuda", f"torch {torch.__version__}, cuda {torch.version.cuda}")
    # Report by UUID: on multi-GPU boxes the nvidia-smi index and the CUDA_VISIBLE_DEVICES
    # integer index can disagree, so pinning by UUID is the only reliable form.
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid,name,memory.used,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15).stdout.strip()
        for line in out.splitlines():
            uuid, name, used, total = [c.strip() for c in line.split(",")]
            free = int(total.split()[0]) - int(used.split()[0])
            print(f"         {uuid}  {name}  free {free} MiB")
        print("         pin one with:  CUDA_VISIBLE_DEVICES=GPU-<uuid>")
    except Exception as exc:  # nvidia-smi missing is not fatal
        report(WARN, "nvidia-smi", f"unavailable ({type(exc).__name__})")


def check_checkpoints() -> None:
    print("\ncheckpoints")
    root = os.path.join(ROOT, "checkpoints")
    if not os.path.isdir(root):
        report(BAD, "checkpoints/", "missing  (python scripts/download_weights.py)")
        return
    for robot, names in CHECKPOINTS.items():
        missing = [n for n in names
                   if not os.path.isfile(os.path.join(root, robot, f"{n}.pth"))]
        if missing:
            report(BAD, f"{robot}/", f"missing {', '.join(missing)}")
        else:
            total = sum(os.path.getsize(os.path.join(root, robot, f"{n}.pth")) for n in names)
            report(OK, f"{robot}/", f"6 files, {total / 1e6:.0f} MB")


def check_backbone_config() -> None:
    print("\nbackbone")
    cfg = os.path.join(ROOT, "src", "dinobotpose", "assets", "dinov3_vitb16_config.json")
    if os.path.isfile(cfg):
        report(OK, "dinov3 config", "bundled, no HuggingFace access needed")
    else:
        report(BAD, "dinov3 config", f"missing at {cfg}")


def check_dataset(val_dir: str | None) -> None:
    print("\ndataset")
    if not val_dir:
        report(WARN, "split", "not checked  (pass --val-dir to verify)")
        return
    if not os.path.isdir(val_dir):
        report(BAD, "split", f"{val_dir} is not a directory")
        return
    jsons = [f for f in os.listdir(val_dir) if f.endswith(".json") and not f.startswith("_")]
    if not jsons:
        report(BAD, "split", f"{val_dir} holds no frame json files")
        return
    report(OK, "split", f"{len(jsons)} frames in {val_dir}")
    # The json files are only an index; the images live in a separate tree that meta.image_path
    # points at. A split that indexes images it cannot reach fails deep inside the loader, so
    # resolve one here where the error is still legible.
    import json
    sample = os.path.join(val_dir, sorted(jsons)[0])
    with open(sample) as fh:
        meta = json.load(fh).get("meta", {})
    rel = meta.get("image_path")
    if not rel:
        report(WARN, "image_path", f"absent from {os.path.basename(sample)}")
        return
    resolved = os.path.normpath(os.path.join(val_dir, rel.replace("../dataset/", "../../../")))
    if os.path.isfile(resolved):
        report(OK, "images", f"resolved e.g. {resolved}")
    else:
        report(BAD, "images", f"{os.path.basename(sample)} points at {resolved}, which is missing")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--val-dir", default=None,
                    help="a split directory to verify, e.g. .../panda-3cam_realsense")
    args = ap.parse_args()

    print(f"dinobotpose doctor — {ROOT}")
    check_packages()
    check_gpu()
    check_backbone_config()
    check_checkpoints()
    check_dataset(args.val_dir)

    print()
    if _failed:
        print("DOCTOR: blocked — fix the FAIL lines above.")
        return 1
    print("DOCTOR: all green.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
