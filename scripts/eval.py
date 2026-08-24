#!/usr/bin/env python3
"""Evaluate DINObotPose on a DREAM split, in the deployed configuration.

This is a thin, opinionated wrapper: it fixes every flag to the values the paper reports with,
so a reproduction cannot silently drift by flipping a knob. The underlying evaluators still take
those flags if you want to ablate (see scripts/selfbbox_eval.py and scripts/kuka_autobbox_eval.py).

    python scripts/eval.py --robot panda --val-dir .../panda-3cam_realsense
    python scripts/eval.py --robot panda --val-dir .../panda_synth_test_photo --occlude-ratio 0.4
    python scripts/eval.py --robot kuka  --val-dir .../kuka_synth_test_dr
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def pick_gpu() -> str | None:
    """Most-free card, addressed by UUID.

    Integer CUDA indices and nvidia-smi indices can disagree on multi-GPU hosts, and picking the
    wrong card silently lands the job on a busy one. UUIDs are unambiguous.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid,memory.used,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15).stdout.strip()
    except Exception:
        return None
    best, best_free = None, -1
    for line in out.splitlines():
        uuid, used, total = [c.strip() for c in line.split(",")]
        free = int(total.split()[0]) - int(used.split()[0])
        if free > best_free:
            best, best_free = uuid, free
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot", choices=["panda", "kuka"], required=True)
    ap.add_argument("--val-dir", required=True, help="split directory of frame json files")
    ap.add_argument("--max-frames", type=int, default=0, help="0 = the full test set")
    ap.add_argument("--batch-size", type=int, default=24,
                    help="frames are fitted independently, so this only trades memory for speed")
    ap.add_argument("--occlude-ratio", type=float, default=0.0,
                    help="occlusion protocol: fraction of the robot region to cover")
    ap.add_argument("--gpu", default=None, help="GPU UUID; default is the most free card")
    ap.add_argument("--dump-npz", default=None, help="write per-frame predictions here")
    args = ap.parse_args()

    ck = os.path.join(ROOT, "checkpoints", args.robot)
    missing = [n for n in ("pass1_detector", "pass1_angle", "pass1_rotation",
                           "pass2_detector", "pass2_angle", "pass2_rotation")
               if not os.path.isfile(os.path.join(ck, f"{n}.pth"))]
    if missing:
        sys.exit(f"missing checkpoints in {ck}: {', '.join(missing)}\n"
                 f"run: python scripts/download_weights.py --robot {args.robot}")

    if args.robot == "panda":
        cmd = [sys.executable, os.path.join(ROOT, "scripts", "selfbbox_eval.py"),
               "--stage1-detector", f"{ck}/pass1_detector.pth",
               "--stage1-angle", f"{ck}/pass1_angle.pth",
               "--stage1-rot", f"{ck}/pass1_rotation.pth",
               "--crop-detector", f"{ck}/pass2_detector.pth",
               "--crop-angle", f"{ck}/pass2_angle.pth",
               "--rot-head", f"{ck}/pass2_rotation.pth"]
    else:
        cmd = [sys.executable, os.path.join(ROOT, "scripts", "kuka_autobbox_eval.py"),
               "--stage1-detector", f"{ck}/pass1_detector.pth",
               "--stage1-angle", f"{ck}/pass1_angle.pth",
               "--stage1-rot", f"{ck}/pass1_rotation.pth",
               "--crop-detector", f"{ck}/pass2_detector.pth",
               "--crop-angle", f"{ck}/pass2_angle.pth",
               "--rot-head", f"{ck}/pass2_rotation.pth"]

    # The deployed configuration, fixed. See configs/deployed.yaml for what is absent and why.
    # --iters is stated explicitly because the two underlying evaluators historically defaulted
    # differently (200 against 250); the deployed budget is 200 for both robots.
    cmd += ["--bbox-from-solved", "--conf-gate", "0.0", "--iters", "200",
            "--val-dir", args.val_dir,
            "--max-frames", str(args.max_frames),
            "--batch-size", str(args.batch_size)]
    if args.robot == "panda":
        cmd += ["--dark-decode"]
        if args.occlude_ratio > 0:
            cmd += ["--occlude-ratio", str(args.occlude_ratio)]
    elif args.occlude_ratio > 0:
        sys.exit("--occlude-ratio is only wired for the panda evaluator")
    if args.dump_npz:
        cmd += ["--dump-npz", args.dump_npz]

    env = dict(os.environ)
    gpu = args.gpu or pick_gpu()
    if gpu:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("HF_HUB_OFFLINE", "1")   # the backbone config ships with the package

    print(f"[eval] {args.robot}  {os.path.basename(args.val_dir.rstrip('/'))}"
          f"  frames={'all' if args.max_frames == 0 else args.max_frames}"
          f"  occ={args.occlude_ratio}  gpu={gpu or 'default'}", flush=True)

    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    sys.stdout.write(proc.stdout[-4000:])
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr[-4000:])
        return proc.returncode

    hits = re.findall(r"ADD-AUC@100mm[: ]+([0-9.]+)", proc.stdout)
    if hits:
        print(f"\nADD-AUC@100mm = {float(hits[-1]) * 100:.2f}")
    else:
        print("\n(no ADD-AUC line found in the evaluator output)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
