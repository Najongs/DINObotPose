#!/usr/bin/env python3
"""Train the pipeline, stage by stage, with the hyperparameters the released weights were made with.

The three training scripts under train/ disagree on flag names (`--data-dir` against `--train-dir`,
`--learning-rate` against `--lr`) and each stage needs the previous stage's checkpoint, so running
them by hand is where a reproduction goes wrong. This wrapper fixes both.

    python scripts/train.py --robot panda --data-root /path/to/Converted_dataset --stage all
    python scripts/train.py --robot panda --data-root ... --stage s2_final --gpu GPU-xxxx
    python scripts/train.py --robot panda --data-root ... --stage all --dry-run   # print, run nothing

Stages, in dependency order:

  s1_detector   full-frame keypoint detector, backbone frozen
  s1_heads      angle and rotation heads on that detector
  s2_base       crop-stage detector, backbone frozen
  s2_viewpoint  continue-train under wide viewpoint and scale augmentation, last 4 blocks open
  s2_final      continue-train the above under distractor-occlusion augmentation  (deployed)
  s2_heads      re-cascade the angle and rotation heads onto the final detector   (deployed)

Two things about this recipe are worth knowing before comparing against the released weights.
The trunk of the released pass-1 detector is bit-identical to stock DINOv3, so `s1_detector`
keeps it frozen; the two continue-training stages open the last four blocks, which is the only
place any trunk weight moves. And the released checkpoints are the end of a longer warm-started
history than this script replays: `s1_detector` and `s2_base` here start from the pretrained
backbone in one run, where the released ones were resumed across several. Expect the lineage and
the ordering to reproduce, not the last decimal.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(ROOT, "train")
BACKBONE = "facebook/dinov3-vitb16-pretrain-lvd1689m"

KUKA_KP = ",".join(f"iiwa7_link_{i}" for i in range(1, 8))

# Where each robot's splits live under --data-root, and the split each stage validates on.
LAYOUT = {
    "panda": {"train": "DREAM_to_DREAM_syn/panda_synth_train_dr",
              "val_syn": "DREAM_to_DREAM_syn/panda_synth_test_dr",
              "val_real": "DREAM_real/panda-3cam_realsense",
              "keypoints": None},
    # KUKA is synthetic only, so its detector validates on the synthetic test split too. Its
    # splits are read straight from the download rather than from the index (prepare_dream.py
    # explains why), so --data-root for KUKA is DREAM's own synthetic directory.
    "kuka": {"train": "kuka_synth_train_dr",
             "val_syn": "kuka_synth_test_dr",
             "val_real": "kuka_synth_test_dr",
             "keypoints": KUKA_KP},
}

STAGES = ["s1_detector", "s1_heads", "s2_base", "s2_viewpoint", "s2_final", "s2_heads"]

# Augmentation shared by both continue-training stages: wide rotation, perspective and shear,
# and enough downscaling to imitate a small or distant robot.
VIEWPOINT_AUG = ["--aug-level", "strong_vp", "--crop-margin", "2.8",
                 "--crop-aspect-jitter", "0.30", "--crop-res-jitter", "0.65", "--crop-res-min", "90"]
# Added on top for the deployed detector: distractors pasted over the robot, and truncation
# at the frame boundary.
OCCLUSION_AUG = ["--distractor-occ-prob", "0.5", "--distractor-occ-max-frac", "0.35",
                 "--frame-boundary-prob", "0.3"]


def out_dir(robot: str, stage: str, kind: str = "") -> str:
    return os.path.join(ROOT, "outputs", robot, stage + (f"_{kind}" if kind else ""))


def detector_cmd(robot, paths, out, warm, crop, unfreeze, epochs, extra):
    cmd = [sys.executable, os.path.join(TRAIN, "train_heatmap.py"),
           "--data-dir", paths["train"],
           "--val-dir", paths["val_real"],
           "--model-name", BACKBONE, "--output-dir", out,
           "--image-size", "512", "--heatmap-size", "512",
           "--unfreeze-blocks", str(unfreeze),
           "--epochs", str(epochs), "--batch-size", "32", "--num-workers", "12",
           "--learning-rate", "1e-4", "--min-lr", "1e-7", "--weight-decay", "1e-5",
           "--occlusion-prob", "0.0", "--fda-prob", "0.0",
           "--amp", "--auto-resume", "--ckpt-every", "800"]
    if unfreeze > 0:
        cmd += ["--backbone-lr", "1e-5"]
    if crop:
        cmd += ["--crop-to-robot", "--crop-aspect", "1.3333333"]
    if warm:
        cmd += ["--checkpoint", warm]
    if paths["keypoints"]:
        cmd += ["--keypoint-names", paths["keypoints"]]
    return cmd + extra


def head_cmd(kind, robot, paths, detector, out, epochs, lr, init, occlude):
    script = "train_angle.py" if kind == "angle" else "train_rotation.py"
    cmd = [sys.executable, os.path.join(TRAIN, script),
           "--detector-ckpt", detector,
           "--train-dir", paths["train"], "--val-dir", paths["val_syn"],
           "--output-dir", out, "--model-name", BACKBONE,
           "--image-size", "512", "--batch-size", "32",
           "--epochs", str(epochs), "--lr", lr, "--min-lr", "1e-6",
           "--weight-decay", "1e-4", "--num-workers", "8", "--fk-robot", robot]
    cmd += ["--fk-weight", "10.0", "--head-type", "mlp"] if kind == "angle" else ["--t-weight", "50.0"]
    if occlude:
        cmd += ["--crop-to-robot", "--crop-margin", "1.5", "--occlude-aug", "0.3", "--auto-resume"]
    if init:
        cmd += ["--init-head", init]
    if paths["keypoints"]:
        cmd += ["--keypoint-names", paths["keypoints"]]
    return cmd


def best(path, name):
    """Where the previous stage leaves its checkpoint. Existence is checked at run time, so
    --dry-run can print the whole plan before any of it has been produced."""
    return os.path.join(path, name)


def plan(robot, paths, stage):
    """(label, command, output dir) for one stage, resolving upstream checkpoints."""
    d1, d2b, d2v, d2f = (out_dir(robot, s) for s in
                         ("s1_detector", "s2_base", "s2_viewpoint", "s2_final"))
    if stage == "s1_detector":
        return [("s1_detector",
                 detector_cmd(robot, paths, d1, None, crop=False, unfreeze=0, epochs=40, extra=
                              ["--aug-level", "strong"]), d1)]
    if stage == "s2_base":
        return [("s2_base",
                 detector_cmd(robot, paths, d2b, None, crop=True, unfreeze=0, epochs=50, extra=
                              ["--aug-level", "strong", "--crop-margin", "1.5"]), d2b)]
    if stage == "s2_viewpoint":
        return [("s2_viewpoint",
                 detector_cmd(robot, paths, d2v, best(d2b, "best_heatmap.pth"), crop=True,
                              unfreeze=4, epochs=12, extra=VIEWPOINT_AUG), d2v)]
    if stage == "s2_final":
        return [("s2_final",
                 detector_cmd(robot, paths, d2f, best(d2v, "best_heatmap.pth"), crop=True,
                              unfreeze=4, epochs=12, extra=VIEWPOINT_AUG + OCCLUSION_AUG), d2f)]
    if stage in ("s1_heads", "s2_heads"):
        crop = stage == "s2_heads"
        det = best(d2f if crop else d1, "best_heatmap.pth")
        # The re-cascade warm-starts from the pass-1 heads; the pass-1 heads start cold.
        init_a = best(out_dir(robot, "s1_heads", "angle"), "best_angle_head.pth") if crop else None
        init_r = best(out_dir(robot, "s1_heads", "rot"), "best_rot_head.pth") if crop else None
        ep, lr = (20, "5e-4") if crop else (60, "1e-3")
        return [(f"{stage}:angle",
                 head_cmd("angle", robot, paths, det, out_dir(robot, stage, "angle"),
                          ep, lr, init_a, crop), out_dir(robot, stage, "angle")),
                (f"{stage}:rot",
                 head_cmd("rot", robot, paths, det, out_dir(robot, stage, "rot"),
                          ep if crop else 30, lr, init_r, crop), out_dir(robot, stage, "rot"))]
    raise SystemExit(f"unknown stage {stage}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot", choices=["panda", "kuka"], required=True)
    ap.add_argument("--data-root", required=True,
                    help="the tree prepare_dream.py wrote (holds DREAM_real/ and DREAM_to_DREAM_syn/)")
    ap.add_argument("--stage", default="all", choices=["all", *STAGES])
    ap.add_argument("--gpu", default=None, help="GPU UUID; default is the most free card")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    paths = {k: (os.path.join(args.data_root, v) if k != "keypoints" else v)
             for k, v in LAYOUT[args.robot].items()}
    for k in ("train", "val_syn", "val_real"):
        if not os.path.isdir(paths[k]):
            sys.exit(f"missing split: {paths[k]}\nrun scripts/prepare_dream.py first")

    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from eval import pick_gpu
    gpu = args.gpu or pick_gpu()
    env = dict(os.environ, HF_HUB_OFFLINE="1", WANDB_MODE=os.environ.get("WANDB_MODE", "offline"),
               PYTORCH_ALLOC_CONF="expandable_segments:True")
    if gpu:
        env["CUDA_VISIBLE_DEVICES"] = gpu

    for stage in (STAGES if args.stage == "all" else [args.stage]):
        for label, cmd, out in plan(args.robot, paths, stage):
            print(f"\n=== {label} -> {out}", flush=True)
            if args.dry_run:
                print("    " + " ".join(cmd))
                continue
            for flag in ("--checkpoint", "--detector-ckpt", "--init-head"):
                if flag in cmd and not os.path.isfile(cmd[cmd.index(flag) + 1]):
                    sys.exit(f"{label}: {flag} {cmd[cmd.index(flag) + 1]} does not exist"
                             f" — run the earlier stage first")
            os.makedirs(out, exist_ok=True)
            log = os.path.join(out, "train.log")
            with open(log, "a") as fh:
                rc = subprocess.run(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
            print(f"    rc={rc}  log={log}")
            if rc != 0:
                return rc
    return 0


if __name__ == "__main__":
    sys.exit(main())
