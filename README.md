# DINObotPose

Monocular robot pose and joint-angle estimation by iterative model fitting on frozen foundation
features. From a single RGB image, with no encoder readings and no ground-truth bounding box, the
pipeline recovers the six-degree-of-freedom camera-to-robot pose together with the joint angles.

This repository reproduces every measured number in the paper.

## What it does

Sub-pixel keypoints are read from a frozen DINOv3 backbone. A first pass fits the whole frame and
projects its own skeleton to define a crop for a second pass, so the pipeline produces the bounding
box that competing methods take from ground truth or an external detector. An iterative fit then
recovers the joint angles together with the camera pose from those keypoints alone, minimizing a
robustly weighted reprojection error through differentiable forward kinematics.

No depth model is trained, no weights are updated on the evaluated data, and one configuration is
used for every camera and both robots.

## Install

Either path works; both pin the exact versions the paper was measured with.

```bash
# uv
uv sync                     # add --extra train to also install the training deps

# conda
conda env create -f environment.yml && conda activate dinobotpose

# plain pip
pip install -r requirements.txt
```

CUDA 12.8 wheels are pinned for torch and torchvision. The backbone architecture config ships in
`src/dinobotpose/assets/`, so **no HuggingFace access is needed at inference time**.

## Get the weights

```bash
python scripts/download_weights.py       # 741 MB for both robots, into checkpoints/
python scripts/doctor.py                 # packages, GPU, checkpoints, dataset
```

The DINOv3 trunk ships once rather than once per detector, with a small override for the
detectors whose continue-training moved their last blocks, so the download is half the size of
the assembled tree (741 MB against 1486 MB; 517 MB for Panda alone). `download_weights.py`
reassembles the six checkpoints per robot; `python scripts/pack_weights.py verify` is the
tensor-by-tensor proof that reassembly reproduces the originals exactly.

`doctor.py` exits non-zero on the first blocking problem and lists your GPUs by UUID, which is the
only unambiguous way to pin a card on a multi-GPU host.

## Get the data

The evaluation runs on the [DREAM](https://github.com/NVlabs/DREAM) benchmark, which we cannot
redistribute. Download it with DREAM's own `data/DOWNLOAD.sh`, then build the evaluation index:

```bash
python scripts/prepare_dream.py --dream-root /path/to/DREAM/data --out /path/to/Converted_dataset
python scripts/doctor.py --val-dir /path/to/Converted_dataset/DREAM_real/panda-3cam_realsense
```

`prepare_dream.py` copies each frame's `objects` and `sim_state` through unchanged and adds a
`meta` block holding the 3x3 intrinsics (read from the split's `_camera_settings.json`) and a
relative path to the RGB file. No pixels are copied. `doctor.py --val-dir` then resolves one
image end to end, which is the cheap way to catch a broken layout before a multi-hour job.

Only the Panda splits are indexed. DREAM keeps the KUKA frames beside their images, so the loader
reads that tree directly: point `--val-dir` at the downloaded `kuka_synth_test_dr` itself. The
script's header explains why indexing it anyway would change what the KUKA model is fed.

## Run

```bash
python scripts/eval.py --robot panda --val-dir .../panda-3cam_realsense
python scripts/eval.py --robot kuka  --val-dir .../kuka_synth_test_dr
python scripts/eval.py --robot panda --val-dir .../panda_synth_test_photo --occlude-ratio 0.4
```

`eval.py` fixes every flag to the deployed configuration so a reproduction cannot drift by
flipping a knob. `configs/deployed.yaml` records that configuration, including what the fit does
*not* use and the measurement that justified each removal.

To reproduce the paper's tables end to end:

```bash
bash scripts/reproduce_paper.sh /path/to/Converted_dataset /path/to/DREAM/data/synthetic
```

Roughly 6 to 9 hours on one RTX A6000; results accumulate in `results/summary.tsv` as each split
finishes, so the run can be interrupted and resumed.

## Retrain

```bash
python scripts/train.py --robot panda --data-root /path/to/Converted_dataset --stage all
python scripts/train.py --robot panda --data-root ... --stage all --dry-run   # print, run nothing
```

Six stages run in dependency order: the full-frame detector and its heads, then the crop-stage
detector, its two continue-training passes under viewpoint and occlusion augmentation, and the
head re-cascade onto the result. `--stage <name>` runs one of them. The wrapper carries each
stage's checkpoint into the next and fixes the hyperparameters the released weights were made
with; `train/` holds the three underlying scripts if you want to vary them. Training needs the
synthetic train split, so pass `prepare_dream.py` a download that includes it.

## Licensing and attribution

**Built with DINOv3.**

The code in this repository is MIT (see `LICENSE`). The released **weights are not**: they contain
the DINOv3 trunk (343 MB of the 741 MB download), unchanged in its first two thirds and
continue-trained in its last blocks, which makes them a derivative of the DINOv3 Materials. They
are therefore distributed under the [DINOv3 License](https://ai.meta.com/resources/models-and-libraries/dinov3-license/),
a copy of which ships as `docs/DINOV3_LICENSE.md` and travels with the weights; downloading them
is acceptance of it. Redistributing them onward is permitted under that same agreement, and
carries its obligations with it, including the "Built with DINOv3" notice and its acceptable-use
terms. Work published using them should acknowledge DINOv3 \[Siméoni et al., 2025\].

The DREAM benchmark is distributed by NVIDIA under its own terms and is not redistributed here.

## Citation

```bibtex
@inproceedings{dinobotpose,
  title     = {Geometry-Guided Monocular Articulated Robot Pose Estimation with Frozen Foundation Features},
  booktitle = {IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year      = {2026}
}
```
