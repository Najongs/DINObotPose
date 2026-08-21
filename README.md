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
uv sync

# conda
conda env create -f environment.yml && conda activate dinobotpose

# plain pip
pip install -r requirements.txt
```

CUDA 12.8 wheels are pinned for torch and torchvision. The backbone architecture config ships in
`src/dinobotpose/assets/`, so **no HuggingFace access is needed at inference time**.

## Get the weights

```bash
python scripts/download_weights.py       # 743 MB per robot, into checkpoints/
python scripts/doctor.py                 # packages, GPU, checkpoints, dataset
```

`doctor.py` exits non-zero on the first blocking problem and lists your GPUs by UUID, which is the
only unambiguous way to pin a card on a multi-GPU host.

## Get the data

The evaluation runs on the [DREAM](https://github.com/NVlabs/DREAM) benchmark, which we cannot
redistribute. Download the Panda real and synthetic splits from NVIDIA, then arrange them so that
each split directory holds the per-frame annotation json files and each json's `meta.image_path`
resolves to the corresponding RGB image. `doctor.py --val-dir <split>` verifies exactly this and
reports the resolved image path, so run it before a long job.

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
bash scripts/reproduce_paper.sh /path/to/Converted_dataset
```

Roughly 6 to 9 hours on one RTX A6000; results accumulate in `results/summary.tsv` as each split
finishes, so the run can be interrupted and resumed.

## Licensing and attribution

- The detector checkpoints contain DINOv3 backbone weights (about 342 MB of each 359 MB file).
  DINOv3 is released by Meta under its own license; using or redistributing these checkpoints is
  subject to that license, and by downloading them you accept it.
- The DREAM benchmark is distributed by NVIDIA under its own terms.

## Citation

```bibtex
@inproceedings{dinobotpose,
  title     = {Geometry-Guided Monocular Articulated Robot Pose Estimation with Frozen Foundation Features},
  booktitle = {IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year      = {2026}
}
```
