#!/bin/bash

# DINOv3 Pose Estimation Evaluation Script

# Path to the checkpoint you want to evaluate
CHECKPOINT="/home/najo/NAS/DIP/DINObotPose/TRAIN/outputs/dinov2_base_multi_robot_20260210_184510/best_model.pth"

# Path to the data directory (NDDS format)
# Example: "/home/najo/NAS/DIP/DINObotPose/DREAM/data/synthetic/panda_synth_test_dr/panda_synth_test_dr"
DATA_DIR="/home/najo/NAS/DIP/DINObotPose/DREAM/data/synthetic/panda_synth_test_dr/panda_synth_test_dr"

# Output directory for results
OUTPUT_DIR="./eval_outputs/epoch_5_results"

python evaluate.py \
    --checkpoint "$CHECKPOINT" \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --batch-size 8 \
    --num-workers 4
    # --multi-robot