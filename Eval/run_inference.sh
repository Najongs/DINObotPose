#!/bin/bash

# Dataset Inference Script for DINOv3 Pose Estimation
# Evaluates trained model on panda-3cam_azure dataset

# Model configuration
MODEL_PATH="/home/najo/NAS/DIP/DINObotPose/TRAIN/outputs/dinov2_base_multi_robot_20260211_100644/best_model.pth"
MODEL_NAME="facebook/dinov3-vitb16-pretrain-lvd1689m"

# Dataset
DATASET_DIR="/home/najo/NAS/DIP/DINObotPose/DREAM/data/real/panda-3cam_azure"

# Output
OUTPUT_DIR="/home/najo/NAS/DIP/DINObotPose/TRAIN/eval_outputs"

# Inference parameters
BATCH_SIZE=16
NUM_WORKERS=4
IMAGE_SIZE=512
HEATMAP_SIZE=512

# Metrics thresholds (matching DREAM defaults)
KP_AUC_THRESHOLD=20.0    # pixels
ADD_AUC_THRESHOLD=0.1    # meters

# Run inference
python inference_dataset.py \
    --model-path "$MODEL_PATH" \
    --model-name "$MODEL_NAME" \
    --dataset-dir "$DATASET_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --batch-size $BATCH_SIZE \
    --num-workers $NUM_WORKERS \
    --image-size $IMAGE_SIZE \
    --heatmap-size $HEATMAP_SIZE \
    --kp-auc-threshold $KP_AUC_THRESHOLD \
    --add-auc-threshold $ADD_AUC_THRESHOLD
