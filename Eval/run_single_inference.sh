#!/bin/bash

# Single Image Inference Script for DINOv3 Pose Estimation

# Model configuration
MODEL_PATH="/home/najo/NAS/DIP/DINObotPose/TRAIN/outputs/dinov2_base_multi_robot_20260211_100644/best_model.pth"
MODEL_NAME="facebook/dinov3-vitb16-pretrain-lvd1689m"

# Input image (change this to your image path)
IMAGE_PATH="/home/najo/NAS/DIP/DINObotPose/DREAM/data/real/panda-3cam_azure/000000.rgb.jpg"

# Output directory
OUTPUT_DIR="./inference_output"

# Model parameters
IMAGE_SIZE=512
HEATMAP_SIZE=512

# Run inference
python inference_single_image.py \
    --model-path "$MODEL_PATH" \
    --model-name "$MODEL_NAME" \
    --image-path "$IMAGE_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --image-size $IMAGE_SIZE \
    --heatmap-size $HEATMAP_SIZE \
    --save-heatmaps \
    --save-combined

echo ""
echo "Inference completed! Check results in: $OUTPUT_DIR"
