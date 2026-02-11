#!/bin/bash

# DINOv3 Pose Estimation Visualization Script
# Visualizes model inference results with heatmaps and keypoint overlays

# ============================================================================
# Configuration
# ============================================================================

# Model checkpoint
CHECKPOINT_PATH="/home/najo/NAS/DIP/DINObotPose/TRAIN/outputs/dinov2_base_multi_robot_20260210_152621/epoch_5.pth"

# Dataset to visualize
DATASET_PATH="/home/najo/NAS/DIP/DINObotPose/DREAM/data"

# Output directory
OUTPUT_DIR="/home/najo/NAS/DIP/DINObotPose/Vis/output_$(date +%Y%m%d_%H%M%S)"

# Multi-robot settings (match your training settings)
MULTI_ROBOT="--multi-robot"
ROBOT_TYPES="--robot-types panda"

# Visualization types to create
# Options: kp_overlay, heatmap_overlay, separate_heatmaps
VIZ_TYPES="kp_overlay heatmap_overlay"

# Number of samples to visualize (0 = all)
NUM_SAMPLES=50

# Number of individual images to save
NUM_IMAGES_TO_SAVE=10

# Video framerate
FRAMERATE=10

# ============================================================================
# Run Visualization
# ============================================================================

echo "=========================================="
echo "DINOv3 Pose Estimation Visualization"
echo "=========================================="
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Dataset: $DATASET_PATH"
echo "Output: $OUTPUT_DIR"
echo "Visualization types: $VIZ_TYPES"
echo "Number of samples: $NUM_SAMPLES"

# Check if ffmpeg is installed
if ! command -v ffmpeg &> /dev/null; then
    echo ""
    echo "WARNING: ffmpeg is not installed"
    echo "  - Videos will not be created"
    echo "  - Only individual frames and images will be saved"
    echo "  - To install: sudo apt-get install ffmpeg"
fi

echo "=========================================="

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run visualization
python visualize_dinov3_inference.py \
    --checkpoint-path "$CHECKPOINT_PATH" \
    --dataset-path "$DATASET_PATH" \
    --output-dir "$OUTPUT_DIR" \
    $MULTI_ROBOT \
    $ROBOT_TYPES \
    --visualization-types $VIZ_TYPES \
    --num-samples $NUM_SAMPLES \
    --num-images-to-save $NUM_IMAGES_TO_SAVE \
    --framerate $FRAMERATE \
    --save-images

echo "=========================================="
echo "Visualization completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="

# Display created files
echo ""
echo "Created files:"
ls -lh "$OUTPUT_DIR"/*.mp4 2>/dev/null || echo "  No videos created"
ls -lh "$OUTPUT_DIR"/*.png 2>/dev/null | head -n 5
