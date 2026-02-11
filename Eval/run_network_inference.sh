#!/bin/bash

# Network Inference Script (DREAM-style)

# Model path
MODEL_PATH="/home/najo/NAS/DIP/DINObotPose/TRAIN/outputs/dinov2_base_multi_robot_20260211_100644/best_model.pth"

# Input image
IMAGE_PATH="/home/najo/NAS/DIP/DINObotPose/DREAM/data/real/panda-3cam_azure/000000.rgb.jpg"

# Ground truth keypoints (optional)
KEYPOINTS_PATH="/home/najo/NAS/DIP/DINObotPose/DREAM/data/real/panda-3cam_azure/000000.json"

# Output directory
OUTPUT_DIR="./network_inference_output"

# Run inference
python network_inference.py \
    -i "$MODEL_PATH" \
    -m "$IMAGE_PATH" \
    -k "$KEYPOINTS_PATH" \
    -o "$OUTPUT_DIR" \
    --model-name facebook/dinov3-vitb16-pretrain-lvd1689m \
    --image-size 512 \
    --heatmap-size 512

echo ""
echo "Inference completed! Check results in: $OUTPUT_DIR"
echo "Generated files:"
echo "  01_keypoints_on_net_input.png - Keypoints overlaid on network input"
echo "  02_belief_map_mosaic.png - Raw belief maps (heatmaps)"
echo "  03_belief_maps_with_keypoints_mosaic.png - Belief maps with keypoints on input"
echo "  04_belief_maps_on_net_input.png - Combined belief map on network input"
echo "  05_belief_maps_on_original.png - Combined belief map on original image"
