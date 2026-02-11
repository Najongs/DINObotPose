"""
Single Image Inference Script for DINOv3 Pose Estimation
Runs inference on a single image and visualizes/saves the results
"""

import argparse
import os
import sys
from pathlib import Path
import json

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms
import cv2

# Import DREAM utilities
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../DREAM')))
import dream

# Import model from TRAIN directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../TRAIN')))
from model import DINOv3PoseEstimator


def get_keypoints_from_heatmaps(heatmaps: torch.Tensor) -> np.ndarray:
    """
    Extract keypoint coordinates from heatmaps using argmax.

    Args:
        heatmaps: (B, N, H, W) tensor

    Returns:
        keypoints: (B, N, 2) numpy array [x, y]
    """
    B, N, H, W = heatmaps.shape
    heatmaps_flat = heatmaps.view(B, N, -1)
    max_indices = torch.argmax(heatmaps_flat, dim=-1)

    y = max_indices // W
    x = max_indices % W

    keypoints = torch.stack([x, y], dim=-1).float()
    return keypoints.cpu().numpy()


def visualize_keypoints(
    image: np.ndarray,
    keypoints: np.ndarray,
    keypoint_names: list,
    confidences: np.ndarray = None,
    skeleton: list = None
) -> np.ndarray:
    """
    Visualize keypoints on image.

    Args:
        image: RGB image (H, W, 3)
        keypoints: (N, 2) keypoint coordinates [x, y]
        keypoint_names: List of keypoint names
        confidences: (N,) confidence scores (optional)
        skeleton: List of (idx1, idx2) pairs for skeleton connections

    Returns:
        Visualized image
    """
    vis_image = image.copy()

    # Define colors for different keypoints
    colors = [
        (255, 0, 0),      # Red
        (0, 255, 0),      # Green
        (0, 0, 255),      # Blue
        (255, 255, 0),    # Yellow
        (255, 0, 255),    # Magenta
        (0, 255, 255),    # Cyan
        (255, 128, 0),    # Orange
        (128, 0, 255),    # Purple
        (0, 128, 255),    # Light Blue
    ]

    # Draw skeleton connections if provided
    if skeleton:
        for idx1, idx2 in skeleton:
            if idx1 < len(keypoints) and idx2 < len(keypoints):
                pt1 = tuple(keypoints[idx1].astype(int))
                pt2 = tuple(keypoints[idx2].astype(int))
                if pt1[0] > 0 and pt1[1] > 0 and pt2[0] > 0 and pt2[1] > 0:
                    cv2.line(vis_image, pt1, pt2, (128, 128, 128), 2)

    # Draw keypoints
    for i, (kp, name) in enumerate(zip(keypoints, keypoint_names)):
        x, y = int(kp[0]), int(kp[1])

        if x > 0 and y > 0:  # Valid keypoint
            color = colors[i % len(colors)]

            # Draw circle
            cv2.circle(vis_image, (x, y), 5, color, -1)
            cv2.circle(vis_image, (x, y), 7, (255, 255, 255), 2)

            # Draw label
            label = name
            if confidences is not None:
                label += f" ({confidences[i]:.2f})"

            cv2.putText(
                vis_image,
                label,
                (x + 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                2
            )
            cv2.putText(
                vis_image,
                label,
                (x + 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1
            )

    return vis_image


def visualize_heatmaps(
    heatmaps: torch.Tensor,
    keypoint_names: list,
    original_size: tuple
) -> np.ndarray:
    """
    Visualize heatmaps as a grid.

    Args:
        heatmaps: (N, H, W) heatmaps
        keypoint_names: List of keypoint names
        original_size: (width, height) of original image

    Returns:
        Grid visualization of heatmaps
    """
    N, H, W = heatmaps.shape

    # Calculate grid size
    n_cols = min(4, N)
    n_rows = (N + n_cols - 1) // n_cols

    # Resize heatmaps to match original image aspect ratio
    aspect_ratio = original_size[0] / original_size[1]
    if aspect_ratio > 1:
        cell_w = 200
        cell_h = int(200 / aspect_ratio)
    else:
        cell_h = 200
        cell_w = int(200 * aspect_ratio)

    # Create grid
    grid = np.zeros((n_rows * cell_h, n_cols * cell_w, 3), dtype=np.uint8)

    for i in range(N):
        row = i // n_cols
        col = i % n_cols

        # Get heatmap
        heatmap = heatmaps[i].cpu().numpy()

        # Normalize to 0-255
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        heatmap = (heatmap * 255).astype(np.uint8)

        # Apply colormap
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # Resize
        heatmap_resized = cv2.resize(heatmap_color, (cell_w, cell_h))

        # Add label
        cv2.putText(
            heatmap_resized,
            keypoint_names[i],
            (5, 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            2
        )
        cv2.putText(
            heatmap_resized,
            keypoint_names[i],
            (5, 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 0),
            1
        )

        # Place in grid
        y_start = row * cell_h
        x_start = col * cell_w
        grid[y_start:y_start+cell_h, x_start:x_start+cell_w] = heatmap_resized

    return grid


@torch.no_grad()
def inference_single_image(args):
    """Run inference on a single image"""

    # Check input
    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"Image not found: {args.image_path}")

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Keypoint names (Panda robot)
    keypoint_names = [
        'panda_link0', 'panda_link2', 'panda_link3',
        'panda_link4', 'panda_link6', 'panda_link7', 'panda_hand'
    ]

    # Skeleton connections for Panda robot
    skeleton = [
        (0, 1),  # link0 -> link2
        (1, 2),  # link2 -> link3
        (2, 3),  # link3 -> link4
        (3, 4),  # link4 -> link6
        (4, 5),  # link6 -> link7
        (5, 6),  # link7 -> hand
    ]

    # Load model
    print(f"\nLoading model from {args.model_path}")
    model = DINOv3PoseEstimator(
        dino_model_name=args.model_name,
        heatmap_size=(args.heatmap_size, args.heatmap_size),
        unfreeze_blocks=0
    ).to(device)

    # Load checkpoint
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        state_dict = checkpoint

    # Handle DDP wrapper
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Load and preprocess image
    print(f"\nLoading image: {args.image_path}")
    image_pil = Image.open(args.image_path).convert('RGB')
    original_size = image_pil.size  # (width, height)
    print(f"Original image size: {original_size}")

    # Store original image as numpy array
    image_np = np.array(image_pil)

    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image_tensor = transform(image_pil).unsqueeze(0).to(device)

    # Run inference
    print("\nRunning inference...")
    pred_heatmaps, pred_angles = model(image_tensor)

    # Extract keypoints
    pred_keypoints = get_keypoints_from_heatmaps(pred_heatmaps)[0]  # (N, 2)

    # Scale keypoints to original image size
    scale_x = original_size[0] / args.heatmap_size
    scale_y = original_size[1] / args.heatmap_size

    pred_keypoints_scaled = pred_keypoints.copy()
    pred_keypoints_scaled[:, 0] *= scale_x
    pred_keypoints_scaled[:, 1] *= scale_y

    # Get predicted angles
    pred_angles_np = pred_angles[0].cpu().numpy()

    # Print results
    print("\n" + "=" * 80)
    print("INFERENCE RESULTS")
    print("=" * 80)

    print("\nPredicted Keypoints (original image coordinates):")
    for i, (name, kp) in enumerate(zip(keypoint_names, pred_keypoints_scaled)):
        print(f"  {i+1}. {name:20s}: ({kp[0]:7.2f}, {kp[1]:7.2f})")

    print("\nPredicted Joint Angles (radians):")
    for i, angle in enumerate(pred_angles_np):
        print(f"  Joint {i+1}: {angle:8.5f} rad ({np.degrees(angle):7.2f}°)")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save results
    base_name = Path(args.image_path).stem

    # 1. Visualize keypoints on image
    vis_keypoints = visualize_keypoints(
        image_np,
        pred_keypoints_scaled,
        keypoint_names,
        skeleton=skeleton
    )

    keypoints_path = output_dir / f"{base_name}_keypoints.jpg"
    cv2.imwrite(str(keypoints_path), cv2.cvtColor(vis_keypoints, cv2.COLOR_RGB2BGR))
    print(f"\nSaved keypoint visualization: {keypoints_path}")

    # 2. Visualize heatmaps
    if args.save_heatmaps:
        heatmaps_vis = visualize_heatmaps(
            pred_heatmaps[0],
            keypoint_names,
            original_size
        )

        heatmaps_path = output_dir / f"{base_name}_heatmaps.jpg"
        cv2.imwrite(str(heatmaps_path), heatmaps_vis)
        print(f"Saved heatmap visualization: {heatmaps_path}")

    # 3. Save raw results as JSON
    results = {
        'image_path': str(args.image_path),
        'image_size': original_size,
        'keypoints': [
            {
                'name': name,
                'x': float(kp[0]),
                'y': float(kp[1])
            }
            for name, kp in zip(keypoint_names, pred_keypoints_scaled)
        ],
        'joint_angles_rad': pred_angles_np.tolist(),
        'joint_angles_deg': np.degrees(pred_angles_np).tolist(),
        'model_path': str(args.model_path),
        'model_name': args.model_name
    }

    json_path = output_dir / f"{base_name}_results.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Saved JSON results: {json_path}")

    # 4. Create combined visualization
    if args.save_combined:
        # Create combined image with original, keypoints, and heatmaps
        h_orig, w_orig = image_np.shape[:2]

        # Resize images to same height
        target_h = 480
        scale = target_h / h_orig
        target_w = int(w_orig * scale)

        img_resized = cv2.resize(image_np, (target_w, target_h))
        kp_resized = cv2.resize(vis_keypoints, (target_w, target_h))

        if args.save_heatmaps:
            # Calculate width for heatmaps
            hm_h, hm_w = heatmaps_vis.shape[:2]
            hm_scale = target_h / hm_h
            hm_w_new = int(hm_w * hm_scale)
            hm_resized = cv2.resize(heatmaps_vis, (hm_w_new, target_h))

            # Combine
            combined = np.hstack([img_resized, kp_resized, hm_resized])
        else:
            combined = np.hstack([img_resized, kp_resized])

        combined_path = output_dir / f"{base_name}_combined.jpg"
        cv2.imwrite(str(combined_path), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
        print(f"Saved combined visualization: {combined_path}")

    print("\n" + "=" * 80)
    print(f"All results saved to: {output_dir}")
    print("=" * 80)

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Single Image Inference for DINOv3 Pose Estimation'
    )

    # Model
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--model-name', type=str,
                        default='facebook/dinov3-vitb16-pretrain-lvd1689m',
                        help='DINOv3 model name')
    parser.add_argument('--image-size', type=int, default=512,
                        help='Input image size')
    parser.add_argument('--heatmap-size', type=int, default=512,
                        help='Output heatmap size')

    # Input/Output
    parser.add_argument('--image-path', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--output-dir', type=str, default='./inference_output',
                        help='Output directory for results')

    # Visualization options
    parser.add_argument('--save-heatmaps', action='store_true', default=True,
                        help='Save heatmap visualizations')
    parser.add_argument('--save-combined', action='store_true', default=True,
                        help='Save combined visualization')
    parser.add_argument('--no-heatmaps', action='store_false', dest='save_heatmaps',
                        help='Disable heatmap visualization')
    parser.add_argument('--no-combined', action='store_false', dest='save_combined',
                        help='Disable combined visualization')

    args = parser.parse_args()

    inference_single_image(args)


if __name__ == '__main__':
    main()
