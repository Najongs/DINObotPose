"""
DINOv3 Pose Estimation Model Visualization Script
Visualizes trained model inference results with heatmaps and keypoint overlays
"""

import argparse
import os
import sys
from pathlib import Path
import subprocess
import shutil

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image as PILImage, ImageDraw, ImageFont
from tqdm import tqdm
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'TRAIN'))
from model import DINOv3PoseEstimator
from dataset import PoseEstimationDataset


def check_ffmpeg_installed():
    """Check if ffmpeg is installed"""
    try:
        result = subprocess.run(['ffmpeg', '-version'],
                              capture_output=True,
                              timeout=5)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def video_from_frames(frames_dir, video_output_path, video_framerate=30):
    """Create video from frames using ffmpeg"""
    # Check if ffmpeg is installed
    if not check_ffmpeg_installed():
        print("=" * 80)
        print("WARNING: ffmpeg is not installed!")
        print("Videos cannot be created, but individual frames have been saved.")
        print("")
        print("To install ffmpeg:")
        print("  Ubuntu/Debian: sudo apt-get install ffmpeg")
        print("  macOS: brew install ffmpeg")
        print("  Or use --skip-video flag to suppress this warning")
        print("=" * 80)
        return False

    force_str = "-y"
    loglevel_str = "-loglevel 24"
    framerate_str = f"-framerate {video_framerate}"
    input_data_str = f'-pattern_type glob -i "{os.path.join(frames_dir, "*.png")}"'
    output_vid_str = f'"{video_output_path}"'
    encoding_str = "-vcodec libx264 -pix_fmt yuv420p"

    ffmpeg_vid_cmd = (
        f"ffmpeg {force_str} {loglevel_str} {framerate_str} "
        f"{input_data_str} {encoding_str} {output_vid_str}"
    )

    print(f"Running command: {ffmpeg_vid_cmd}")
    result = subprocess.call(ffmpeg_vid_cmd, shell=True)
    return result == 0


def heatmap_to_colormap(heatmap, colormap='hot'):
    """Convert grayscale heatmap to colormap"""
    # Normalize heatmap to [0, 1]
    heatmap = heatmap - heatmap.min()
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    # Apply colormap (use new matplotlib API)
    try:
        cmap = plt.colormaps.get_cmap(colormap)
    except AttributeError:
        # Fallback for older matplotlib versions
        cmap = cm.get_cmap(colormap)

    colored = cmap(heatmap)

    # Convert to PIL Image (RGB)
    colored_rgb = (colored[:, :, :3] * 255).astype(np.uint8)
    return PILImage.fromarray(colored_rgb)


def extract_keypoints_from_heatmaps(heatmaps):
    """
    Extract keypoint locations from heatmaps using argmax

    Args:
        heatmaps: (N, H, W) heatmaps

    Returns:
        keypoints: (N, 2) keypoint locations [x, y]
    """
    N, H, W = heatmaps.shape
    keypoints = np.zeros((N, 2), dtype=np.float32)

    for i in range(N):
        heatmap = heatmaps[i]
        # Find maximum location
        max_idx = np.argmax(heatmap)
        y = max_idx // W
        x = max_idx % W
        keypoints[i] = [x, y]

    return keypoints


def overlay_points_on_image(image, keypoints, color=(255, 0, 0), radius=8, thickness=2, with_text=True):
    """
    Overlay keypoint circles on image

    Args:
        image: PIL Image
        keypoints: (N, 2) array of [x, y] coordinates
        color: RGB color tuple
        radius: circle radius
        thickness: circle line thickness
        with_text: whether to add text labels

    Returns:
        PIL Image with overlays
    """
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)

    # Try to load a font (use default if not available)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        font = ImageFont.load_default()

    for i, (x, y) in enumerate(keypoints):
        if x < 0 or y < 0:  # Invalid keypoint
            continue

        # Draw circle
        bbox = [x - radius, y - radius, x + radius, y + radius]
        draw.ellipse(bbox, outline=color, width=thickness)

        # Draw text label
        if with_text:
            text = str(i)
            draw.text((x + radius + 5, y - radius), text, fill=color, font=font)

    return img_draw


def visualize_network_inference(args):
    """Main visualization function"""

    # Check inputs
    assert os.path.exists(args.checkpoint_path), f'Checkpoint "{args.checkpoint_path}" does not exist.'
    assert os.path.exists(args.dataset_path), f'Dataset path "{args.dataset_path}" does not exist.'

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine visualization types
    do_kp_overlay = 'kp_overlay' in args.visualization_types
    do_heatmap_overlay = 'heatmap_overlay' in args.visualization_types
    do_separate_heatmaps = 'separate_heatmaps' in args.visualization_types

    videos_to_make = []
    save_images = args.save_images

    if do_kp_overlay:
        idx_kp_overlay = len(videos_to_make)
        videos_to_make.append({
            'frames_dir': os.path.join(args.output_dir, 'frames_kp_overlay'),
            'output_path': os.path.join(args.output_dir, 'kp_overlay.mp4'),
            'frame': None,
        })
        os.makedirs(videos_to_make[-1]['frames_dir'], exist_ok=True)

    if do_heatmap_overlay:
        idx_heatmap_overlay = len(videos_to_make)
        videos_to_make.append({
            'frames_dir': os.path.join(args.output_dir, 'frames_heatmap_overlay'),
            'output_path': os.path.join(args.output_dir, 'heatmap_overlay.mp4'),
            'frame': None,
        })
        os.makedirs(videos_to_make[-1]['frames_dir'], exist_ok=True)

    if do_separate_heatmaps:
        idx_separate_heatmaps = len(videos_to_make)
        videos_to_make.append({
            'frames_dir': os.path.join(args.output_dir, 'frames_separate_heatmaps'),
            'output_path': os.path.join(args.output_dir, 'separate_heatmaps.mp4'),
            'frame': None,
        })
        os.makedirs(videos_to_make[-1]['frames_dir'], exist_ok=True)

    if len(videos_to_make) == 0:
        print("No visualizations selected. Exiting.")
        return

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu', weights_only=False)

    # Get config
    config = checkpoint.get('config', {})
    model_name = config.get('model_name', 'facebook/dinov3-vitb16-pretrain-lvd1689m')
    heatmap_size = (config.get('heatmap_size', 512), config.get('heatmap_size', 512))
    keypoint_names = config.get('keypoint_names', [
        'panda_link0', 'panda_link2', 'panda_link3',
        'panda_link4', 'panda_link6', 'panda_link7', 'panda_hand'
    ])

    print(f"Model: {model_name}")
    print(f"Heatmap size: {heatmap_size}")
    print(f"Keypoints: {keypoint_names}")

    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = DINOv3PoseEstimator(
        dino_model_name=model_name,
        heatmap_size=heatmap_size
    ).to(device)

    # Load model weights (handle DDP wrapper)
    state_dict = checkpoint['model_state_dict']
    if any(k.startswith('module.') for k in state_dict.keys()):
        # Remove 'module.' prefix from DDP checkpoint
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()
    print("Model loaded successfully")

    # Create dataset
    print(f"\nLoading dataset from {args.dataset_path}")
    dataset = PoseEstimationDataset(
        data_dir=args.dataset_path,
        keypoint_names=keypoint_names,
        image_size=(512, 512),
        heatmap_size=heatmap_size,
        augment=False,
        multi_robot=args.multi_robot,
        robot_types=args.robot_types
    )

    print(f"Dataset size: {len(dataset)} samples")

    # Limit number of samples if specified
    num_samples = min(args.num_samples, len(dataset)) if args.num_samples > 0 else len(dataset)
    print(f"Visualizing {num_samples} samples")

    # Process dataset
    sample_results = []

    print("\nRunning inference...")
    with torch.no_grad():
        for idx in tqdm(range(num_samples)):
            sample = dataset[idx]

            # Get image tensor and move to device
            image_tensor = sample['image'].unsqueeze(0).to(device)  # (1, 3, H, W)

            # Run inference
            pred_heatmaps, pred_angles = model(image_tensor)

            # Move to CPU and convert to numpy
            pred_heatmaps = pred_heatmaps[0].cpu().numpy()  # (N, H, W)
            pred_angles = pred_angles[0].cpu().numpy()  # (9,)

            # Extract keypoints from heatmaps
            pred_keypoints = extract_keypoints_from_heatmaps(pred_heatmaps)

            # Get ground truth keypoints (if available)
            gt_keypoints = sample['keypoints'].numpy()  # (N, 2)

            # Load original image for visualization
            sample_info = dataset.samples[idx]
            image_raw = PILImage.open(sample_info['image_path']).convert('RGB')

            # Scale keypoints to original image size
            orig_w, orig_h = image_raw.size
            scale_x = orig_w / heatmap_size[1]
            scale_y = orig_h / heatmap_size[0]

            pred_keypoints_scaled = pred_keypoints.copy()
            pred_keypoints_scaled[:, 0] *= scale_x
            pred_keypoints_scaled[:, 1] *= scale_y

            gt_keypoints_scaled = gt_keypoints.copy()
            gt_keypoints_scaled[:, 0] *= scale_x
            gt_keypoints_scaled[:, 1] *= scale_y

            sample_results.append({
                'image_raw': image_raw,
                'pred_heatmaps': pred_heatmaps,
                'pred_keypoints': pred_keypoints_scaled,
                'gt_keypoints': gt_keypoints_scaled,
                'pred_angles': pred_angles,
                'sample_name': sample['name']
            })

    # Create visualizations
    print("\nCreating visualizations...")
    for idx, result in enumerate(tqdm(sample_results)):
        image_raw = result['image_raw']
        pred_heatmaps = result['pred_heatmaps']
        pred_keypoints = result['pred_keypoints']
        gt_keypoints = result['gt_keypoints']

        frame_filename = f"{str(idx+1).zfill(6)}.png"

        # 1. Keypoint overlay visualization
        if do_kp_overlay:
            img_kp = image_raw.copy()

            # Overlay ground truth (green)
            if not args.no_ground_truth:
                img_kp = overlay_points_on_image(
                    img_kp, gt_keypoints,
                    color=(0, 255, 0), radius=10, thickness=3, with_text=False
                )

            # Overlay predictions (red)
            img_kp = overlay_points_on_image(
                img_kp, pred_keypoints,
                color=(255, 0, 0), radius=8, thickness=2, with_text=True
            )

            videos_to_make[idx_kp_overlay]['frame'] = img_kp
            img_kp.save(os.path.join(videos_to_make[idx_kp_overlay]['frames_dir'], frame_filename))

            # Save some images separately
            if save_images and idx < args.num_images_to_save:
                img_kp.save(os.path.join(args.output_dir, f'kp_overlay_{idx:04d}.png'))

        # 2. Heatmap overlay visualization
        if do_heatmap_overlay:
            # Sum all heatmaps
            summed_heatmap = pred_heatmaps.sum(axis=0)

            # Resize heatmap to match image size
            heatmap_resized = cv2.resize(summed_heatmap, image_raw.size, interpolation=cv2.INTER_LINEAR)

            # Convert to colormap
            heatmap_colored = heatmap_to_colormap(heatmap_resized, colormap='hot')

            # Blend with original image
            img_heatmap = PILImage.blend(image_raw, heatmap_colored, alpha=0.5)

            # Overlay keypoints
            if not args.no_ground_truth:
                img_heatmap = overlay_points_on_image(
                    img_heatmap, gt_keypoints,
                    color=(0, 255, 0), radius=10, thickness=3, with_text=False
                )

            img_heatmap = overlay_points_on_image(
                img_heatmap, pred_keypoints,
                color=(255, 0, 0), radius=8, thickness=2, with_text=True
            )

            videos_to_make[idx_heatmap_overlay]['frame'] = img_heatmap
            img_heatmap.save(os.path.join(videos_to_make[idx_heatmap_overlay]['frames_dir'], frame_filename))

            if save_images and idx < args.num_images_to_save:
                img_heatmap.save(os.path.join(args.output_dir, f'heatmap_overlay_{idx:04d}.png'))

        # 3. Separate heatmaps visualization
        if do_separate_heatmaps:
            num_kps = len(pred_heatmaps)
            cols = 4
            rows = (num_kps + cols - 1) // cols

            fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
            axes = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes

            for i in range(num_kps):
                ax = axes[i]
                heatmap = pred_heatmaps[i]
                ax.imshow(heatmap, cmap='hot', interpolation='bilinear')
                ax.set_title(f'{keypoint_names[i]}', fontsize=10)
                ax.axis('off')

                # Mark predicted location
                kp_x, kp_y = pred_keypoints[i] / np.array([image_raw.size[0] / heatmap_size[1],
                                                           image_raw.size[1] / heatmap_size[0]])
                ax.plot(kp_x, kp_y, 'r+', markersize=15, markeredgewidth=3)

            # Hide extra subplots
            for i in range(num_kps, len(axes)):
                axes[i].axis('off')

            plt.tight_layout()

            # Save to buffer and convert to PIL
            fig.canvas.draw()
            img_separate = PILImage.frombytes('RGB', fig.canvas.get_width_height(),
                                             fig.canvas.tostring_rgb())
            plt.close(fig)

            videos_to_make[idx_separate_heatmaps]['frame'] = img_separate
            img_separate.save(os.path.join(videos_to_make[idx_separate_heatmaps]['frames_dir'], frame_filename))

            if save_images and idx < args.num_images_to_save:
                img_separate.save(os.path.join(args.output_dir, f'separate_heatmaps_{idx:04d}.png'))

    # Create videos from frames
    videos_created = []
    if not args.skip_video:
        print("\nCreating videos...")
        ffmpeg_available = check_ffmpeg_installed()

        if not ffmpeg_available:
            print("\nffmpeg not found - skipping video creation")
            print("Frame directories preserved for manual video creation")
        else:
            for video in videos_to_make:
                print(f"Creating video: {video['output_path']}")
                success = video_from_frames(video['frames_dir'], video['output_path'], args.framerate)

                if success:
                    videos_created.append(video['output_path'])
                    # Optionally clean up frame directories
                    if not args.keep_frames:
                        shutil.rmtree(video['frames_dir'])
                        print(f"Cleaned up frames directory: {video['frames_dir']}")
                else:
                    print(f"Failed to create video: {video['output_path']}")
                    print(f"Frames preserved at: {video['frames_dir']}")

    print(f"\nVisualization complete! Results saved to: {args.output_dir}")

    # Print summary
    print("\n" + "=" * 80)
    print("Summary:")
    print(f"  Total samples processed: {num_samples}")
    print(f"  Individual images saved: {min(args.num_images_to_save, num_samples) if args.save_images else 0}")

    if not args.skip_video:
        if videos_created:
            print(f"  Videos created: {len(videos_created)}")
            for video_path in videos_created:
                print(f"    - {video_path}")
        else:
            print("  Videos created: 0 (ffmpeg not available)")
            print(f"  Frame directories preserved:")
            for video in videos_to_make:
                if os.path.exists(video['frames_dir']):
                    num_frames = len(list(Path(video['frames_dir']).glob('*.png')))
                    print(f"    - {video['frames_dir']} ({num_frames} frames)")

    print("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Visualize DINOv3 Pose Estimation Model Inference',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        '-c', '--checkpoint-path',
        required=True,
        help='Path to model checkpoint (.pth file)'
    )
    parser.add_argument(
        '-d', '--dataset-path',
        required=True,
        help='Path to dataset directory'
    )
    parser.add_argument(
        '-o', '--output-dir',
        required=True,
        help='Path to output directory for visualizations'
    )

    # Dataset options
    parser.add_argument(
        '--multi-robot',
        action='store_true',
        help='Load data from multiple robot subdirectories'
    )
    parser.add_argument(
        '--robot-types',
        nargs='+',
        default=None,
        help='Filter specific robot types (e.g., panda kuka baxter)'
    )

    # Visualization options
    parser.add_argument(
        '-v', '--visualization-types',
        nargs='+',
        choices=['kp_overlay', 'heatmap_overlay', 'separate_heatmaps'],
        default=['kp_overlay', 'heatmap_overlay'],
        help='Types of visualizations to create'
    )
    parser.add_argument(
        '--no-ground-truth',
        action='store_true',
        help='Do not show ground truth keypoints'
    )

    # Output options
    parser.add_argument(
        '--save-images',
        action='store_true',
        default=True,
        help='Save individual images in addition to videos'
    )
    parser.add_argument(
        '--num-images-to-save',
        type=int,
        default=10,
        help='Number of individual images to save'
    )
    parser.add_argument(
        '--skip-video',
        action='store_true',
        help='Skip video creation (only save frames/images)'
    )
    parser.add_argument(
        '--keep-frames',
        action='store_true',
        help='Keep frame directories after video creation'
    )
    parser.add_argument(
        '-fps', '--framerate',
        type=float,
        default=10.0,
        help='Framerate for output videos'
    )

    # Processing options
    parser.add_argument(
        '-n', '--num-samples',
        type=int,
        default=0,
        help='Number of samples to process (0 = all)'
    )

    args = parser.parse_args()

    visualize_network_inference(args)
