"""
Network Inference Script for DINOv3 Pose Estimation
Similar to DREAM's network_inference.py
"""

import argparse
import math
import os
import sys
from PIL import Image as PILImage

import numpy as np
import torch

# Import DREAM utilities
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../DREAM')))
import dream

# Import model from TRAIN directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../TRAIN')))
from model import DINOv3PoseEstimator

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def get_keypoints_from_heatmaps(heatmaps_tensor):
    """
    Extract keypoint coordinates from heatmaps using argmax.
    Args:
        heatmaps_tensor: (B, N, H, W) tensor
    Returns:
        keypoints: (N, 2) numpy array [x, y]
    """
    B, N, H, W = heatmaps_tensor.shape
    heatmaps_flat = heatmaps_tensor.view(B, N, -1)
    max_indices = torch.argmax(heatmaps_flat, dim=-1)

    y = max_indices // W
    x = max_indices % W

    keypoints = torch.stack([x, y], dim=-1).float()
    return keypoints[0].cpu().numpy()  # Return first batch


def generate_belief_map_visualizations(
    belief_maps, keypoint_projs_detected, keypoint_projs_gt=None
):
    """Generate belief map visualizations with keypoints overlaid"""

    belief_map_images = dream.image_proc.images_from_belief_maps(
        belief_maps, normalization_method=6
    )

    belief_map_images_kp = []
    for kp in range(len(keypoint_projs_detected)):
        if keypoint_projs_gt:
            keypoint_projs = [keypoint_projs_gt[kp], keypoint_projs_detected[kp]]
            color = ["green", "red"]
        else:
            keypoint_projs = [keypoint_projs_detected[kp]]
            color = "red"
        belief_map_image_kp = dream.image_proc.overlay_points_on_image(
            belief_map_images[kp],
            keypoint_projs,
            annotation_color_dot=color,
            annotation_color_text=color,
            point_diameter=4,
        )
        belief_map_images_kp.append(belief_map_image_kp)

    n_cols = int(math.ceil(len(keypoint_projs_detected) / 2.0))
    belief_maps_kp_mosaic = dream.image_proc.mosaic_images(
        belief_map_images_kp,
        rows=2,
        cols=n_cols,
        inner_padding_px=10,
        fill_color_rgb=(0, 0, 0),
    )
    return belief_maps_kp_mosaic


def network_inference(args):

    # Input argument handling
    assert os.path.exists(
        args.input_params_path
    ), 'Expected input_params_path "{}" to exist, but it does not.'.format(
        args.input_params_path
    )

    # Create output directory if specified
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        print("# Saving visualizations to:  {} ...".format(args.output_dir))

    assert os.path.exists(
        args.image_path
    ), 'Expected image_path "{}" to exist, but it does not.'.format(args.image_path)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("# Using device: {}".format(device))

    # Keypoint names
    keypoint_names = [
        'panda_link0', 'panda_link2', 'panda_link3',
        'panda_link4', 'panda_link6', 'panda_link7', 'panda_hand'
    ]

    # Load network
    print("# Creating network...")
    model = DINOv3PoseEstimator(
        dino_model_name=args.model_name,
        heatmap_size=(args.heatmap_size, args.heatmap_size),
        unfreeze_blocks=0
    ).to(device)

    print("Loading network with weights from:  {} ...".format(args.input_params_path))
    checkpoint = torch.load(args.input_params_path, map_location=device, weights_only=False)

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Handle DDP wrapper
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Load in image
    print("# Loading image:  {} ...".format(args.image_path))
    image_rgb_OrigInput_asPilImage = PILImage.open(args.image_path).convert("RGB")
    orig_image_dim = tuple(image_rgb_OrigInput_asPilImage.size)  # (width, height)

    # Image preprocessing
    import torchvision.transforms as TVTransforms
    transform = TVTransforms.Compose([
        TVTransforms.Resize((args.image_size, args.image_size)),
        TVTransforms.ToTensor(),
        TVTransforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Process image for network input
    print("Detecting keypoints...")
    image_tensor = transform(image_rgb_OrigInput_asPilImage).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_heatmaps, pred_angles = model(image_tensor)

    # Extract keypoint coordinates from heatmaps
    kp_coords_wrtNetOutput_asArray = get_keypoints_from_heatmaps(pred_heatmaps)

    # Scale keypoints to original image size
    scale_x = orig_image_dim[0] / args.heatmap_size
    scale_y = orig_image_dim[1] / args.heatmap_size

    kp_coords_wrtOrigInput_asArray = kp_coords_wrtNetOutput_asArray.copy()
    kp_coords_wrtOrigInput_asArray[:, 0] *= scale_x
    kp_coords_wrtOrigInput_asArray[:, 1] *= scale_y

    print(
        "Detected keypoints in input image:\n{}".format(kp_coords_wrtOrigInput_asArray)
    )

    # Scale keypoints to network input size
    input_image_dim = (args.image_size, args.image_size)
    scale_to_input_x = input_image_dim[0] / args.heatmap_size
    scale_to_input_y = input_image_dim[1] / args.heatmap_size

    kp_coords_wrtNetInput_asArray = kp_coords_wrtNetOutput_asArray.copy()
    kp_coords_wrtNetInput_asArray[:, 0] *= scale_to_input_x
    kp_coords_wrtNetInput_asArray[:, 1] *= scale_to_input_y

    # Get network input image
    image_rgb_NetInput_asPilImage = image_rgb_OrigInput_asPilImage.resize(
        input_image_dim, resample=PILImage.BILINEAR
    )

    # Read in keypoints if provided
    if args.keypoints_path:
        print(
            "# Loading ground truth keypoints from {} ...".format(args.keypoints_path)
        )
        keypoints_gt = dream.utilities.load_keypoints(
            args.keypoints_path,
            "panda",
            keypoint_names,
        )
        kp_coords_gt_wrtOrig = keypoints_gt["projections"]
        print(
            "Ground truth keypoints in input image:\n{}".format(
                np.array(kp_coords_gt_wrtOrig)
            )
        )

        # Scale GT keypoints to network input size
        kp_coords_gt_wrtNetInput_asList = []
        for kp_gt in kp_coords_gt_wrtOrig:
            x_scaled = kp_gt[0] * input_image_dim[0] / orig_image_dim[0]
            y_scaled = kp_gt[1] * input_image_dim[1] / orig_image_dim[1]
            kp_coords_gt_wrtNetInput_asList.append([x_scaled, y_scaled])

        # Scale GT keypoints to network output size
        kp_coords_gt_wrtNetOutput_asList = []
        for kp_gt in kp_coords_gt_wrtOrig:
            x_scaled = kp_gt[0] * args.heatmap_size / orig_image_dim[0]
            y_scaled = kp_gt[1] * args.heatmap_size / orig_image_dim[1]
            kp_coords_gt_wrtNetOutput_asList.append([x_scaled, y_scaled])
    else:
        print("# Not loading ground truth keypoints (not provided)")
        kp_coords_gt_wrtNetInput_asList = None
        kp_coords_gt_wrtNetOutput_asList = None

    # Generate visualization output 1: keypoints on network input image
    keypoints_wrtNetInput_overlay = dream.image_proc.overlay_points_on_image(
        image_rgb_NetInput_asPilImage,
        kp_coords_gt_wrtNetInput_asList,
        keypoint_names,
        annotation_color_dot="green",
        annotation_color_text="white",
    )
    keypoints_wrtNetInput_overlay = dream.image_proc.overlay_points_on_image(
        keypoints_wrtNetInput_overlay,
        kp_coords_wrtNetInput_asArray,
        keypoint_names,
        annotation_color_dot="red",
        annotation_color_text="white",
    )
    if args.output_dir:
        output_path = os.path.join(args.output_dir, "01_keypoints_on_net_input.png")
        keypoints_wrtNetInput_overlay.save(output_path)
        print("  Saved: {}".format(output_path))
    else:
        keypoints_wrtNetInput_overlay.show(
            title="Keypoints (possibly with ground truth) on net input image"
        )

    # Generate visualization output 2: mosaic of raw belief maps
    belief_maps_overlay = generate_belief_map_visualizations(
        pred_heatmaps[0],
        kp_coords_wrtNetOutput_asArray,
        kp_coords_gt_wrtNetOutput_asList,
    )
    if args.output_dir:
        output_path = os.path.join(args.output_dir, "02_belief_map_mosaic.png")
        belief_maps_overlay.save(output_path)
        print("  Saved: {}".format(output_path))
    else:
        belief_maps_overlay.show(title="Belief map output mosaic")

    # Generate visualization output 3: mosaic of belief maps overlaid on input image
    belief_maps_wrtNetOutput_asListOfPilImages = dream.image_proc.images_from_belief_maps(
        pred_heatmaps[0], normalization_method=6
    )
    blended_array = []

    for n in range(len(kp_coords_wrtNetOutput_asArray)):
        bm_wrtNetOutput_asPilImage = belief_maps_wrtNetOutput_asListOfPilImages[n]
        kp = kp_coords_wrtNetInput_asArray[n]
        fname = keypoint_names[n]

        bm_wrtNetInput_asPilImage = bm_wrtNetOutput_asPilImage.resize(
            input_image_dim, resample=PILImage.BILINEAR
        )
        blended = PILImage.blend(
            image_rgb_NetInput_asPilImage, bm_wrtNetInput_asPilImage, alpha=0.5
        )
        blended = dream.image_proc.overlay_points_on_image(
            blended,
            [kp],
            [fname],
            annotation_color_dot="red",
            annotation_color_text="white",
        )
        blended_array.append(blended)

    n_cols = int(math.ceil(len(kp_coords_wrtNetOutput_asArray) / 2.0))
    belief_maps_with_kp_overlaid_mosaic = dream.image_proc.mosaic_images(
        blended_array, rows=2, cols=n_cols, fill_color_rgb=(0, 0, 0)
    )
    if args.output_dir:
        output_path = os.path.join(args.output_dir, "03_belief_maps_with_keypoints_mosaic.png")
        belief_maps_with_kp_overlaid_mosaic.save(output_path)
        print("  Saved: {}".format(output_path))
    else:
        belief_maps_with_kp_overlaid_mosaic.show(
            title="Mosaic of belief maps, with keypoints, on original"
        )

    # Squash belief maps into one combined image
    belief_map_combined_wrtNetOutput_asTensor = pred_heatmaps[0].sum(dim=0)
    belief_map_combined_wrtNetOutput_asPilImage = dream.image_proc.image_from_belief_map(
        belief_map_combined_wrtNetOutput_asTensor, normalization_method=6
    )
    belief_map_combined_wrtNetInput_asPilImage = belief_map_combined_wrtNetOutput_asPilImage.resize(
        input_image_dim, resample=PILImage.BILINEAR
    )

    # Generate visualization output 4: combined belief map on network input
    belief_map_combined_wrtNetInput_overlay = PILImage.blend(
        image_rgb_NetInput_asPilImage,
        belief_map_combined_wrtNetInput_asPilImage,
        alpha=0.5,
    )
    belief_map_combined_wrtNetInput_overlay = dream.image_proc.overlay_points_on_image(
        belief_map_combined_wrtNetInput_overlay,
        kp_coords_wrtNetInput_asArray,
        keypoint_names,
        annotation_color_dot="red",
        annotation_color_text="white",
    )
    if args.output_dir:
        output_path = os.path.join(args.output_dir, "04_belief_maps_on_net_input.png")
        belief_map_combined_wrtNetInput_overlay.save(output_path)
        print("  Saved: {}".format(output_path))
    else:
        belief_map_combined_wrtNetInput_overlay.show(
            title="Belief maps, with keypoints, on net input image"
        )

    # Generate visualization output 5: combined belief map on original image
    belief_map_combined_wrtOrigInput_asPilImage = belief_map_combined_wrtNetOutput_asPilImage.resize(
        orig_image_dim, resample=PILImage.BILINEAR
    )
    belief_map_combined_wrtOrigInput_overlay = PILImage.blend(
        image_rgb_OrigInput_asPilImage,
        belief_map_combined_wrtOrigInput_asPilImage,
        alpha=0.5,
    )
    belief_map_combined_wrtOrigInput_overlay = dream.image_proc.overlay_points_on_image(
        belief_map_combined_wrtOrigInput_overlay,
        kp_coords_wrtOrigInput_asArray,
        keypoint_names,
        annotation_color_dot="red",
        annotation_color_text="white",
    )
    if args.output_dir:
        output_path = os.path.join(args.output_dir, "05_belief_maps_on_original.png")
        belief_map_combined_wrtOrigInput_overlay.save(output_path)
        print("  Saved: {}".format(output_path))
    else:
        belief_map_combined_wrtOrigInput_overlay.show(
            title="Belief maps, with keypoints, on original image"
        )

    print("Done.")


if __name__ == "__main__":

    print(
        "---------- Running 'network_inference.py' -------------------------------------------------"
    )

    # Parse input arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-i",
        "--input-params-path",
        required=True,
        help="Path to network parameters file.",
    )
    parser.add_argument(
        "-m", "--image_path", required=True, help="Path to image used for inference."
    )
    parser.add_argument(
        "-k",
        "--keypoints_path",
        default=None,
        help="Path to NDDS dataset with ground truth keypoints information.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help="Directory to save visualization images. If not specified, images will be displayed instead.",
    )
    parser.add_argument(
        "--model-name",
        default="facebook/dinov3-vitb16-pretrain-lvd1689m",
        help="DINOv3 model name",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=512,
        help="Input image size",
    )
    parser.add_argument(
        "--heatmap-size",
        type=int,
        default=512,
        help="Output heatmap size",
    )
    parser.add_argument(
        "-g",
        "--gpu-ids",
        nargs="+",
        type=int,
        default=None,
        help="The GPU IDs (ignored for now, kept for compatibility).",
    )
    parser.add_argument(
        "-p",
        "--image-preproc-override",
        default=None,
        help="Image preprocessing override (ignored for now, kept for compatibility).",
    )
    args = parser.parse_args()

    # Run network inference
    network_inference(args)
