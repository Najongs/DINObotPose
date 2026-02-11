import os
import torch
import numpy as np
from tqdm import tqdm
import argparse
import json
from pathlib import Path
import sys

# DREAM 라이브러리 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../DREAM')))

from TRAIN.model import DINOv3PoseEstimator
from TRAIN.dataset import PoseEstimationDataset
from torch.utils.data import DataLoader

import dream
from dream import analysis as dream_analysis

def get_keypoints_from_heatmaps(heatmaps):
    """
    Extract keypoint coordinates from heatmaps using argmax.
    heatmaps: (B, N, H, W)
    Returns: (B, N, 2) [x, y] coordinates
    """
    B, N, H, W = heatmaps.shape
    heatmaps_flat = heatmaps.view(B, N, -1)
    max_indices = torch.argmax(heatmaps_flat, dim=-1)
    
    y = max_indices // W
    x = max_indices % W
    
    return torch.stack([x, y], dim=-1).float()

def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    config = checkpoint.get('config', {})
    
    # Model parameters
    model_name = args.model_name or config.get('model_name', 'facebook/dinov2-base')
    image_size = args.image_size or config.get('image_size', 512)
    heatmap_size = args.heatmap_size or config.get('heatmap_size', 512)
    keypoint_names = config.get('keypoint_names', [
        'panda_link0', 'panda_link2', 'panda_link3',
        'panda_link4', 'panda_link6', 'panda_link7', 'panda_hand'
    ])

    # Create model
    model = DINOv3PoseEstimator(
        dino_model_name=model_name,
        heatmap_size=(heatmap_size, heatmap_size)
    ).to(device)

    # Load state dict
    state_dict = checkpoint['model_state_dict']
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    # Create dataset
    dataset = PoseEstimationDataset(
        data_dir=args.data_dir,
        keypoint_names=keypoint_names,
        image_size=(image_size, image_size),
        heatmap_size=(heatmap_size, heatmap_size),
        augment=False,
        multi_robot=args.multi_robot,
        robot_types=args.robot_types
    )
    
    if len(dataset) == 0:
        print(f"Error: No samples found in {args.data_dir}")
        return

    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True
    )

    # PnP를 위한 카메라 정보 로드
    camera_K = None
    cam_settings_path = Path(args.data_dir) / "_camera_settings.json"
    if not cam_settings_path.exists():
        for p in Path(args.data_dir).rglob("_camera_settings.json"):
            cam_settings_path = p
            break
            
    if cam_settings_path.exists():
        print(f"Loading camera intrinsics from {cam_settings_path}")
        camera_K = dream.utilities.load_camera_intrinsics(str(cam_settings_path))
        raw_res = dream.utilities.load_image_resolution(str(cam_settings_path))
        print(f"Camera K:\n{camera_K}")
    else:
        print("Warning: _camera_settings.json not found. ADD score will be skipped.")
        raw_res = (640, 480) # Default

    all_kp_projs_detected_raw = []
    all_kp_projs_gt_raw = []
    all_gt_kp_positions = []
    all_angles_gt = []
    all_angles_pred = []
    
    print(f"Evaluating on {len(dataset)} samples...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            images = batch['image'].to(device)
            
            # Inference
            pred_heatmaps, pred_angles = model(images)
            
            # Extract 2D Keypoints
            pred_kps = get_keypoints_from_heatmaps(pred_heatmaps).cpu().numpy()
            gt_kps = batch['keypoints'].numpy()
            gt_kps_3d = batch['keypoints_3d'].numpy()
            
            scale_x = raw_res[0] / heatmap_size
            scale_y = raw_res[1] / heatmap_size
            
            for i in range(len(pred_kps)):
                p_kps = pred_kps[i].copy()
                g_kps = gt_kps[i].copy()
                
                p_kps[:, 0] *= scale_x
                p_kps[:, 1] *= scale_y
                g_kps[:, 0] *= scale_x
                g_kps[:, 1] *= scale_y
                
                all_kp_projs_detected_raw.append(p_kps)
                all_kp_projs_gt_raw.append(g_kps)
                all_gt_kp_positions.append(gt_kps_3d[i])
            
            all_angles_gt.append(batch['angles'].numpy())
            all_angles_pred.append(pred_angles.cpu().numpy())

    all_kp_projs_detected_raw = np.array(all_kp_projs_detected_raw)
    all_kp_projs_gt_raw = np.array(all_kp_projs_gt_raw)
    all_gt_kp_positions = np.array(all_gt_kp_positions)
    all_angles_gt = np.concatenate(all_angles_gt, axis=0)
    all_angles_pred = np.concatenate(all_angles_pred, axis=0)

    # 1. Keypoint Metrics
    kp_metrics = dream_analysis.keypoint_metrics(
        all_kp_projs_detected_raw.reshape(-1, 2),
        all_kp_projs_gt_raw.reshape(-1, 2),
        raw_res
    )

    # 2. PnP and ADD Metrics
    pnp_results = None
    if camera_K is not None:
        pnp_add = []
        all_n_inframe_projs_gt = []
        
        for kp_projs_est, kp_projs_gt, kp_pos_gt in zip(
            all_kp_projs_detected_raw, all_kp_projs_gt_raw, all_gt_kp_positions
        ):
            n_inframe = 0
            for pt in kp_projs_gt:
                if 0 <= pt[0] < raw_res[0] and 0 <= pt[1] < raw_res[1]:
                    n_inframe += 1
            all_n_inframe_projs_gt.append(n_inframe)

            pnp_retval, translation, quaternion = dream.geometric_vision.solve_pnp(
                kp_pos_gt, kp_projs_est, camera_K
            )

            if pnp_retval:
                add = dream.geometric_vision.add_from_pose(
                    translation, quaternion, kp_pos_gt, camera_K
                )
                pnp_add.append(add)
            else:
                pnp_add.append(-999.0)

        pnp_results = dream_analysis.pnp_metrics(pnp_add, all_n_inframe_projs_gt)

    # 3. Angle Metrics (Rescale from [-1, 1] to [-pi, pi])
    all_angles_pred_rad = all_angles_pred * np.pi
    all_angles_gt_rad = all_angles_gt * np.pi
    
    angle_mae = np.mean(np.abs(all_angles_pred_rad - all_angles_gt_rad))
    angle_mae_deg = np.rad2deg(angle_mae)

    print("\n" + "="*50)
    print("              Evaluation Results")
    print("="*50)
    print(f"Keypoint L2 Error (px):")
    print(f"  - Mean:   {kp_metrics['l2_error_mean_px']:.4f}")
    print(f"  - Median: {kp_metrics['l2_error_median_px']:.4f}")
    print(f"  - AUC:    {kp_metrics['l2_error_auc']:.4f}")
    
    if pnp_results:
        print("-" * 50)
        print(f"PnP / ADD Metric (m):")
        print(f"  - ADD Mean:   {pnp_results['add_mean']:.4f}")
        print(f"  - ADD Median: {pnp_results['add_median']:.4f}")
        print(f"  - ADD AUC:    {pnp_results['add_auc']:.4f}")
        print(f"  - Success:    {pnp_results['num_pnp_found']}/{pnp_results['num_pnp_possible']} frames")

    print("-" * 50)
    print(f"Joint Angle Error:")
    print(f"  - MAE (rad): {angle_mae:.4f}")
    print(f"  - MAE (deg): {angle_mae_deg:.2f}")
    print("="*50)

    results = {
        "keypoint": kp_metrics,
        "pnp_add": pnp_results,
        "angle": { "mae_rad": float(angle_mae), "mae_deg": float(angle_mae_deg) }
    }
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "eval_results.json", "w") as f:
        json.dump(results, f, indent=4, default=lambda x: float(x) if isinstance(x, np.float32) else x)
    print(f"Results saved to: {output_dir / 'eval_results.json'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DINOv3 Pose Estimator with DREAM Metrics")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth checkpoint")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to test data directory")
    parser.add_argument("--output-dir", type=str, default="./eval_outputs", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--multi-robot", action="store_true")
    parser.add_argument("--robot-types", type=str, nargs="+", default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--heatmap-size", type=int, default=None)
    args = parser.parse_args()
    evaluate(args)