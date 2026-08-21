import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from torchvision.ops import roi_align
import numpy as np
import cv2

FEATURE_DIM = 512
NUM_JOINTS = 7  # DO NOT CHANGE: This value is intentionally set to 7.


def solve_pnp_batch(kp_2d, kp_3d_robot, camera_K, reproj_thresh=5.0, depth_range=(0.2, 3.0)):
    """
    Solve PnP to find camera extrinsic and transform robot-frame 3D to camera-frame.
    Validates results using reprojection RMSE and depth sanity checks.

    Args:
        kp_2d: (B, N, 2) - soft-argmax 2D pixel coordinates
        kp_3d_robot: (B, N, 3) - FK robot-frame 3D keypoints
        camera_K: (B, 3, 3) - camera intrinsic matrix
        reproj_thresh: float - max reprojection RMSE (px) to consider valid
        depth_range: tuple - (min_z, max_z) in meters for depth sanity check

    Returns:
        kp_3d_cam: (B, N, 3) - camera-frame 3D keypoints
        valid_mask: (B,) - quality-validated PnP flag per batch
        reproj_errors: (B,) - reprojection RMSE per sample (inf if failed)
    """
    B = kp_2d.shape[0]
    results = []
    valids = []
    reproj_rmses = []

    for b in range(B):
        pts2d = kp_2d[b].detach().cpu().numpy().astype(np.float64)  # (N, 2)
        pts3d = kp_3d_robot[b].detach().cpu().numpy().astype(np.float64)  # (N, 3)
        K = camera_K[b].detach().cpu().numpy().astype(np.float64)  # (3, 3)

        try:
            success, rvec, tvec = cv2.solvePnP(
                pts3d, pts2d, K, None,
                flags=cv2.SOLVEPNP_ITERATIVE,
                useExtrinsicGuess=False,
                rvec=np.zeros(3),
                tvec=np.zeros(3)
            )

            if success:
                R, _ = cv2.Rodrigues(rvec)  # (3, 3)
                kp_cam = (pts3d @ R.T) + tvec.T  # (N, 3)

                # Reprojection error check
                proj_pts, _ = cv2.projectPoints(pts3d, rvec, tvec, K, None)
                proj_pts = proj_pts.reshape(-1, 2)
                reproj_rmse = np.sqrt(np.mean(np.sum((proj_pts - pts2d) ** 2, axis=1)))

                # Depth sanity check: all keypoints should have z in reasonable range
                z_vals = kp_cam[:, 2]
                depth_ok = np.all(z_vals > depth_range[0]) and np.all(z_vals < depth_range[1])

                is_valid = (reproj_rmse < reproj_thresh) and depth_ok

                results.append(torch.from_numpy(kp_cam).float())
                valids.append(is_valid)
                reproj_rmses.append(reproj_rmse)
            else:
                results.append(torch.zeros_like(kp_3d_robot[b]))
                valids.append(False)
                reproj_rmses.append(float('inf'))
        except Exception:
            results.append(torch.zeros_like(kp_3d_robot[b]))
            valids.append(False)
            reproj_rmses.append(float('inf'))

    kp_3d_cam = torch.stack([r.cpu() for r in results], dim=0).to(kp_2d.device)
    valid_mask = torch.tensor(valids, device=kp_2d.device, dtype=torch.bool)
    reproj_errors = torch.tensor(reproj_rmses, device=kp_2d.device, dtype=torch.float32)
    return kp_3d_cam, valid_mask, reproj_errors


def solve_pnp_ransac_batch(kp_2d, kp_3d_robot, camera_K,
                           reproj_thresh=5.0, depth_range=(0.2, 3.0),
                           ransac_reproj=3.0, min_inliers=4):
    """
    RANSAC-based EPnP: robust to 2D keypoint outliers.
    Falls back to iterative PnP with RANSAC inliers if initial RANSAC succeeds.

    Args:
        kp_2d: (B, N, 2) - soft-argmax 2D pixel coordinates
        kp_3d_robot: (B, N, 3) - FK robot-frame 3D keypoints
        camera_K: (B, 3, 3) - camera intrinsic matrix
        reproj_thresh: float - max reprojection RMSE (px) for final validation
        depth_range: tuple - (min_z, max_z) in meters
        ransac_reproj: float - RANSAC inlier threshold in pixels
        min_inliers: int - minimum inliers required (EPnP needs >= 4)

    Returns:
        kp_3d_cam: (B, N, 3) - camera-frame 3D keypoints
        valid_mask: (B,) - quality-validated flag
        reproj_errors: (B,) - reprojection RMSE
        n_inliers: (B,) - number of RANSAC inliers per sample
    """
    B = kp_2d.shape[0]
    results = []
    valids = []
    reproj_rmses = []
    inlier_counts = []

    for b in range(B):
        pts2d = kp_2d[b].detach().cpu().numpy().astype(np.float64)
        pts3d = kp_3d_robot[b].detach().cpu().numpy().astype(np.float64)
        K = camera_K[b].detach().cpu().numpy().astype(np.float64)

        try:
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                pts3d, pts2d, K, None,
                iterationsCount=200,
                reprojectionError=ransac_reproj,
                flags=cv2.SOLVEPNP_EPNP
            )

            n_inl = len(inliers) if inliers is not None else 0

            if success and n_inl >= min_inliers:
                # Refine with iterative PnP using only inlier points
                inl_idx = inliers.flatten()
                pts2d_inl = pts2d[inl_idx]
                pts3d_inl = pts3d[inl_idx]

                ok2, rvec2, tvec2 = cv2.solvePnP(
                    pts3d_inl, pts2d_inl, K, None,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                    useExtrinsicGuess=True,
                    rvec=rvec.copy(), tvec=tvec.copy()
                )
                if ok2:
                    rvec, tvec = rvec2, tvec2

                R, _ = cv2.Rodrigues(rvec)
                kp_cam = (pts3d @ R.T) + tvec.T  # all N keypoints transformed

                # Reprojection error (on ALL points, not just inliers)
                proj_pts, _ = cv2.projectPoints(pts3d, rvec, tvec, K, None)
                proj_pts = proj_pts.reshape(-1, 2)
                reproj_rmse = np.sqrt(np.mean(np.sum((proj_pts - pts2d) ** 2, axis=1)))

                z_vals = kp_cam[:, 2]
                depth_ok = np.all(z_vals > depth_range[0]) and np.all(z_vals < depth_range[1])
                is_valid = (reproj_rmse < reproj_thresh) and depth_ok

                results.append(torch.from_numpy(kp_cam).float())
                valids.append(is_valid)
                reproj_rmses.append(reproj_rmse)
                inlier_counts.append(n_inl)
            else:
                results.append(torch.zeros_like(kp_3d_robot[b]))
                valids.append(False)
                reproj_rmses.append(float('inf'))
                inlier_counts.append(n_inl)
        except Exception:
            results.append(torch.zeros_like(kp_3d_robot[b]))
            valids.append(False)
            reproj_rmses.append(float('inf'))
            inlier_counts.append(0)

    kp_3d_cam = torch.stack([r.cpu() for r in results], dim=0).to(kp_2d.device)
    valid_mask = torch.tensor(valids, device=kp_2d.device, dtype=torch.bool)
    reproj_errors = torch.tensor(reproj_rmses, device=kp_2d.device, dtype=torch.float32)
    n_inliers_t = torch.tensor(inlier_counts, device=kp_2d.device, dtype=torch.int32)
    return kp_3d_cam, valid_mask, reproj_errors, n_inliers_t


def solve_pnp_conf_batch(kp_2d, kp_3d_robot, camera_K, kp_confidence,
                         min_kp=4, reproj_thresh=5.0, depth_range=(0.2, 3.0)):
    """
    Confidence-filtered PnP: drop low-confidence keypoints, then RANSAC+refine.
    Keeps top-K keypoints by heatmap confidence (K >= min_kp).

    Args:
        kp_2d: (B, N, 2)
        kp_3d_robot: (B, N, 3)
        camera_K: (B, 3, 3)
        kp_confidence: (B, N) heatmap peak values
        min_kp: minimum keypoints to use (4 for EPnP)
        reproj_thresh: reprojection RMSE threshold
        depth_range: valid depth range

    Returns:
        kp_3d_cam: (B, N, 3) - all N keypoints transformed
        valid_mask: (B,)
        reproj_errors: (B,)
        n_used: (B,) - number of keypoints used for PnP
    """
    B, N = kp_2d.shape[:2]
    results = []
    valids = []
    reproj_rmses = []
    n_used_list = []

    for b in range(B):
        pts2d_all = kp_2d[b].detach().cpu().numpy().astype(np.float64)
        pts3d_all = kp_3d_robot[b].detach().cpu().numpy().astype(np.float64)
        K = camera_K[b].detach().cpu().numpy().astype(np.float64)
        conf = kp_confidence[b].detach().cpu().numpy()

        # Sort by confidence, keep top keypoints (at least min_kp)
        sorted_idx = np.argsort(-conf)  # descending
        # Try top-5, top-6, top-7 — use the one that works best
        best_result = None
        best_reproj = float('inf')
        best_n_used = 0

        for n_use in [N, N - 1, N - 2]:
            if n_use < min_kp:
                break
            sel_idx = sorted_idx[:n_use]
            pts2d_sel = pts2d_all[sel_idx]
            pts3d_sel = pts3d_all[sel_idx]

            try:
                success, rvec, tvec, inliers = cv2.solvePnPRansac(
                    pts3d_sel, pts2d_sel, K, None,
                    iterationsCount=200, reprojectionError=3.0,
                    flags=cv2.SOLVEPNP_EPNP
                )
                n_inl = len(inliers) if inliers is not None else 0

                if success and n_inl >= min_kp:
                    # Refine with inliers
                    inl_idx = inliers.flatten()
                    ok2, rvec2, tvec2 = cv2.solvePnP(
                        pts3d_sel[inl_idx], pts2d_sel[inl_idx], K, None,
                        flags=cv2.SOLVEPNP_ITERATIVE,
                        useExtrinsicGuess=True,
                        rvec=rvec.copy(), tvec=tvec.copy()
                    )
                    if ok2:
                        rvec, tvec = rvec2, tvec2

                    # Transform ALL keypoints (not just selected)
                    R, _ = cv2.Rodrigues(rvec)
                    kp_cam = (pts3d_all @ R.T) + tvec.T

                    # Reprojection on ALL points
                    proj_pts, _ = cv2.projectPoints(pts3d_all, rvec, tvec, K, None)
                    proj_pts = proj_pts.reshape(-1, 2)
                    reproj_rmse = np.sqrt(np.mean(np.sum((proj_pts - pts2d_all) ** 2, axis=1)))

                    if reproj_rmse < best_reproj:
                        best_reproj = reproj_rmse
                        best_result = kp_cam
                        best_n_used = n_use
            except Exception:
                continue

        if best_result is not None:
            z_vals = best_result[:, 2]
            depth_ok = np.all(z_vals > depth_range[0]) and np.all(z_vals < depth_range[1])
            is_valid = (best_reproj < reproj_thresh) and depth_ok

            results.append(torch.from_numpy(best_result).float())
            valids.append(is_valid)
            reproj_rmses.append(best_reproj)
            n_used_list.append(best_n_used)
        else:
            results.append(torch.zeros_like(kp_3d_robot[b]))
            valids.append(False)
            reproj_rmses.append(float('inf'))
            n_used_list.append(0)

    kp_3d_cam = torch.stack([r.cpu() for r in results], dim=0).to(kp_2d.device)
    valid_mask = torch.tensor(valids, device=kp_2d.device, dtype=torch.bool)
    reproj_errors = torch.tensor(reproj_rmses, device=kp_2d.device, dtype=torch.float32)
    n_used = torch.tensor(n_used_list, device=kp_2d.device, dtype=torch.int32)
    return kp_3d_cam, valid_mask, reproj_errors, n_used


def soft_argmax_2d(heatmaps, temperature=100.0):
    """
    Differentiable soft-argmax to extract (u, v) from heatmaps.
    """
    B, N, H, W = heatmaps.shape
    device = heatmaps.device

    x_coords = torch.arange(W, device=device, dtype=torch.float32)
    y_coords = torch.arange(H, device=device, dtype=torch.float32)

    heatmaps_flat = heatmaps.reshape(B, N, -1)
    if isinstance(temperature, torch.Tensor):
        temperature = temperature.clamp(min=1.0, max=1000.0)

    weights = F.softmax(heatmaps_flat * temperature, dim=-1)
    weights = weights.reshape(B, N, H, W)

    x = (weights.sum(dim=2) * x_coords).sum(dim=-1)
    y = (weights.sum(dim=3) * y_coords).sum(dim=-1)

    return torch.stack([x, y], dim=-1)  # (B, N, 2)


def windowed_soft_argmax_2d(heatmaps, temperature=100.0, win=15):
    """Soft-argmax restricted to a ±win px box around the hard-argmax peak. Robust to distractor
    second-modes that pull the GLOBAL soft-argmax far off (diagnosed: global base 2D mean 101px /
    p90 229px from distractor mass; windowed -> 5.6px). Same sub-pixel precision near the peak."""
    B, N, H, W = heatmaps.shape
    device = heatmaps.device
    flat = heatmaps.reshape(B, N, -1)
    idx = flat.argmax(-1)
    px = (idx % W).float(); py = (idx // W).float()                     # (B,N) hard peak
    xs = torch.arange(W, device=device, dtype=torch.float32)
    ys = torch.arange(H, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    xx = xx.reshape(-1); yy = yy.reshape(-1)                            # (HW,)
    inwin = ((xx[None, None] - px[..., None]).abs() <= win) & \
            ((yy[None, None] - py[..., None]).abs() <= win)             # (B,N,HW)
    if isinstance(temperature, torch.Tensor):
        temperature = temperature.clamp(min=1.0, max=1000.0)
    w = F.softmax(flat.masked_fill(~inwin, -1e4) * temperature, dim=-1)
    x = (w * xx[None, None]).sum(-1); y = (w * yy[None, None]).sum(-1)
    return torch.stack([x, y], dim=-1)

class DINOv3Backbone(nn.Module):
    def __init__(self, model_name, unfreeze_blocks=2, pretrained=True):
        super().__init__()
        self.model_name = model_name
        # pretrained=False -> random-init (from-scratch) baseline: same ViT-B/16 architecture,
        # random weights (AutoModel.from_config bypasses downloaded checkpoints).
        if pretrained:
            # Release build: construct the architecture from the bundled config instead of
            # fetching the gated DINOv3 repo. The backbone weights ship inside our detector
            # checkpoints (223 of 261 tensors), so load_state_dict fully populates the model
            # and no HuggingFace access is required at any point.
            _cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     'assets', 'dinov3_vitb16_config.json')
            if os.path.isfile(_cfg_path):
                self.model = AutoModel.from_config(AutoConfig.from_pretrained(_cfg_path))
            else:
                self.model = AutoModel.from_pretrained(model_name)
        else:
            cfg = AutoConfig.from_pretrained(model_name)
            self.model = AutoModel.from_config(cfg)

        if not pretrained:
            # random-init control. unfreeze_blocks<=0 -> FROZEN random backbone (head-only,
            # parameter-matched control). Otherwise the ENTIRE backbone trains (embeddings + all
            # blocks) — from-scratch needs patch/pos/CLS trainable too.
            full = unfreeze_blocks > 0
            for param in self.model.parameters():
                param.requires_grad = full
            msg = "full backbone trainable (from-scratch)" if full else "FROZEN random backbone (head-only)"
            print(f"  [DINOv3Backbone] random-init ({model_name}): {msg}")
            return

        # Freeze backbone parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze last N blocks for fine-tuning.
        #   DINOv3:     model.encoder.layers  or  model.blocks
        #   SigLIP2:    model.vision_model.encoder.layers
        #   plain ViT:  model.encoder.layer   (singular — google/vit)
        if unfreeze_blocks > 0:
            layers = None
            vt = getattr(self.model, "vision_model", self.model)
            if hasattr(vt, "encoder") and hasattr(vt.encoder, "layers"):
                layers = vt.encoder.layers
            elif hasattr(self.model, "encoder") and hasattr(self.model.encoder, "layers"):
                layers = self.model.encoder.layers
            elif hasattr(self.model, "encoder") and hasattr(self.model.encoder, "layer"):
                layers = self.model.encoder.layer
            elif hasattr(self.model, "layer"):
                layers = self.model.layer          # DINOv3 (transformers 5.x: model.layer ModuleList)
            elif hasattr(self.model, "blocks"):
                layers = self.model.blocks
            if layers is not None:
                for i in range(len(layers) - unfreeze_blocks, len(layers)):
                    for param in layers[i].parameters():
                        param.requires_grad = True
                print(f"  [DINOv3Backbone] unfroze last {unfreeze_blocks}/{len(layers)} blocks")
            else:
                print("  [DINOv3Backbone] WARNING: could not locate encoder layers to unfreeze")

    def forward(self, image_tensor_batch):
        model_type = getattr(self.model.config, "model_type", "")
        if "siglip" in self.model_name:
            # SigLIP/SigLIP2 vision tower returns ALL patch tokens, no CLS/register tokens
            # (verified: siglip2-base-patch16-512 -> (B,1024,768), a perfect 32x32 grid).
            vt = getattr(self.model, "vision_model", self.model)
            outputs = vt(pixel_values=image_tensor_batch, interpolate_pos_encoding=True)
            patch_tokens = outputs.last_hidden_state
        elif model_type == "vit":
            # plain HF ViT (supervised google/vit or random-init): trained at 384, our input is
            # 512 -> MUST interpolate pos-embed. CLS at index 0, no register tokens.
            outputs = self.model(pixel_values=image_tensor_batch, interpolate_pos_encoding=True)
            patch_tokens = outputs.last_hidden_state[:, 1:, :]
        else: # DINOv3 계열 (interpolates pos-embed internally)
            outputs = self.model(pixel_values=image_tensor_batch)
            tokens = outputs.last_hidden_state
            num_reg = int(getattr(self.model.config, "num_register_tokens", 0))
            patch_tokens = tokens[:, 1 + num_reg :, :]
        return patch_tokens

class AdaptiveNorm2d(nn.Module):
    def __init__(self, num_channels, num_groups=32):
        super().__init__()
        self.gn = nn.GroupNorm(num_groups, num_channels)
        self.ln = nn.LayerNorm(num_channels)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        gn_out = self.gn(x)
        ln_out = self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        alpha = torch.sigmoid(self.alpha)
        return alpha * gn_out + (1 - alpha) * ln_out


class TokenFuser(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.projection = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        self.refine_blocks = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            AdaptiveNorm2d(out_channels, num_groups=32),
            nn.GELU(),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            AdaptiveNorm2d(out_channels, num_groups=32)
        )
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)

    def forward(self, x):
        projected = self.projection(x)
        refined = self.refine_blocks(projected)
        residual = self.residual_conv(x)
        return torch.nn.functional.gelu(refined + residual)


class SpatialGlobalModulation(nn.Module):
    def __init__(self, global_dim, feature_dim, dropout_p=0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(global_dim, feature_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(feature_dim * 2, feature_dim * 2)
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x, global_context):
        gamma_beta = self.mlp(global_context)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return x * (1 + gamma) + beta

class ViTKeypointHead(nn.Module):
    def __init__(self, input_dim=768, num_joints=7, heatmap_size=(512, 512)):
        super().__init__()
        self.heatmap_size = heatmap_size
        self.token_fuser = TokenFuser(input_dim, 256)
        self.global_mod1 = SpatialGlobalModulation(global_dim=input_dim, feature_dim=256)
        self.global_mod2 = SpatialGlobalModulation(global_dim=input_dim, feature_dim=128)
        self.global_mod3 = SpatialGlobalModulation(global_dim=input_dim, feature_dim=64)
        self.spatial_dropout = nn.Dropout2d(p=0.1)

        self.decoder_block1 = nn.Sequential(
            nn.Conv2d(256, 128 * 4, kernel_size=3, padding=1, bias=False),  
            nn.PixelShuffle(upscale_factor=2),  
            AdaptiveNorm2d(128, num_groups=32),
            nn.GELU()
        )
        self.decoder_block2 = nn.Sequential(
            nn.Conv2d(128, 64 * 4, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(upscale_factor=2),  
            AdaptiveNorm2d(64, num_groups=16),
            nn.GELU()
        )
        self.decoder_block3 = nn.Sequential(
            nn.Conv2d(64, 32 * 4, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(upscale_factor=2),  
            AdaptiveNorm2d(32, num_groups=8),
            nn.GELU()
        )

        self.heatmap_predictor = nn.Conv2d(32, num_joints, kernel_size=3, padding=1)
        self.final_upsample = nn.Sequential(
            nn.Conv2d(num_joints, num_joints * 4, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(upscale_factor=2),  
            nn.Conv2d(num_joints, num_joints * 4, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(upscale_factor=2)   
        )

    def forward(self, dino_features):
        b, n, d = dino_features.shape
        h = w = int(math.sqrt(n))
        x = dino_features.permute(0, 2, 1).reshape(b, d, h, w)
        global_context = F.adaptive_avg_pool2d(x, 1).flatten(1)
        x = self.token_fuser(x)
        x = self.global_mod1(x, global_context)
        x = self.decoder_block1(x)
        x = self.global_mod2(x, global_context)
        x = self.decoder_block2(x)
        x = self.global_mod3(x, global_context)
        x = self.decoder_block3(x)
        x = self.spatial_dropout(x)
        heatmaps = self.heatmap_predictor(x)
        heatmaps = self.final_upsample(heatmaps)
        if heatmaps.shape[2:] != self.heatmap_size:
            heatmaps = F.interpolate(heatmaps, size=self.heatmap_size, mode='bilinear', align_corners=False)
        return heatmaps

# Forward Kinematics (Fixed for brevity)
def _rotation_matrix_z(theta):
    c, s = torch.cos(theta), torch.sin(theta)
    zero, one = torch.zeros_like(c), torch.ones_like(c)
    return torch.stack([torch.stack([c, -s, zero], dim=-1), torch.stack([s, c, zero], dim=-1), torch.stack([zero, zero, one], dim=-1)], dim=-2)

def _make_transform(xyz, rpy):
    rx, ry, rz = rpy
    cx, sx, cy, sy, cz, sz = math.cos(rx), math.sin(rx), math.cos(ry), math.sin(ry), math.cos(rz), math.sin(rz)
    R = [[cz*cy, cz*sy*sx - sz*cx, cz*sy*cx + sz*sx], [sz*cy, sz*sy*sx + cz*cx, sz*sy*cx - cz*sx], [-sy, cy*sx, cy*cx]]
    return [[R[0][0], R[0][1], R[0][2], xyz[0]], [R[1][0], R[1][1], R[1][2], xyz[1]], [R[2][0], R[2][1], R[2][2], xyz[2]], [0, 0, 0, 1]]

_PANDA_JOINTS = [{'xyz': (0, 0, 0.333), 'rpy': (0, 0, 0)}, {'xyz': (0, 0, 0), 'rpy': (-math.pi/2, 0, 0)}, {'xyz': (0, -0.316, 0), 'rpy': (math.pi/2, 0, 0)}, {'xyz': (0.0825, 0, 0), 'rpy': (math.pi/2, 0, 0)}, {'xyz': (-0.0825, 0.384, 0), 'rpy': (-math.pi/2, 0, 0)}, {'xyz': (0, 0, 0), 'rpy': (math.pi/2, 0, 0)}, {'xyz': (0.088, 0, 0), 'rpy': (math.pi/2, 0, 0)}]
_PANDA_FIXED_J8, _PANDA_FIXED_HAND = {'xyz': (0, 0, 0.107), 'rpy': (0, 0, 0)}, {'xyz': (0, 0, 0), 'rpy': (0, 0, -math.pi/4)}
_PANDA_JOINT_LIMITS = [(-2.8973, 2.8973), (-1.7628, 1.7628), (-2.8973, 2.8973), (-3.0718, -0.0698), (-2.8973, 2.8973), (-0.0175, 3.7525), (-2.8973, 2.8973)]

def panda_forward_kinematics(joint_angles):
    B = joint_angles.shape[0]; device, dtype = joint_angles.device, joint_angles.dtype
    fixed_transforms = [torch.tensor(_make_transform(j['xyz'], j['rpy']), device=device, dtype=dtype) for j in _PANDA_JOINTS]
    T_j8 = torch.tensor(_make_transform(_PANDA_FIXED_J8['xyz'], _PANDA_FIXED_J8['rpy']), device=device, dtype=dtype)
    T_hand = torch.tensor(_make_transform(_PANDA_FIXED_HAND['xyz'], _PANDA_FIXED_HAND['rpy']), device=device, dtype=dtype)
    cumul = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)
    all_transforms = [cumul.clone()]
    for i in range(7):
        theta = joint_angles[:, i]
        R_joint = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1).clone()
        R_joint[:, :3, :3] = _rotation_matrix_z(theta)
        cumul = cumul @ fixed_transforms[i].unsqueeze(0) @ R_joint
        all_transforms.append(cumul.clone())
    cumul_j8 = cumul @ T_j8.unsqueeze(0); all_transforms.append(cumul_j8.clone())
    cumul_hand = cumul_j8 @ T_hand.unsqueeze(0); all_transforms.append(cumul_hand.clone())
    kp_indices = [0, 2, 3, 4, 6, 7, 9]; keypoints = [all_transforms[idx][:, :3, 3] for idx in kp_indices]
    return torch.stack(keypoints, dim=1)


# KUKA iiwa7 (DREAM kuka_allegro). Fixed joint transforms FIT to DREAM synthetic data
# (kuka_synth_{train,test}_dr, sim_state joint angles vs keypoint 3D locations) — reproduces
# link_1..7 origins to 0.003mm RMS across the full joint distribution (train/held-out/test).
# NOTE: J1 pinned to physical base (0,0,0.15); the 3D POSITIONS are exact and are all the
# kinematic solver/PnP needs. Intermediate-frame ORIENTATIONS are data-fit (gauge), not
# URDF-canonical — revisit before mesh render-and-compare. Offset magnitudes measured constant:
# [0.15,0.19,0.21,0.19,0.21,0.1995,0.1012]m; wrist J6/J7 carry the classic iiwa 0.0607m offset.
_IIWA7_JOINTS = [
    {'xyz': (0.000000, 0.000000, 0.150000), 'rpy': (0.000000, 0.000000, 0.000000)},
    {'xyz': (-0.081983, 0.051227, 0.163568), 'rpy': (1.865504, 0.373009, 3.188495)},
    {'xyz': (0.015935, 0.209395, -0.000017), 'rpy': (1.570870, 0.016876, 3.065619)},
    {'xyz': (0.000002, -0.000004, 0.190000), 'rpy': (1.570770, 0.266807, 0.016807)},
    {'xyz': (-0.055366, 0.202570, 0.000011), 'rpy': (-1.570841, 3.477888, 0.266833)},
    {'xyz': (0.020034, 0.057294, 0.190001), 'rpy': (1.570855, -0.539054, -0.336375)},
    {'xyz': (0.041580, 0.069515, 0.060698), 'rpy': (-1.570796, 3.141593, 0.000000)},
]
# iiwa7 joint limits (LBR iiwa 7 R800, rad): ±170,±120,±170,±120,±170,±120,±175 deg
_IIWA7_JOINT_LIMITS = [(-2.9671, 2.9671), (-2.0944, 2.0944), (-2.9671, 2.9671),
                       (-2.0944, 2.0944), (-2.9671, 2.9671), (-2.0944, 2.0944), (-3.0543, 3.0543)]

def iiwa7_forward_kinematics(joint_angles):
    """joint_angles: (B,7) rad (iiwa7_joint_1..7) -> (B,7,3) robot-frame keypoints link_1..7.
    Matches DREAM kuka detector keypoint set (iiwa7_link_1..7)."""
    B = joint_angles.shape[0]; device, dtype = joint_angles.device, joint_angles.dtype
    fixed = [torch.tensor(_make_transform(j['xyz'], j['rpy']), device=device, dtype=dtype) for j in _IIWA7_JOINTS]
    cumul = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)
    keypoints = []
    for i in range(7):
        R_joint = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1).clone()
        R_joint[:, :3, :3] = _rotation_matrix_z(joint_angles[:, i])
        cumul = cumul @ fixed[i].unsqueeze(0) @ R_joint
        keypoints.append(cumul[:, :3, 3])
    return torch.stack(keypoints, dim=1)


# Baxter LEFT arm (DREAM baxter). Fixed joint transforms FIT to DREAM baxter synthetic data
# (40-start scipy LM; single-start hit local minima) — reproduces left_{s0,s1,e0,e1,w0,w1,w2}
# origins to 0.003mm RMS (train/held-out/test). Positions exact (solver/PnP ready); intermediate
# frame orientations are gauge (J1 not pinned). Derivation: Eval/baxter_fk_fit.py.
# Last joint (left_w2) does not move its own keypoint -> unobservable (head predicts s0..w1, w2=0).
_BAXTER_LEFT_JOINTS = [
    {'xyz': (0.000000, 0.000000, 0.278138), 'rpy': (1.201124, 1.097729, -1.273329)},
    {'xyz': (-0.144622, -0.095069, 0.218853), 'rpy': (2.657936, -2.326256, 1.468750)},
    {'xyz': (0.074866, 0.069275, -0.000001), 'rpy': (2.526099, 1.570785, -3.010447)},
    {'xyz': (-0.000000, 0.068998, 0.262420), 'rpy': (1.570789, 2.806062, 4.712390)},
    {'xyz': (-0.034109, -0.097813, -0.000001), 'rpy': (-1.570785, 2.031842, 2.806071)},
    {'xyz': (0.004449, 0.008956, 0.270700), 'rpy': (1.570800, 0.209421, -2.031830)},
    {'xyz': (-0.024111, 0.113441, -0.000000), 'rpy': (1.896772, 2.661118, -1.469447)},
]
# Baxter left-arm joint limits (rad): s0,s1,e0,e1,w0,w1,w2
_BAXTER_LEFT_JOINT_LIMITS = [(-1.7016, 1.7016), (-2.147, 1.047), (-3.0541, 3.0541),
                             (-0.05, 2.618), (-3.059, 3.059), (-1.5707, 2.094), (-3.059, 3.059)]

def baxter_left_forward_kinematics(joint_angles):
    """joint_angles: (B,7) rad (left_s0,s1,e0,e1,w0,w1,w2) -> (B,7,3) robot-frame keypoints.
    Matches DREAM baxter detector keypoint set (left_s0..left_w2)."""
    B = joint_angles.shape[0]; device, dtype = joint_angles.device, joint_angles.dtype
    fixed = [torch.tensor(_make_transform(j['xyz'], j['rpy']), device=device, dtype=dtype) for j in _BAXTER_LEFT_JOINTS]
    cumul = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)
    keypoints = []
    for i in range(7):
        R_joint = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1).clone()
        R_joint[:, :3, :3] = _rotation_matrix_z(joint_angles[:, i])
        cumul = cumul @ fixed[i].unsqueeze(0) @ R_joint
        keypoints.append(cumul[:, :3, 3])
    return torch.stack(keypoints, dim=1)


# =====================================================================================
# Baxter WHOLE-BODY (17-keypoint) FK — matches the DREAM baxter benchmark keypoint set
#   torso_t0, left_{s0,s1,e0,e1,w0,w1,w2}, left_hand, right_{s0..w2}, right_hand
# Fixed joint transforms FIT to DREAM baxter synth data (per-arm 40-start scipy LM + joint
# polish; Eval/baxter_fullbody_fk_fit.py) — reproduces ALL 17 GT `location`s to 0.012mm RMS
# (max 0.05mm) on 400 held-out AND 400 test frames. Left/right arms fit standalone (8 kp incl.
# hand); the right chain is placed into the left gauge by a fixed rigid _BAXTER_FB_BR; torso is
# a fixed point _BAXTER_FB_TORSO. Positions exact (solver/PnP ready); intermediate frame
# orientations are gauge. w2 (both arms) does NOT move any keypoint — the hand offset is a pure
# +z (on the w2 roll axis) — so w2 is UNOBSERVABLE and fixed to 0 (the head predicts 12 joints:
# left {s0,s1,e0,e1,w0,w1} + right {s0,s1,e0,e1,w0,w1}).
_BAXTER_FB_LEFT = [
    {'xyz': (0.256262, -0.057765, 0.355434), 'rpy': (1.159134, 1.266669, -0.852888)},
    {'xyz': (0.035532, -0.059149, 0.270350), 'rpy': (2.060004, -1.570803, 1.622546)},
    {'xyz': (0.000001, 0.101999, 0.000000), 'rpy': (1.655866, 1.570753, -3.056536)},
    {'xyz': (0.000003, 0.069002, 0.262419), 'rpy': (1.570768, 2.594572, 4.712341)},
    {'xyz': (-0.053881, -0.088475, -0.000002), 'rpy': (-1.570753, 2.095780, 2.594614)},
    {'xyz': (0.005012, 0.008656, 0.270700), 'rpy': (1.570901, 1.074047, -2.095684)},
    {'xyz': (-0.101958, 0.055269, -0.000004), 'rpy': (1.570729, 2.130766, -2.067601)},
    {'xyz': (0.000000, 0.000001, 0.113550), 'rpy': (0.000000, 0.000000, 0.000000)},
]
_BAXTER_FB_RIGHT = [
    {'xyz': (-0.154570, 0.020230, 0.279668), 'rpy': (1.540141, -0.110858, -0.110573)},
    {'xyz': (-0.052109, -0.045225, 0.270351), 'rpy': (1.570804, 3.408976, 0.714787)},
    {'xyz': (0.098375, 0.026949, -0.000000), 'rpy': (1.570793, -0.147431, 1.838170)},
    {'xyz': (0.068253, -0.010138, 0.262420), 'rpy': (1.570740, -2.423445, 2.994206)},
    {'xyz': (0.068161, -0.078005, -0.000005), 'rpy': (-1.570591, 1.796742, -2.423234)},
    {'xyz': (0.002240, 0.009743, 0.270700), 'rpy': (1.570821, -0.836442, -1.796748)},
    {'xyz': (0.086085, 0.077715, -0.000002), 'rpy': (1.570789, -4.251388, -3.978034)},
    {'xyz': (0.000003, 0.000000, 0.113550), 'rpy': (1.072065, 0.924819, 0.724986)},
]
_BAXTER_FB_BR = {'xyz': (0.010167, 0.118766, 0.715264), 'rpy': (1.415203, 1.577114, 0.926216)}
_BAXTER_FB_TORSO = (0.376135, 0.061497, 0.599164)
# 12 observable joints: [left s0,s1,e0,e1,w0,w1 | right s0,s1,e0,e1,w0,w1]. (w2 fixed 0.)
_BAXTER_FB_JOINT_LIMITS = [(-1.7016, 1.7016), (-2.147, 1.047), (-3.0541, 3.0541),
                           (-0.05, 2.618), (-3.059, 3.059), (-1.5707, 2.094)] * 2


def _baxter_arm_chain(angles7, fixed_list):
    """angles7: (B,7) rad (s0..w2). fixed_list: 8 transforms. -> (B,8,3) keypoints (s0..w2, hand)."""
    B = angles7.shape[0]; device, dtype = angles7.device, angles7.dtype
    fixed = [torch.tensor(_make_transform(j['xyz'], j['rpy']), device=device, dtype=dtype) for j in fixed_list]
    cumul = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)
    pts = []
    for i in range(7):
        R_joint = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1).clone()
        R_joint[:, :3, :3] = _rotation_matrix_z(angles7[:, i])
        cumul = cumul @ fixed[i].unsqueeze(0) @ R_joint
        pts.append(cumul[:, :3, 3])
    pts.append((cumul @ fixed[7].unsqueeze(0))[:, :3, 3])   # hand (fixed, no joint)
    return torch.stack(pts, dim=1)


def baxter_forward_kinematics(joint_angles):
    """joint_angles: (B,12) rad = [left s0,s1,e0,e1,w0,w1 | right s0,s1,e0,e1,w0,w1] ->
    (B,17,3) robot-frame keypoints in DREAM baxter order
    [torso_t0, left_s0..left_w2, left_hand, right_s0..right_w2, right_hand]. w2 fixed 0."""
    B = joint_angles.shape[0]; device, dtype = joint_angles.device, joint_angles.dtype
    z = torch.zeros(B, 1, device=device, dtype=dtype)
    left7 = torch.cat([joint_angles[:, :6], z], dim=1)      # w2 = 0
    right7 = torch.cat([joint_angles[:, 6:12], z], dim=1)
    left8 = _baxter_arm_chain(left7, _BAXTER_FB_LEFT)        # (B,8,3) left gauge (common frame)
    right8 = _baxter_arm_chain(right7, _BAXTER_FB_RIGHT)     # (B,8,3) right standalone gauge
    BR = torch.tensor(_make_transform(_BAXTER_FB_BR['xyz'], _BAXTER_FB_BR['rpy']), device=device, dtype=dtype)
    right8 = torch.einsum('ij,bnj->bni', BR[:3, :3], right8) + BR[:3, 3]   # place into common frame
    torso = torch.tensor(_BAXTER_FB_TORSO, device=device, dtype=dtype).view(1, 1, 3).expand(B, 1, 3)
    return torch.cat([torso, left8, right8], dim=1)          # (B,17,3)


class DirectJointAngleHead(nn.Module):
    """
    Feature + UV with confidence weighting.
    Use UV but weight by heatmap confidence.
    """

    def __init__(self, input_dim=768, num_joints=7, num_angles=7):
        super().__init__()
        self.num_angles = num_angles
        self.num_joints = num_joints
        self.n_iter = 4

        # UV encoder
        self.uv_encoder = nn.Sequential(
            nn.Linear(num_joints * 2, 256),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(256, 256)
        )

        # Confidence encoder (weight UV by heatmap quality)
        self.conf_encoder = nn.Sequential(
            nn.Linear(num_joints, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # 0~1 weight
        )

        # Iterative residual
        self.fc1 = nn.Linear(input_dim + 256 + num_angles, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.decpose = nn.Linear(1024, num_angles)
        self.drop1 = nn.Dropout(p=0.3)
        self.drop2 = nn.Dropout(p=0.3)
        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)

    def forward(self, dino_features, predicted_heatmaps, camera_K=None):
        B = dino_features.shape[0]
        
        xf = dino_features.mean(dim=1)  # (B, 768)
        
        # UV + confidence
        uv = soft_argmax_2d(predicted_heatmaps, temperature=100.0)
        
        # Heatmap confidence (max value per joint)
        conf = predicted_heatmaps.flatten(2).max(dim=2)[0]  # (B, N)
        uv_weight = self.conf_encoder(conf)  # (B, 1)
        
        uv_flat = uv.reshape(B, -1)
        uv_feat = self.uv_encoder(uv_flat) * uv_weight  # Weight by confidence
        
        xf_combined = torch.cat([xf, uv_feat], dim=1)

        pred_pose = torch.zeros(B, self.num_angles, device=xf.device)
        for _ in range(self.n_iter):
            xc = torch.cat([xf_combined, pred_pose], dim=1)
            xc = self.drop1(F.relu(self.fc1(xc)))
            xc = self.drop2(F.relu(self.fc2(xc)))
            pred_pose = self.decpose(xc) + pred_pose

        return pred_pose, uv, None


class DINOv3PoseEstimatorV4(nn.Module):
    def __init__(self, dino_model_name, heatmap_size, unfreeze_blocks=2, fix_joint7_zero=True):
        super().__init__()
        self.dino_model_name = dino_model_name
        self.heatmap_size = heatmap_size
        self.fix_joint7_zero = fix_joint7_zero
        self.backbone = DINOv3Backbone(dino_model_name, unfreeze_blocks=unfreeze_blocks)
        feat_dim = self.backbone.model.config.hidden_size if "conv" not in dino_model_name else self.backbone.model.config.hidden_sizes[-1]
        self.keypoint_head = ViTKeypointHead(input_dim=feat_dim, heatmap_size=heatmap_size)

        # Joint angle head (predicts 6 angles usually)
        num_angles_to_predict = 6 if fix_joint7_zero else 7
        self.joint_angle_head = DirectJointAngleHead(input_dim=feat_dim, num_joints=NUM_JOINTS, num_angles=num_angles_to_predict)

    def forward(self, image_tensor_batch, camera_K=None, **kwargs):
        dino_features = self.backbone(image_tensor_batch)
        predicted_heatmaps = self.keypoint_head(dino_features)

        # Joint angle head outputs normalized angles
        joint_angles_norm, _, _ = self.joint_angle_head(dino_features, predicted_heatmaps, camera_K=camera_K)
        
        if self.fix_joint7_zero:
            zeros = torch.zeros(joint_angles_norm.shape[0], 1, device=joint_angles_norm.device)
            joint_angles_norm = torch.cat([joint_angles_norm, zeros], dim=1)

        result = {
            'heatmaps_2d': predicted_heatmaps,
            'joint_angles': joint_angles_norm,  # normalized space (B, 7)
        }
        return result

