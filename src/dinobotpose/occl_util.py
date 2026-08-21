"""
Deterministic synthetic-occlusion painter, matching RoboPEPP's occlusion-robustness protocol
(arXiv:2411.17662 Sec. "occlusion analysis"): overlay BLACK rectangular or circular masks at random
positions ON the robot (RoI = GT-keypoint bbox + margin), sized so the covered fraction of the RoI
area hits the requested ratio (0.1/0.2/0.3/0.4). RoboPEPP runs this on Panda Photo (synthetic).

Determinism: the RNG is seeded from (frame-id, ratio), so every pipeline stage (pose eval, SAM/RC
refinement) paints EXACTLY the same occluders for a given frame — required because the render-compare
stage re-loads images independently of the pose stage.
"""
import zlib
import numpy as np
import torch

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
_BLACK = (0.0 - IMAGENET_MEAN) / IMAGENET_STD          # black in normalized space, (3,1,1)


def _roi(kp_xy, valid, S, margin=0.10):
    """GT-keypoint bbox + margin, clipped to the image. kp_xy: (N,2) in image space."""
    p = kp_xy[valid > 0]
    if len(p) < 2:
        return 0, 0, S, S
    x0, y0 = p[:, 0].min(), p[:, 1].min()
    x1, y1 = p[:, 0].max(), p[:, 1].max()
    mx = margin * max(x1 - x0, y1 - y0)
    return (int(max(0, x0 - mx)), int(max(0, y0 - mx)),
            int(min(S, x1 + mx)), int(min(S, y1 + mx)))


def paste_occluders_(img, kp_xy, valid, ratio, fid, S=None):
    """In-place: paint black rect/circle occluders on a NORMALIZED image tensor (3,H,W) until
    `ratio` of the RoI area is covered. kp_xy in image space (same scale as img). Returns the
    binary occlusion mask (H,W) for diagnostics."""
    if ratio <= 0:
        return None
    S = S or img.shape[-1]
    rng = np.random.RandomState(zlib.crc32(f"{fid}|{ratio:.2f}".encode()) & 0x7FFFFFFF)
    x0, y0, x1, y1 = _roi(np.asarray(kp_xy), np.asarray(valid), S)
    roi_area = max(1, (x1 - x0) * (y1 - y0))
    occ = np.zeros((img.shape[-2], img.shape[-1]), dtype=bool)
    black = _BLACK.to(img.device, img.dtype)
    guard = 0
    while occ[y0:y1, x0:x1].sum() < ratio * roi_area and guard < 60:
        guard += 1
        # random center INSIDE the RoI (ensures the mask covers part of the robot region)
        cx = rng.randint(x0, max(x0 + 1, x1)); cy = rng.randint(y0, max(y0 + 1, y1))
        # size ~ enough to plausibly close the remaining deficit in a few shapes
        rem = ratio * roi_area - occ[y0:y1, x0:x1].sum()
        r = int(np.clip(np.sqrt(max(rem, 0.02 * roi_area)) * rng.uniform(0.4, 0.9), 6, S // 3))
        if rng.rand() < 0.5:                                     # rectangle
            hw = int(r * rng.uniform(0.6, 1.6)); hh = max(4, int(r * r / max(hw, 1)))
            ax0, ay0 = max(0, cx - hw // 2), max(0, cy - hh // 2)
            ax1, ay1 = min(S, cx + hw // 2), min(S, cy + hh // 2)
            occ[ay0:ay1, ax0:ax1] = True
        else:                                                    # circle
            yy, xx = np.ogrid[:img.shape[-2], :img.shape[-1]]
            occ |= (xx - cx) ** 2 + (yy - cy) ** 2 <= (r // 2 + 2) ** 2
    m = torch.from_numpy(occ).to(img.device)
    img[:, m] = black.expand(3, S, S)[:, m]
    return occ


def paste_occluders_batch_(imgs, kps_xy, valids, ratio, fids):
    """In-place batched variant. imgs (B,3,H,W) normalized; kps_xy (B,N,2) image-space."""
    if ratio <= 0:
        return
    for b in range(imgs.shape[0]):
        paste_occluders_(imgs[b], kps_xy[b], valids[b], ratio, str(fids[b]))


def paste_random_occluders_(imgs, kps_xy, valids, max_ratio, prob=0.5, rng=None):
    """TRAIN-time augmentation: with probability `prob` per sample, occlude a Uniform(0.05,
    max_ratio) fraction of the RoI (non-deterministic). Teaches heads to handle the degraded
    heatmap/conf/keypoint inputs an occluded robot produces (backbone/detector stay frozen)."""
    if max_ratio <= 0:
        return
    rng = rng or np.random
    for b in range(imgs.shape[0]):
        if rng.rand() < prob:
            r = float(rng.uniform(0.05, max_ratio))
            paste_occluders_(imgs[b], kps_xy[b], valids[b], r, f"aug{rng.randint(1 << 30)}")
