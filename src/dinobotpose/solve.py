"""
Stage 2 — Kinematic angle + pose solver (the "decisive experiment").

Idea (see plan resilient-sparking-castle.md):
    Stop regressing joint angles. Detect 2D keypoints well (any checkpoint with a
    ViTKeypointHead -> heatmaps_2d), then SOLVE the joint angles (theta) and the
    camera pose (R, t) geometrically by minimizing a confidence-weighted
    reprojection error of the analytic forward kinematics:

        min_{theta, R, t}  sum_j  conf_j * rho( project(FK(theta)_j; R, t, K) - kp2d_j )

    Joint angles are clamped to their mechanical limits via a sigmoid
    re-parametrization; the camera rotation uses the 6D continuous representation
    (Zhou et al. 2019). Initialization comes from a single OpenCV PnP solve on
    FK(theta_mean), with optional multi-start to escape local minima.

This is the cheap, keypoint-based cousin of RoboPose's render-and-compare.

It is intentionally model-agnostic: it only requires the model forward to return
`heatmaps_2d`. Works with model.py / model_v3.py / model_v4.py checkpoints and the
2D-pretrained best_heatmap.pth.

Usage (quick decisive run on a few hundred frames):
    python Eval/solve_pose_kinematic.py \
        -p TRAIN/outputs_heatmap/best_heatmap.pth \
        -d Dataset/Converted_dataset/DREAM_to_DREAM_syn/panda_synth_test_dr \
        --model-module model_v4 --model-class DINOv3PoseEstimatorV4 \
        --max-frames 300 -o Eval/results_kinematic
"""

import argparse
import importlib
import json
import math
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

TRAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../TRAIN'))
sys.path.append(TRAIN_DIR)
sys.path.append(os.path.dirname(__file__))

from model_v4 import panda_forward_kinematics, soft_argmax_2d, _PANDA_JOINT_LIMITS  # noqa: E402

# Hardcoded mean used only as the optimization start point for theta.
PANDA_JOINT_MEAN = torch.tensor([-5.22e-02, 2.68e-01, 6.04e-03, -2.01e+00, 1.49e-02, 1.99e+00, 0.0])

# Keypoint names exactly as expected by EvalDataset / FK keypoint order.
KEYPOINT_NAMES = ['panda_link0', 'panda_link2', 'panda_link3',
                  'panda_link4', 'panda_link6', 'panda_link7', 'panda_hand']

# OPT-IN determinism (default None = deployed unseeded behavior, bit-identical). When set, every
# pnp_init call re-seeds cv2's RANSAC RNG to this value so the solve is independent of RNG history
# — needed to prove z-bound do-no-harm bitwise: the z-bound re-solve makes extra PnP calls that
# would otherwise drift the global RNG and perturb later batches' in-bound frames.
_PNP_SEED = None


def set_pnp_seed(s):
    global _PNP_SEED
    _PNP_SEED = s


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def rot6d_to_matrix(d6):
    """(B, 6) 6D rotation representation -> (B, 3, 3). Zhou et al. 2019."""
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)  # columns are basis vectors


def matrix_to_rot6d(R):
    """(B, 3, 3) -> (B, 6): first two columns flattened."""
    return torch.cat([R[..., 0], R[..., 1]], dim=-1)


def ik_from_3d(pred_kp_3d, theta_mean, lo, hi, iters=150, lr=5e-2):
    """
    Recover joint angles from predicted robot-frame 3D keypoints by fitting FK.
    (Mirrors eval_3d_v3.optimize_ik_batch but with joint-limit clamping.)
    pred_kp_3d: (B,7,3) tensor. Returns theta (B,7) with joint7=0.
    """
    B = pred_kp_3d.shape[0]
    device = pred_kp_3d.device
    theta0 = theta_mean.unsqueeze(0).expand(B, 7).clone()
    theta0[:, 6] = 0.0
    theta0 = torch.max(torch.min(theta0, hi), lo)
    p = theta_to_p(theta0, lo, hi).clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([p], lr=lr)
    for _ in range(iters):
        opt.zero_grad()
        theta = p_to_theta(p, lo, hi)
        theta = torch.cat([theta[:, :6], torch.zeros(B, 1, device=device)], dim=1)
        fk = panda_forward_kinematics(theta)
        loss = F.mse_loss(fk, pred_kp_3d)
        loss.backward()
        opt.step()
    with torch.no_grad():
        theta = p_to_theta(p, lo, hi)
        theta = torch.cat([theta[:, :6], torch.zeros(B, 1, device=device)], dim=1)
    return theta.detach()


def masked_median(vals, mask):
    """Per-row median of `vals` (B,N) over entries where mask (B,N) is truthy.
    Rows with no active entry fall back to the row median over all N. Returns (B,)."""
    B, N = vals.shape
    big = vals.max().detach() + 1.0 if vals.numel() else torch.tensor(1.0, device=vals.device)
    m = mask.bool()
    filled = torch.where(m, vals, torch.full_like(vals, float(big)))
    cnt = m.sum(dim=1).clamp(min=1)                       # (B,)
    srt, _ = torch.sort(filled, dim=1)                    # active entries sort to the front
    # median index = floor((cnt-1)/2) among the active (front) entries
    mid = ((cnt - 1) // 2).long()
    med = srt.gather(1, mid.unsqueeze(1)).squeeze(1)      # (B,)
    # rows with zero active entries -> plain median over all N
    none_active = mask.bool().sum(dim=1) == 0
    if none_active.any():
        med = torch.where(none_active, vals.median(dim=1).values, med)
    return med


def project_points(pts_robot, R, t, K):
    """
    pts_robot: (B, N, 3) robot-frame FK keypoints
    R: (B, 3, 3) robot->camera, t: (B, 3), K: (B, 3, 3)
    returns (B, N, 2) pixel coords
    """
    pts_cam = torch.bmm(pts_robot, R.transpose(1, 2)) + t.unsqueeze(1)  # (B,N,3)
    pts_img = torch.bmm(pts_cam, K.transpose(1, 2))                      # (B,N,3)
    z = pts_img[..., 2:3].clamp(min=1e-6)
    return pts_img[..., :2] / z, pts_cam


# ---------------------------------------------------------------------------
# Joint-limit re-parametrization: theta = lo + (hi-lo) * sigmoid(p)
# ---------------------------------------------------------------------------
def make_limits(device, dtype):
    lo = torch.tensor([l for l, _ in _PANDA_JOINT_LIMITS], device=device, dtype=dtype)
    hi = torch.tensor([h for _, h in _PANDA_JOINT_LIMITS], device=device, dtype=dtype)
    return lo, hi  # (7,)


def theta_to_p(theta, lo, hi):
    """Inverse sigmoid map, with safe clamping just inside the limits."""
    frac = ((theta - lo) / (hi - lo)).clamp(1e-4, 1 - 1e-4)
    return torch.log(frac / (1 - frac))


def p_to_theta(p, lo, hi):
    return lo + (hi - lo) * torch.sigmoid(p)


# ---------------------------------------------------------------------------
# PnP initialization (single solve on FK(theta_init))
# ---------------------------------------------------------------------------
def border_mask(kp_2d, margin, img_size):
    """Keypoints whose DECODED 2D position sits within `margin` px of (or outside) the image
    border. A keypoint that has left the frame produces a heatmap peak PINNED to the border that
    is confidently wrong — a sharp, high-confidence, geometrically bogus detection. `conf_gate`
    filters on the confidence axis and so structurally cannot see these; the frame boundary is an
    independent axis. kp_2d: (B,N,2) tensor or np, margin px, img_size px. Returns (B,N) bool."""
    if isinstance(kp_2d, np.ndarray):
        x, y = kp_2d[..., 0], kp_2d[..., 1]
        return (x < margin) | (x > img_size - margin) | (y < margin) | (y > img_size - margin)
    x, y = kp_2d[..., 0], kp_2d[..., 1]
    return (x < margin) | (x > img_size - margin) | (y < margin) | (y > img_size - margin)


def pnp_init(kp_2d, kp_3d_robot, K, conf=None, conf_gate=0.0, min_kp=6, pnp_rel=0.0, pnp_drop=3,
             deprio=None):
    """
    Confidence-ranked PnP for a robust (R, t) initialization.
    kp_2d: (B,N,2) np, kp_3d_robot: (B,N,3) np, K: (B,3,3) np, conf: (B,N) np or None.
    deprio: (B,N) bool np or None — keypoints pushed to the BACK of the confidence ranking
            (border-pinned off-frame hallucinations). They are therefore excluded from the
            top-k minimal init set, but remain available to the degeneracy fallback so a frame
            can never be starved below 4 points.

    KEY EMPIRICAL FINDING (Eval/solve_sweep.py, strided 600 frames/cam): the cleaner the PnP
    INIT, the better the final pose — because the gradient refine uses ALL points to polish, so
    the init only has to pick the right basin, and a few HIGH-confidence points give the right
    basin while low-conf points (far-camera distance noise) reproject within tolerance but bias
    EPnP's depth. Initializing from the top-(N-pnp_drop) most-confident keypoints (pnp_drop=3 ->
    top-4) beats all-7 init on EVERY split (mean ADD-AUC 0.663 vs 0.600). Minimal sets can be
    degenerate, so we FALL BACK to progressively more points until PnP is valid.
    returns R (B,3,3), t (B,3) numpy, valid (B,) bool
    """
    B, N = kp_2d.shape[:2]
    if _PNP_SEED is not None:
        cv2.setRNGSeed(int(_PNP_SEED))       # opt-in determinism (do-no-harm proof); default no-op
    Rs = np.tile(np.eye(3), (B, 1, 1)).astype(np.float64)
    ts = np.tile(np.array([0.0, 0.0, 1.2]), (B, 1)).astype(np.float64)
    valid = np.zeros(B, dtype=bool)
    k0 = max(4, N - pnp_drop)  # smallest (cleanest) init set; grow on degeneracy
    for b in range(B):
        if conf is not None:
            rank_key = conf[b].astype(np.float64).copy()
            if deprio is not None:
                # push border-pinned keypoints below every real detection, order-preserving
                rank_key[deprio[b]] -= 1e3
            order = np.argsort(-rank_key)
        else:
            order = np.arange(N)
        for k in range(k0, N + 1):                     # top-k, fall back to more pts if invalid
            sel = order[:k]
            p3 = kp_3d_robot[b][sel].astype(np.float64)
            p2 = kp_2d[b][sel].astype(np.float64)
            try:
                ok, rvec, tvec, inl = cv2.solvePnPRansac(
                    p3, p2, K[b].astype(np.float64), None,
                    iterationsCount=200, reprojectionError=5.0, flags=cv2.SOLVEPNP_EPNP)
                if ok and inl is not None and len(inl) >= 4:
                    idx = inl.flatten()
                    ok2, rvec, tvec = cv2.solvePnP(
                        p3[idx], p2[idx], K[b].astype(np.float64), None,
                        useExtrinsicGuess=True, rvec=rvec, tvec=tvec,
                        flags=cv2.SOLVEPNP_ITERATIVE)
                    R, _ = cv2.Rodrigues(rvec)
                    if np.all(np.isfinite(R)) and np.all(np.isfinite(tvec)) and tvec.flatten()[2] > 0:
                        Rs[b], ts[b], valid[b] = R, tvec.flatten(), True
                        break
            except cv2.error:
                pass
    return Rs, ts, valid


# ---------------------------------------------------------------------------
# Core optimizer (batched, all frames independent)
# ---------------------------------------------------------------------------
def heatmap_cov_inv(heatmaps, kp2d, win=15, sigma_min=1.0, sigma_max=64.0):
    """Per-keypoint ANISOTROPIC 2x2 inverse covariance from the heatmap's local second moments
    around the soft-argmax peak. Occluded/ambiguous keypoints produce diffuse or multimodal
    heatmaps -> large covariance -> smoothly down-weighted (Mahalanobis) instead of hard-gated.
    heatmaps: (B,N,H,W) (heatmap res == image res in this repo), kp2d: (B,N,2) px. Returns (B,N,2,2)."""
    B, N, H, W = heatmaps.shape
    dev = heatmaps.device
    r = win // 2
    cx = kp2d[..., 0].round().long().clamp(r, W - 1 - r)          # (B,N)
    cy = kp2d[..., 1].round().long().clamp(r, H - 1 - r)
    off = torch.arange(-r, r + 1, device=dev)
    gy = (cy.unsqueeze(-1) + off).unsqueeze(-1)                    # (B,N,win,1)
    gx = (cx.unsqueeze(-1) + off).unsqueeze(-2)                    # (B,N,1,win)
    patch = heatmaps.clamp(min=0.0)[
        torch.arange(B, device=dev)[:, None, None, None],
        torch.arange(N, device=dev)[None, :, None, None],
        gy.expand(B, N, win, win), gx.expand(B, N, win, win)]      # (B,N,win,win)
    p = patch / patch.sum(dim=(-1, -2), keepdim=True).clamp(min=1e-8)
    xs = off.view(1, 1, 1, win).expand(B, N, win, win).float()     # local coords about the peak
    ys = off.view(1, 1, win, 1).expand(B, N, win, win).float()
    mx = (p * xs).sum(dim=(-1, -2)); my = (p * ys).sum(dim=(-1, -2))
    vxx = (p * (xs - mx[..., None, None]) ** 2).sum(dim=(-1, -2))
    vyy = (p * (ys - my[..., None, None]) ** 2).sum(dim=(-1, -2))
    vxy = (p * (xs - mx[..., None, None]) * (ys - my[..., None, None])).sum(dim=(-1, -2))
    lo, hi = sigma_min ** 2, sigma_max ** 2
    vxx = vxx.clamp(lo, hi); vyy = vyy.clamp(lo, hi); vxy = vxy.clamp(-hi, hi)
    det = (vxx * vyy - vxy ** 2).clamp(min=lo * lo * 0.25)
    inv = torch.stack([torch.stack([vyy / det, -vxy / det], -1),
                       torch.stack([-vxy / det, vxx / det], -1)], -2)  # (B,N,2,2)
    return inv


def solve_batch(kp_2d, conf, K, fix_joint7=True, iters=250, lr=5e-2,
                img_size=512, device='cuda', prior_w=2e-3, theta_init=None,
                conf_gate=0.0, anchor_init_w=0.0, min_kp=6, pnp_rel=0.0, pnp_drop=3,
                R_init=None, t_init=None, gt_tz=None, depth_w=0.0, return_pose=False,
                cov_inv=None, prior_adaptive=0.0, freeze_theta=False,
                border_margin=0.0, resolve_reproj_thr=0.0, resolve_drops=(0, 1, 2),
                _resolve_stats=None, robust_scale=0.0, shed_k=0.0, drop_mask=None,
                _shed_stats=None, z_bound=None, z_target=0.0, z_anchor_w=0.5,
                _zbound_stats=None):
    """
    kp_2d: (B,N,2) tensor, conf: (B,N) tensor, K: (B,3,3) tensor.
    theta_init: optional (B,7) tensor — learned-prediction init (refinement mode).
                If None, init from the joint mean (cold start).

    Occlusion handling (the off-frame/occluded-keypoint failure tail):
      conf_gate>0     : HARD-reject keypoints with conf<gate (weight exactly 0) so PnP and
                        the reprojection residual fully ignore hallucinated occluded points.
      anchor_init_w>0 : anchor angles to theta_init (the LEARNED prediction), not the dataset
                        mean. Joints whose keypoints are gated out keep no data constraint, so
                        this prior makes them fall back to the learned estimate instead of
                        drifting -> "when occluded, predict from the kinematic/learned prior".
      cov_inv (B,N,2,2): OPTIONAL anisotropic inverse covariance (from heatmap_cov_inv) —
                        the reprojection residual becomes a Mahalanobis (whitened) distance, so
                        diffuse/ambiguous heatmaps are down-weighted CONTINUOUSLY per-direction
                        (upgrade of the scalar-conf weighting; conf_gate still composes).
      border_margin>0 : BORDER gate (off by default). Keypoints decoded within this many px of the
                        image border are treated as off-frame hallucinations: pushed out of the
                        minimal-set PnP init and zero-weighted in the refine. Orthogonal to
                        conf_gate — a border-pinned peak is SHARP (high conf) but wrong, so the
                        confidence axis cannot reject it.
      resolve_reproj_thr>0 : MULTI-START re-solve (off by default). Frames whose solved
                        reprojection exceeds this many px are re-solved from several alternative
                        PnP inits (grown/shrunk minimal sets, `resolve_drops`). A candidate is
                        adopted ONLY if it lowers the solver's own reprojection residual — the
                        same structural do-no-harm guard used for the refine below, so a re-solve
                        can never make a frame worse by the solver's own measure.
      robust_scale>0  : ADAPTIVE robust loss (off by default -> fixed 8px Geman-McClure knee).
                        Replaces the fixed IRLS knee with `robust_scale x per-frame median
                        residual` (MAD-style, robot-agnostic). Restores the outlier gap on diffuse
                        heatmaps whose residuals sit far above 8px.
      shed_k>0        : SOFT FLOOR (off by default). After an initial solve, drop keypoints whose
                        reprojection residual exceeds `shed_k x per-frame median` (letting the
                        retained count fall BELOW min_kp, never below 4) and re-solve. Adopted
                        only if it lowers reprojection ON THE RETAINED KEYPOINTS (GT-free
                        do-no-harm). Targets the forced-in outlier that drives the divergent tail.
      drop_mask (B,N) : internal — force-zero these keypoints after the floor (shed re-solve).
      z_bound (lo,hi) : PHYSICAL-PLAUSIBILITY DEPTH BOUND (off by default). Targets the z-collapse
                        failure mode: monocular scale ambiguity lets the solver settle in a
                        near-zero-depth basin that FITS the 2D well (LOW reprojection) but is
                        physically impossible. Neither conf_gate, border_margin, resolve_reproj_thr
                        nor shed_k can catch it — the collapsed solution has the LOWEST reprojection,
                        so every reproj-keyed guard structurally misses it. After the solve, frames
                        whose median camera-frame skeleton depth lands outside [lo,hi] m are RE-SOLVED
                        from theta_init with the whole skeleton re-initialized to a plausible depth
                        (`z_target`) and held there by a depth anchor (`z_anchor_w`) while R and theta
                        re-fit the 2D. Adoption is keyed on PHYSICAL VALIDITY (recovered median depth
                        back in-bounds), NOT reprojection — a reproj gate would reject every recovery.
                        In-bound frames are never touched (post-hoc conditional) -> do-no-harm.
      z_target (m)    : plausible re-init depth for the z_bound re-solve. >0 = fixed scene depth
                        (e.g. the split's median SOLVED depth over healthy frames — a GT-free scene
                        prior). <=0 = per-frame clamp of the collapsed depth into [lo,hi] (weaker floor).
      z_anchor_w      : depth-anchor weight holding the re-solve at z_target (default 0.5).
    Returns theta (B,7), kp_cam (B,N,3), reproj_px (B,).
    """
    B, N = kp_2d.shape[:2]
    dtype = torch.float32
    lo, hi = make_limits(device, dtype)

    theta_mean = PANDA_JOINT_MEAN.to(device, dtype)

    # --- init theta (learned prior if given, else mean), init R,t via PnP ---
    if theta_init is not None:
        theta0 = theta_init.to(device, dtype).clone()
    else:
        theta0 = theta_mean.unsqueeze(0).expand(B, 7).clone()
    if fix_joint7:
        theta0[:, 6] = 0.0
    theta0 = torch.max(torch.min(theta0, hi), lo)  # clamp into limits
    fk0 = panda_forward_kinematics(theta0)  # (B,N,3) robot frame
    # BORDER gate: computed on the DECODED 2d, before the PnP init (the init is what the
    # off-frame trigger breaks; the existing divergence guard cannot recover a bad init).
    bmask = border_mask(kp_2d, border_margin, img_size) if border_margin > 0 else None
    R0, t0, _ = pnp_init(kp_2d.cpu().numpy(), fk0.detach().cpu().numpy(),
                         K.cpu().numpy(), conf.cpu().numpy(), conf_gate=conf_gate,
                         min_kp=min_kp, pnp_rel=pnp_rel, pnp_drop=pnp_drop,
                         deprio=bmask.cpu().numpy() if bmask is not None else None)
    R0 = torch.from_numpy(R0).to(device, dtype)
    t0 = torch.from_numpy(t0).to(device, dtype)
    # Optional learned/oracle pose init to escape the far-camera rotation-basin ambiguity
    # (the reprojection objective is degenerate; a prior on R is what pins the basin).
    if R_init is not None:
        R0 = R_init.to(device, dtype).clone()
    if t_init is not None:
        t0 = t_init.to(device, dtype).clone()

    # --- init geometry (for the per-frame divergence guard) ---
    fk_init = panda_forward_kinematics(theta0)
    R0m = rot6d_to_matrix(matrix_to_rot6d(R0))  # orthonormalized to match refine path
    uv_init, kpcam_init = project_points(fk_init, R0m, t0, K)
    reproj_init = (uv_init - kp_2d).norm(dim=-1).mean(dim=1)  # (B,)

    # --- learnable params ---
    p = theta_to_p(theta0, lo, hi).clone().detach().requires_grad_(not freeze_theta)
    d6 = matrix_to_rot6d(R0).clone().detach().requires_grad_(True)
    t = t0.clone().detach().requires_grad_(True)

    # known-joint mode: hold theta at theta0 (=GT), optimize only camera pose (R,t)
    opt = torch.optim.Adam(([d6, t] if freeze_theta else [p, d6, t]), lr=lr)
    # base confidence weights, normalized per-frame
    base_w = conf.clamp(min=1e-3)
    if conf_gate > 0.0:
        keep = conf >= conf_gate                                   # (B,N) HARD-reject occluded kp
        if min_kp > 0:
            # floor: always keep the top-min_kp by conf so far cameras (low conf everywhere)
            # are not starved into a degenerate PnP. Only drops the genuine off-frame tail.
            rank = torch.argsort(conf, dim=1, descending=True)
            topk = torch.zeros_like(keep)
            topk.scatter_(1, rank[:, :min_kp], True)
            keep = keep | topk
        base_w = base_w * keep.to(base_w.dtype)
    if bmask is not None:
        # zero-weight border-pinned keypoints in the refine too, with the same top-min_kp floor
        # used by conf_gate so a frame is never starved into a degenerate fit.
        bkeep = ~bmask
        if min_kp > 0:
            rank = torch.argsort(conf, dim=1, descending=True)
            topk = torch.zeros_like(bkeep)
            topk.scatter_(1, rank[:, :min_kp], True)
            bkeep = bkeep | topk
        base_w = base_w * bkeep.to(base_w.dtype)
    if drop_mask is not None:
        # SHED override: force-zero these keypoints AFTER the min_kp floor, so the count of
        # retained keypoints is allowed to fall BELOW the floor. Used by the shed_k re-solve
        # (the floor is load-bearing only for INIT; here we already have a good init to inherit).
        base_w = base_w * (~drop_mask.bool()).to(base_w.dtype)
    base_w = base_w / base_w.sum(dim=1, keepdim=True).clamp(min=1e-6)  # (B,N)
    theta_init_p = theta0.detach().clone()  # anchor target = learned init (post-clamp)
    w = base_w.clone()
    huber_px = 8.0  # robust threshold in pixels

    for it in range(iters):
        opt.zero_grad()
        theta = theta0 if freeze_theta else p_to_theta(p, lo, hi)
        if fix_joint7:
            theta = torch.cat([theta[:, :6], torch.zeros(B, 1, device=device)], dim=1)
        fk = panda_forward_kinematics(theta)            # (B,N,3)
        R = rot6d_to_matrix(d6)
        uv, _ = project_points(fk, R, t, K)             # (B,N,2)
        err = uv - kp_2d                                 # (B,N,2)
        if cov_inv is not None:
            # Mahalanobis/whitened residual: sharp peaks count at full strength, diffuse
            # (occluded) peaks are attenuated per-direction. Scale by a nominal sigma so the
            # whitened magnitude stays in "pixels" for the shared Huber/IRLS thresholds.
            quad = torch.einsum('bni,bnij,bnj->bn', err, cov_inv, err).clamp(min=0)
            resid_px = quad.sqrt() * 2.0                 # nominal sigma 2px -> whitened px
        else:
            resid_px = err.norm(dim=-1)                  # (B,N) pixels
        # IRLS robust reweighting (Geman-McClure-ish): outliers get down-weighted.
        if it > 30:
            if robust_scale > 0.0:
                # ADAPTIVE GM scale (MAD-style): tune the down-weighting knee from the per-frame
                # median residual over the CURRENTLY-KEPT keypoints, not a fixed 8px. On diffuse
                # KUKA heatmaps the residuals sit far above 8px, so a fixed knee treats almost
                # everything as an inlier; a median-relative knee re-establishes the outlier gap.
                rd = resid_px.detach()
                med = masked_median(rd, base_w > 0)                     # (B,)
                c = (robust_scale * med).clamp(min=2.0).unsqueeze(1)    # (B,1) px
                robust = (c ** 2) / (c ** 2 + rd ** 2)
            else:
                robust = (huber_px ** 2) / (huber_px ** 2 + resid_px.detach() ** 2)
            w = base_w * robust
            w = w / w.sum(dim=1, keepdim=True).clamp(min=1e-6)
        if cov_inv is not None:
            # Huber on the whitened distance (already combines both axes)
            res_n = resid_px / img_size
            loss_per = F.huber_loss(res_n, torch.zeros_like(res_n), delta=0.01, reduction='none')
        else:
            # LEGACY path preserved bit-exact: per-component Huber on normalized residual
            res = err / img_size
            loss_per = F.huber_loss(res, torch.zeros_like(res), delta=0.01, reduction='none').sum(-1)
        loss = (w * loss_per).sum(dim=1).mean()
        # Light prior on theta to resolve depth-ambiguous null-space directions.
        loss = loss + prior_w * ((theta[:, :6] - theta_mean[:6]) ** 2).mean()
        if prior_adaptive > 0.0:
            # Occlusion-adaptive configuration prior ("masked-state prior", analytic form):
            # DREAM synth joints are INDEPENDENT (max |corr| 0.06) but NOT uniform, so the full
            # information content of a learned state prior reduces to per-joint Gaussians.
            # Weight grows as keypoint evidence disappears -> "the less we see, the more we lean
            # on the plausible-configuration prior" (per-frame, differentiable).
            sigma = torch.tensor([1.02, 0.65, 0.50, 0.50, 0.75, 0.50], device=device)  # synth stds
            vis_frac = (conf > max(conf_gate, 0.05)).float().mean(dim=1)               # (B,)
            occ_w = prior_adaptive * (1.0 - vis_frac).clamp(min=0.0)                   # (B,)
            maha = (((theta[:, :6] - theta_mean[:6]) / sigma) ** 2).mean(dim=1)        # (B,)
            loss = loss + (occ_w * maha).mean()
        # GT-depth ceiling probe: anchor solved root depth t_z to GT base depth, re-solve R,theta
        # consistently around it (gauge-safe oracle test of "would correct depth fix the pose?").
        if depth_w > 0.0 and gt_tz is not None:
            loss = loss + depth_w * ((t[:, 2] - gt_tz) ** 2).mean()
        # Anchor to the LEARNED init: occluded joints (no data constraint after gating)
        # fall back to the learned prediction instead of drifting.
        if anchor_init_w > 0.0:
            loss = loss + anchor_init_w * ((theta[:, :6] - theta_init_p[:, :6]) ** 2).mean()
        loss.backward()
        opt.step()

    with torch.no_grad():
        theta = p_to_theta(p, lo, hi)
        if fix_joint7:
            theta = torch.cat([theta[:, :6], torch.zeros(B, 1, device=device)], dim=1)
        fk = panda_forward_kinematics(theta)
        R = rot6d_to_matrix(d6)
        uv, kp_cam = project_points(fk, R, t, K)
        reproj_px = (uv - kp_2d).norm(dim=-1).mean(dim=1)  # (B,)
        t_out = t.clone()

        # Per-frame divergence guard: keep refined only where it lowered reprojection,
        # else fall back to the (learned) init. Refinement is then never harmful.
        # nan reproj (degenerate minimal-PnP frame) counts as worse -> fall back to init.
        worse = ~(reproj_px < reproj_init)
        if worse.any():
            theta[worse] = theta0[worse]
            kp_cam[worse] = kpcam_init[worse]
            reproj_px[worse] = reproj_init[worse]
            R[worse] = R0m[worse]
            t_out[worse] = t0[worse]

    # --- SHED re-solve: soft floor. Drop residual-outlier keypoints and re-fit below the floor ---
    # The min_kp floor forces the top-min_kp confident keypoints into EVERY fit. On visually
    # near-identical links (KUKA) a forced-in but geometrically-inconsistent keypoint can pull the
    # solve into a divergent pose. Here, AFTER an initial solve, we flag keypoints whose reprojection
    # residual exceeds shed_k x the per-frame median, drop them (drop_mask bypasses the floor), and
    # re-solve inheriting the same init. STRUCTURAL do-no-harm: the candidate is compared to the
    # incumbent ON THE RETAINED KEYPOINTS ONLY (a fair, GT-free geometric test) and adopted only if
    # it is better there — so shedding a genuinely-good point that was merely sacrificed cannot win.
    if shed_k > 0.0 and not freeze_theta and drop_mask is None:
        with torch.no_grad():
            fk_s = panda_forward_kinematics(theta)
            uv_s, _ = project_points(fk_s, R, t_out, K)
            rj = (uv_s - kp_2d).norm(dim=-1)                       # (B,N) per-kp residual px
            kept = base_w > 0
            med = masked_median(rj, kept)                         # (B,)
            shed = kept & (rj > shed_k * med.unsqueeze(1))        # (B,N) outliers among kept
            n_after = (kept & ~shed).sum(dim=1)
            do_shed = shed.any(dim=1) & (n_after >= 4)            # never starve below 4 points
        nshed = int(do_shed.sum())
        if _shed_stats is not None:
            _shed_stats['flagged'] = _shed_stats.get('flagged', 0) + nshed
            _shed_stats['total'] = _shed_stats.get('total', 0) + B
        if do_shed.any():
            idx = do_shed.nonzero(as_tuple=True)[0]
            dm = shed[idx]
            th_c, kc_c, rp_c, R_c, t_c = solve_batch(
                kp_2d[idx], conf[idx], K[idx], fix_joint7=fix_joint7, iters=iters, lr=lr,
                img_size=img_size, device=device, prior_w=prior_w, theta_init=theta0[idx],
                conf_gate=conf_gate, anchor_init_w=anchor_init_w, min_kp=min_kp,
                pnp_rel=pnp_rel, pnp_drop=pnp_drop,
                R_init=(R_init[idx] if R_init is not None else None),
                t_init=(t_init[idx] if t_init is not None else None),
                gt_tz=(gt_tz[idx] if gt_tz is not None else None), depth_w=depth_w,
                return_pose=True, cov_inv=(cov_inv[idx] if cov_inv is not None else None),
                prior_adaptive=prior_adaptive, freeze_theta=False, border_margin=border_margin,
                resolve_reproj_thr=0.0, robust_scale=robust_scale, shed_k=0.0, drop_mask=dm)
            with torch.no_grad():
                retain = (~dm).to(kp_2d.dtype)                    # (len(idx),N) retained-kp mask
                den = retain.sum(1).clamp(min=1)
                fk_o = panda_forward_kinematics(theta[idx])
                uv_o, _ = project_points(fk_o, R[idx], t_out[idx], K[idx])
                ro = ((uv_o - kp_2d[idx]).norm(dim=-1) * retain).sum(1) / den
                fk_n = panda_forward_kinematics(th_c)
                uv_n, _ = project_points(fk_n, R_c, t_c, K[idx])
                rn = ((uv_n - kp_2d[idx]).norm(dim=-1) * retain).sum(1) / den
                better = rn < ro
                if better.any():
                    gi = idx[better]
                    theta[gi] = th_c[better]; kp_cam[gi] = kc_c[better]
                    reproj_px[gi] = rp_c[better]; R[gi] = R_c[better]; t_out[gi] = t_c[better]
                    if _shed_stats is not None:
                        _shed_stats['adopted'] = _shed_stats.get('adopted', 0) + int(better.sum())

    # --- Multi-start / grown-set re-solve on frames flagged by high solved reprojection ---
    # The diagnosed failure is a WRONG-BASIN init, which the guard above cannot repair (it can
    # only fall back to that same broken init). So re-run the whole solve from alternative PnP
    # inits: each candidate uses a different minimal-set size and, crucially, does NOT take the
    # learned R_init override — so candidates can land in genuinely different rotation basins.
    if resolve_reproj_thr > 0.0 and not freeze_theta:
        flagged = reproj_px > resolve_reproj_thr
        nflag = int(flagged.sum())
        if _resolve_stats is not None:
            _resolve_stats['flagged'] = _resolve_stats.get('flagged', 0) + nflag
            _resolve_stats['total'] = _resolve_stats.get('total', 0) + B
        if nflag > 0:
            idx = flagged.nonzero(as_tuple=True)[0]
            adopted = torch.zeros(len(idx), dtype=torch.bool, device=device)

            def _sub(x):
                return None if x is None else x[idx]

            for drop in resolve_drops:
                th_c, kc_c, rp_c, R_c, t_c = solve_batch(
                    kp_2d[idx], conf[idx], K[idx], fix_joint7=fix_joint7, iters=iters, lr=lr,
                    img_size=img_size, device=device, prior_w=prior_w,
                    theta_init=theta0[idx], conf_gate=conf_gate, anchor_init_w=anchor_init_w,
                    min_kp=min_kp, pnp_rel=pnp_rel, pnp_drop=drop,
                    R_init=None, t_init=None, gt_tz=_sub(gt_tz), depth_w=depth_w,
                    return_pose=True, cov_inv=_sub(cov_inv),
                    prior_adaptive=prior_adaptive, freeze_theta=False,
                    border_margin=border_margin, resolve_reproj_thr=0.0,
                    robust_scale=robust_scale, shed_k=0.0)
                # STRUCTURAL do-no-harm: adopt ONLY where the candidate lowers the solver's own
                # reprojection residual. Never adopt a worse-reprojection solution.
                with torch.no_grad():
                    better = rp_c < reproj_px[idx]
                    if better.any():
                        gi = idx[better]
                        theta[gi] = th_c[better]
                        kp_cam[gi] = kc_c[better]
                        reproj_px[gi] = rp_c[better]
                        R[gi] = R_c[better]
                        t_out[gi] = t_c[better]
                        adopted |= better
            if _resolve_stats is not None:
                _resolve_stats['adopted'] = _resolve_stats.get('adopted', 0) + int(adopted.sum())

    # --- Z-BOUND re-solve: physical-plausibility depth constraint on the whole skeleton ---
    # The z-collapse failure: monocular scale ambiguity lets the solve settle at a near-zero base
    # depth that reprojects WELL (low residual) but is physically impossible. Every reproj-keyed
    # guard above (resolve, shed) structurally misses it because the collapsed basin has the LOWEST
    # reprojection. Here we key on the PHYSICS: median camera-frame skeleton depth out of [lo,hi] ->
    # re-init the whole skeleton to a plausible depth (z_target) and re-solve, holding depth with an
    # anchor so R,theta re-fit the 2D at a realistic distance instead of re-collapsing. Adopt on
    # recovered physical validity (median depth back in-bounds), NOT reprojection.
    if z_bound is not None and not freeze_theta and drop_mask is None:
        lo_z, hi_z = float(z_bound[0]), float(z_bound[1])
        with torch.no_grad():
            keptm = base_w > 0
            med_z = masked_median(kp_cam[..., 2], keptm)          # (B,) median cam depth
            bad = (med_z < lo_z) | (med_z > hi_z)
        nbad = int(bad.sum())
        if _zbound_stats is not None:
            _zbound_stats['flagged'] = _zbound_stats.get('flagged', 0) + nbad
            _zbound_stats['total'] = _zbound_stats.get('total', 0) + B
        if bad.any():
            idx = bad.nonzero(as_tuple=True)[0]
            with torch.no_grad():
                if z_target > 0.0:
                    z_re = torch.full((len(idx),), float(z_target), device=device).clamp(lo_z, hi_z)
                else:
                    z_re = med_z[idx].clamp(lo_z, hi_z)           # per-frame floor into bounds
                # translate the whole camera-frame skeleton along z so its median lands at z_re
                # (base-link is at the robot origin, so t_z IS the base depth); hold with the anchor.
                t_re = t_out[idx].clone()
                t_re[:, 2] = t_re[:, 2] + (z_re - med_z[idx])
            th_c, kc_c, rp_c, R_c, t_c = solve_batch(
                kp_2d[idx], conf[idx], K[idx], fix_joint7=fix_joint7, iters=iters, lr=lr,
                img_size=img_size, device=device, prior_w=prior_w, theta_init=theta0[idx],
                conf_gate=conf_gate, anchor_init_w=anchor_init_w, min_kp=min_kp,
                pnp_rel=pnp_rel, pnp_drop=pnp_drop, R_init=R[idx], t_init=t_re,
                gt_tz=z_re, depth_w=z_anchor_w, return_pose=True,
                cov_inv=(cov_inv[idx] if cov_inv is not None else None),
                prior_adaptive=prior_adaptive, freeze_theta=False, border_margin=border_margin,
                resolve_reproj_thr=0.0, robust_scale=robust_scale, shed_k=0.0, z_bound=None)
            with torch.no_grad():
                med_z_c = masked_median(kc_c[..., 2], base_w[idx] > 0)
                ok = (med_z_c >= lo_z) & (med_z_c <= hi_z)        # physical validity, NOT reproj
                if ok.any():
                    gi = idx[ok]
                    theta[gi] = th_c[ok]; kp_cam[gi] = kc_c[ok]
                    reproj_px[gi] = rp_c[ok]; R[gi] = R_c[ok]; t_out[gi] = t_c[ok]
                    if _zbound_stats is not None:
                        _zbound_stats['recovered'] = _zbound_stats.get('recovered', 0) + int(ok.sum())

    if return_pose:
        return theta.detach(), kp_cam.detach(), reproj_px.detach(), R.detach(), t_out.detach()
    return theta.detach(), kp_cam.detach(), reproj_px.detach()


def sample_heatmap_at(heatmaps, uv, H, W):
    """heatmaps (B,N,H,W), uv (B,N,2) px -> (B,N) bilinearly-sampled heatmap value at uv.
    Differentiable w.r.t. uv (-> w.r.t. theta,R,t). 'border' padding keeps gradient finite
    when a projection lands off-image."""
    B, N = uv.shape[:2]
    hm = heatmaps.reshape(B * N, 1, H, W)
    g = uv.reshape(B * N, 1, 1, 2).clone()
    g0 = g[..., 0] / max(W - 1, 1) * 2 - 1
    g1 = g[..., 1] / max(H - 1, 1) * 2 - 1
    grid = torch.stack([g0, g1], dim=-1)
    s = F.grid_sample(hm, grid, mode='bilinear', align_corners=True, padding_mode='border')
    return s.reshape(B, N)


def solve_batch_heatmap(heatmaps, K, fix_joint7=True, iters=200, lr=1e-2,
                        img_size=512, device='cuda', theta_init=None,
                        anchor_w=0.15, prior_w=2e-3):
    """
    Heatmap-based BIDIRECTIONAL refiner. Instead of fitting the argmax keypoints, optimize
    (theta, R, t) to MAXIMIZE the heatmap response at the reprojected FK keypoints — so the
    kinematic chain (FK) selects the heatmap mode that is structurally consistent and IGNORES
    broken argmax detections (kinematics corrects keypoints). A weak anchor to the argmax keeps
    the basin; per-frame guard keeps the init if heatmap response doesn't improve.

    heatmaps: (B,N,H,W). K: (B,3,3). theta_init: (B,N_ang+1) learned angles.
    Returns theta (B,7), kp_cam (B,N,3), response (B,).
    """
    B, N, H, W = heatmaps.shape
    dtype = torch.float32
    lo, hi = make_limits(device, dtype)
    theta_mean = PANDA_JOINT_MEAN.to(device, dtype)

    kp_argmax = soft_argmax_2d(heatmaps)                 # (B,N,2)
    conf = heatmaps.flatten(2).max(dim=2)[0]             # (B,N)
    hm_norm = heatmaps / (heatmaps.flatten(2).max(dim=2)[0].view(B, N, 1, 1) + 1e-6)

    theta0 = (theta_init.to(device, dtype).clone() if theta_init is not None
              else theta_mean.unsqueeze(0).expand(B, 7).clone())
    if fix_joint7:
        theta0[:, 6] = 0.0
    theta0 = torch.max(torch.min(theta0, hi), lo)
    fk0 = panda_forward_kinematics(theta0)
    R0, t0, _ = pnp_init(kp_argmax.cpu().numpy(), fk0.detach().cpu().numpy(),
                         K.cpu().numpy(), conf.cpu().numpy())
    R0 = torch.from_numpy(R0).to(device, dtype); t0 = torch.from_numpy(t0).to(device, dtype)

    base_w = conf.clamp(min=1e-3); base_w = base_w / base_w.sum(dim=1, keepdim=True)

    # init heatmap response (for the guard)
    with torch.no_grad():
        uv_i, kpcam_init = project_points(fk0, rot6d_to_matrix(matrix_to_rot6d(R0)), t0, K)
        resp_init = (base_w * sample_heatmap_at(hm_norm, uv_i, H, W)).sum(dim=1)

    p = theta_to_p(theta0, lo, hi).clone().detach().requires_grad_(True)
    d6 = matrix_to_rot6d(R0).clone().detach().requires_grad_(True)
    t = t0.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([p, d6, t], lr=lr)

    for _ in range(iters):
        opt.zero_grad()
        theta = p_to_theta(p, lo, hi)
        if fix_joint7:
            theta = torch.cat([theta[:, :6], torch.zeros(B, 1, device=device)], dim=1)
        fk = panda_forward_kinematics(theta)
        R = rot6d_to_matrix(d6)
        uv, _ = project_points(fk, R, t, K)
        s = sample_heatmap_at(hm_norm, uv, H, W)         # (B,N) in [0,1]
        loss_hm = -(base_w * s).sum(dim=1).mean()        # maximize heatmap response
        # weak anchor to argmax (basin), confidence-weighted, robust
        res = (uv - kp_argmax) / img_size
        anchor = (base_w * F.huber_loss(res, torch.zeros_like(res), delta=0.02,
                                        reduction='none').sum(-1)).sum(dim=1).mean()
        loss = loss_hm + anchor_w * anchor + prior_w * ((theta[:, :6] - theta_mean[:6]) ** 2).mean()
        loss.backward(); opt.step()

    with torch.no_grad():
        theta = p_to_theta(p, lo, hi)
        if fix_joint7:
            theta = torch.cat([theta[:, :6], torch.zeros(B, 1, device=device)], dim=1)
        fk = panda_forward_kinematics(theta)
        R = rot6d_to_matrix(d6)
        uv, kp_cam = project_points(fk, R, t, K)
        resp = (base_w * sample_heatmap_at(hm_norm, uv, H, W)).sum(dim=1)
        # guard: keep refined only where heatmap response improved, else fall back to init
        worse = resp < resp_init
        if worse.any():
            theta[worse] = theta0[worse]
            kp_cam[worse] = kpcam_init[worse]
            resp[worse] = resp_init[worse]
    return theta.detach(), kp_cam.detach(), resp.detach()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
# NOTE: the standalone diagnostic CLI of the research tree was removed for the release;
# the entry point is scripts/eval.py.
