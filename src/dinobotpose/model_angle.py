"""
Learned joint-angle predictor (Stage 1.5).

Replaces the global-feature DirectJointAngleHead (which was ill-posed -> ~17-19 deg)
with a head that consumes the DETECTED 2D keypoint geometry + DINOv3 appearance:

  K-normalized bearings + all-pairs bearing differences   (probe: 7.36 deg from 2D ALONE)
  + per-joint heatmap confidence
  + DINOv3 global feature
  + per-keypoint sampled local features (grid_sample at the detected 2D locations)
      -> MLP -> sin/cos -> joint angles

The DINOv3 backbone and the (Stage-1) keypoint head are FROZEN; only AngleHead trains.
Training uses the model's OWN predicted 2D keypoints so the head is robust to real
detector noise. Output feeds the kinematic refiner (Eval/solve_pose_kinematic.py) as init.
"""
import itertools
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_v4 import DINOv3Backbone, ViTKeypointHead, soft_argmax_2d, panda_forward_kinematics

NUM_KP = 7
NUM_ANG = 6  # joint 7 fixed to 0


def to_bearings(kp2d, K):
    """kp2d (B,7,2) px, K (B,3,3) -> per-point bearings (B,7,2) = ((x-cx)/fx,(y-cy)/fy).
    Camera-intrinsics-invariant direction of each keypoint."""
    fx = K[:, 0, 0:1]; fy = K[:, 1, 1:2]
    cx = K[:, 0, 2:3]; cy = K[:, 1, 2:3]
    bx = (kp2d[:, :, 0] - cx) / fx          # (B,7)
    by = (kp2d[:, :, 1] - cy) / fy
    return torch.stack([bx, by], dim=-1)    # (B,7,2)


def geo_dim(n_kp):
    """Length of the keypoints_to_geo vector for n_kp keypoints: 2*n + 2*C(n,2)."""
    return 2 * n_kp + n_kp * (n_kp - 1)


def keypoints_to_geo(kp2d, K):
    """
    Returns geometric feature (B, 2n + 2*C(n,2)) = per-point bearings + all-pairs differences
    (n = number of keypoints, inferred from kp2d). For n=7 this is (B, 14+42)=(B,56), unchanged.
    Pairwise differences are translation-invariant relative geometry (encode link orientations).
    """
    bearings = to_bearings(kp2d, K)           # (B,n,2)
    n = bearings.shape[1]
    diffs = []
    for i, j in itertools.combinations(range(n), 2):
        diffs.append(bearings[:, i] - bearings[:, j])  # (B,2)
    diffs = torch.cat(diffs, dim=1)            # (B, 2*C(n,2))
    return torch.cat([bearings.reshape(bearings.shape[0], -1), diffs], dim=1)  # (B, 2n+2*C(n,2))


def sample_kp_features(tokens, kp2d, heatmap_size):
    """
    tokens: (B, Npatch, C) DINOv3 patch tokens. kp2d: (B,7,2) px @ heatmap res.
    Returns per-keypoint features (B,7,C) via bilinear grid_sample on the token map.
    """
    B, Np, C = tokens.shape
    s = int(round(Np ** 0.5))
    fmap = tokens.transpose(1, 2).reshape(B, C, s, s)        # (B,C,s,s)
    H = W = heatmap_size
    grid = kp2d.clone()
    grid[..., 0] = grid[..., 0] / (W - 1) * 2 - 1            # -> [-1,1]
    grid[..., 1] = grid[..., 1] / (H - 1) * 2 - 1
    grid = grid.unsqueeze(1)                                  # (B,1,7,2)
    samp = F.grid_sample(fmap, grid, mode='bilinear', align_corners=True)  # (B,C,1,7)
    return samp.squeeze(2).transpose(1, 2)                    # (B,7,C)


def sample_kp_features_map(fmap, kp2d, heatmap_size):
    """Conv-feature-map variant of sample_kp_features (for the P1b trainable ResNet trunk).
    fmap: (B,C,h,w) conv feature map (any spatial res). kp2d: (B,7,2) px @ heatmap res.
    Returns per-keypoint features (B,7,C) via bilinear grid_sample. Normalization uses the
    heatmap extent so it is independent of the feature-map resolution."""
    H = W = heatmap_size
    grid = kp2d.clone()
    grid[..., 0] = grid[..., 0] / (W - 1) * 2 - 1            # -> [-1,1]
    grid[..., 1] = grid[..., 1] / (H - 1) * 2 - 1
    grid = grid.unsqueeze(1)                                  # (B,1,7,2)
    samp = F.grid_sample(fmap, grid, mode='bilinear', align_corners=True)  # (B,C,1,7)
    return samp.squeeze(2).transpose(1, 2)                    # (B,7,C)


def sample_kp_patch(tokens, kp2d, heatmap_size, k=3, step_px=16.0):
    """
    Per-keypoint k×k feature PATCH (vs the single-point sample above). A single bilinear
    sample misses the local ORIENTATION of a link/gripper — which is exactly what wrist-roll
    angles need (the keypoints barely move, so 2D geometry is degenerate; appearance resolves
    it). Returns (B, 7, k*k*C): the k×k grid of token features centered on each keypoint.
    step_px = spacing between grid taps in heatmap px (≈ one 16px patch token).
    """
    B, Np, C = tokens.shape
    s = int(round(Np ** 0.5))
    fmap = tokens.transpose(1, 2).reshape(B, C, s, s)        # (B,C,s,s)
    H = W = heatmap_size
    off = (torch.arange(k, device=kp2d.device, dtype=kp2d.dtype) - (k - 1) / 2) * step_px  # (k,)
    oy, ox = torch.meshgrid(off, off, indexing='ij')         # (k,k)
    offs = torch.stack([ox.reshape(-1), oy.reshape(-1)], dim=-1)  # (k*k,2) px
    # (B,7,1,2) + (1,1,kk,2) -> (B,7,kk,2)
    pts = kp2d.unsqueeze(2) + offs.view(1, 1, -1, 2)
    g = pts.clone()
    g[..., 0] = g[..., 0] / (W - 1) * 2 - 1
    g[..., 1] = g[..., 1] / (H - 1) * 2 - 1                   # (B,7,kk,2)
    samp = F.grid_sample(fmap, g, mode='bilinear', align_corners=True, padding_mode='border')  # (B,C,7,kk)
    samp = samp.permute(0, 2, 3, 1).reshape(B, NUM_KP, k * k * C)  # (B,7,kk*C)
    return samp


class AngleHead(nn.Module):
    def __init__(self, feat_dim=768, hidden=512, n_ang=NUM_ANG, dropout=0.1, kp_in=None, n_kp=NUM_KP):
        super().__init__()
        self.n_ang = n_ang
        self.n_kp = n_kp
        kp_in = kp_in or feat_dim          # per-keypoint feature dim (feat_dim, or k*k*feat_dim for patch)
        self.geo_mlp = nn.Sequential(
            nn.Linear(geo_dim(n_kp), 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 256), nn.GELU(),
        )
        self.conf_proj = nn.Sequential(nn.Linear(n_kp, 64), nn.GELU())
        self.global_proj = nn.Sequential(nn.Linear(feat_dim, 256), nn.GELU())
        self.kp_proj = nn.Sequential(nn.Linear(kp_in, 128), nn.GELU())
        fuse_in = 256 + 64 + 256 + n_kp * 128
        self.fuse = nn.Sequential(
            nn.Linear(fuse_in, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, n_ang * 2),
        )

    def forward(self, geo, conf, gfeat, kpfeat):
        g = self.geo_mlp(geo)
        c = self.conf_proj(conf)
        gl = self.global_proj(gfeat)
        kp = self.kp_proj(kpfeat).flatten(1)          # (B, n_kp*128)
        x = torch.cat([g, c, gl, kp], dim=1)
        sc = self.fuse(x).view(-1, self.n_ang, 2)
        sc = F.normalize(sc, dim=-1)                  # unit sin/cos
        ang = torch.atan2(sc[..., 0], sc[..., 1])     # (B, n_ang)
        return ang, sc


class AngleHeadMCL(AngleHead):
    """Multiple-hypothesis (Multiple-Choice-Learning) angle head. Under occlusion the posterior
    p(angles|image) is MULTIMODAL (several joint configs explain the visible 2D, esp. base-yaw J0);
    a single regressor lands on the wrong MEAN of the modes. This head emits K hypotheses; an MCL
    'winner-take-all' loss (only the best hypothesis per frame is supervised) makes them SPECIALIZE
    into the modes, and at inference the kinematic solver/reprojection SELECTS the consistent one.
    Returns ang (B,K,n_ang), sc (B,K,n_ang,2)."""
    def __init__(self, feat_dim=768, hidden=512, n_ang=NUM_ANG, dropout=0.1, kp_in=None, n_hyp=4):
        super().__init__(feat_dim=feat_dim, hidden=hidden, n_ang=n_ang, dropout=dropout, kp_in=kp_in)
        self.n_hyp = n_hyp
        # replace the final projection to emit K hypotheses
        in_f = self.fuse[-1].in_features
        self.fuse[-1] = nn.Linear(in_f, n_hyp * n_ang * 2)

    def forward(self, geo, conf, gfeat, kpfeat):
        g = self.geo_mlp(geo)
        c = self.conf_proj(conf)
        gl = self.global_proj(gfeat)
        kp = self.kp_proj(kpfeat).flatten(1)
        x = torch.cat([g, c, gl, kp], dim=1)
        sc = self.fuse(x).view(-1, self.n_hyp, self.n_ang, 2)
        sc = F.normalize(sc, dim=-1)                  # (B,K,n_ang,2)
        ang = torch.atan2(sc[..., 0], sc[..., 1])     # (B,K,n_ang)
        return ang, sc


class AngleHeadMixSel(AngleHeadMCL):
    """MCL hypotheses + a LEARNED APPEARANCE selector (committee P3). MCL lets the geometric solver
    select the hypothesis at test, which is degenerate under close-range 90px reproj (why MCL was
    refuted). Instead this head predicts p(mode|appearance) from the frozen global feature and picks
    directly. Pre-checks passed: catastrophic-frame error is SPREAD (global selector OK) and a linear
    probe on gfeat predicts will-flip at AUC 0.83 (appearance carries the mode signal).
    Returns ang (B,K,n_ang), sc (B,K,n_ang,2), sel_logits (B,K)."""
    def __init__(self, feat_dim=768, hidden=512, n_ang=NUM_ANG, dropout=0.1, kp_in=None, n_hyp=2):
        super().__init__(feat_dim=feat_dim, hidden=hidden, n_ang=n_ang, dropout=dropout, kp_in=kp_in, n_hyp=n_hyp)
        self.selector = nn.Sequential(nn.Linear(feat_dim, 128), nn.GELU(), nn.Linear(128, n_hyp))

    def forward(self, geo, conf, gfeat, kpfeat):
        ang, sc = super().forward(geo, conf, gfeat, kpfeat)   # (B,K,n_ang), (B,K,n_ang,2)
        sel_logits = self.selector(gfeat)                     # (B,K) from appearance only
        return ang, sc, sel_logits


class AngleHeadPARE(nn.Module):
    """PARE-style per-joint attention (Kocabas ICCV'21). Each joint has a LEARNED query that attends
    over the frozen backbone's spatial patch tokens -> its own pooled feature (an occluded joint
    borrows context from visible neighbours). Per-joint INDEPENDENT pathways add spatial information
    and avoid the shared-pooled-feature zero-sum that made the MoE regress the base joint. Fused with
    the keypoint-geometry vector so the observable joints keep their bearing cue. Returns ang (B,n_ang),
    sc (B,n_ang,2)."""
    def __init__(self, feat_dim=768, n_ang=NUM_ANG, hidden=256, dropout=0.1):
        super().__init__()
        self.n_ang = n_ang
        self.q = nn.Parameter(torch.randn(n_ang, feat_dim) * 0.02)   # per-joint queries
        self.k_proj = nn.Linear(feat_dim, feat_dim)
        self.v_proj = nn.Linear(feat_dim, feat_dim)
        self.scale = feat_dim ** -0.5
        self.geo_mlp = nn.Sequential(nn.Linear(14 + 42, 128), nn.GELU(), nn.Dropout(dropout))
        self.head = nn.Sequential(nn.Linear(feat_dim + 128, hidden), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 2))

    def forward(self, tokens, geo):                     # tokens (B,Np,C), geo (B,56)
        k = self.k_proj(tokens); v = self.v_proj(tokens)
        attn = torch.einsum('jc,bnc->bjn', self.q, k) * self.scale     # (B,n_ang,Np)
        attn = attn.softmax(dim=-1)
        f = torch.einsum('bjn,bnc->bjc', attn, v)                      # (B,n_ang,C) per-joint feature
        g = self.geo_mlp(geo).unsqueeze(1).expand(-1, self.n_ang, -1)  # (B,n_ang,128)
        sc = self.head(torch.cat([f, g], dim=-1))                      # (B,n_ang,2)
        sc = F.normalize(sc, dim=-1)
        ang = torch.atan2(sc[..., 0], sc[..., 1])
        return ang, sc


# RoboPEPP Panda joint_mean (test.py:113) — IEF state init (= mean joint config).
# In normalized space RoboPEPP inits at 0 (=mean after denorm); we init the sin/cos state
# at the mean config directly since our head regresses sin/cos, not a normalized angle.
PANDA_ANGLE_MEAN = torch.tensor([-0.0522, 0.2677, 0.0060, -2.0052, 0.0149, 1.9856])


class AngleHeadIEF(nn.Module):
    """RoboPEPP JointNet (model.py:15-51) Iterative-Error-Feedback port. Flat heads (mlp/patch/MoE)
    map the frozen fused feature -> angle in ONE shot and ceiling at good-frame ADD 0.788 (oracle-angle
    0.899); their θ is noisy so the solver over-fits 2D. IEF starts from the mean joint config and runs
    n_iter residual refinement steps, feeding the CURRENT angle state back as a condition each step so
    the kinematic-chain coupling is resolved iteratively (an inductive bias flat heads lack, frozen-
    backbone compatible). The fused CONTEXT feature (geo + conf + global gfeat + keypoint tokens, same
    fusion as AngleHead) is computed ONCE (no re-crop); only the sin/cos state is fed back.
    Returns (ang (B,n_ang), sc (B,n_ang,2)) — plus per-iterate sc (B,n_iter,n_ang,2) when deep_sup."""
    def __init__(self, feat_dim=768, hidden=1024, n_ang=NUM_ANG, dropout=0.3, kp_in=None,
                 n_iter=3, deep_sup=True):
        super().__init__()
        self.n_ang, self.n_iter, self.deep_sup = n_ang, n_iter, deep_sup
        kp_in = kp_in or feat_dim
        # --- context encoder: same fusion as AngleHead, computed once ---
        self.geo_mlp = nn.Sequential(nn.Linear(14 + 42, 256), nn.GELU(), nn.Dropout(dropout),
                                     nn.Linear(256, 256), nn.GELU())
        self.conf_proj = nn.Sequential(nn.Linear(NUM_KP, 64), nn.GELU())
        self.global_proj = nn.Sequential(nn.Linear(feat_dim, 256), nn.GELU())
        self.kp_proj = nn.Sequential(nn.Linear(kp_in, 128), nn.GELU())
        ctx = 256 + 64 + 256 + NUM_KP * 128
        # --- IEF regressor: [ctx, current sin/cos state (2*n_ang)] -> residual sin/cos ---
        self.fc1 = nn.Linear(ctx + 2 * n_ang, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.dec = nn.Linear(hidden, 2 * n_ang)
        self.d1, self.d2 = nn.Dropout(dropout), nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.dec.weight, gain=0.01)   # small initial residual (stability)
        nn.init.zeros_(self.dec.bias)
        m = PANDA_ANGLE_MEAN[:n_ang]
        self.register_buffer('init_sc', torch.stack([torch.sin(m), torch.cos(m)], -1).reshape(-1))  # (2*n_ang,)

    def forward(self, geo, conf, gfeat, kpfeat):
        ctx = torch.cat([self.geo_mlp(geo), self.conf_proj(conf), self.global_proj(gfeat),
                         self.kp_proj(kpfeat).flatten(1)], dim=1)          # (B,ctx) computed once
        B = ctx.shape[0]
        sc = self.init_sc.to(ctx.dtype).unsqueeze(0).expand(B, -1).clone()  # mean-config init
        iters = []
        for _ in range(self.n_iter):
            xc = self.d2(F.gelu(self.fc2(self.d1(F.gelu(self.fc1(torch.cat([ctx, sc], 1)))))))
            sc = sc + self.dec(xc)                                          # residual (IEF)
            scn = F.normalize(sc.view(B, self.n_ang, 2), dim=-1)           # back to unit circle
            iters.append(scn)
            sc = scn.view(B, -1)                                           # normalized state feedback
        ang = torch.atan2(iters[-1][..., 0], iters[-1][..., 1])
        if self.deep_sup:
            return ang, iters[-1], torch.stack(iters, 1)                   # (B,n_ang),(B,n_ang,2),(B,n_iter,n_ang,2)
        return ang, iters[-1]


def compute_k_value(kp2d_orig, K_orig, real_size_mm=1000.0):
    """RootNet geometric depth prior (HoRoPose `lib/core/function.py:88-97`).

        k = sqrt(fx * fy * real_w * real_h / area),  area = max(|bbox_w|, |bbox_h|)^2

    = "the depth at which a real_size_mm x real_size_mm object would fill this bbox". Returns mm.

    ⚠️ kp2d_orig / K_orig MUST be in the ORIGINAL image frame (before --crop-to-robot's
    crop+resize-to-512). A crop that is resized to a fixed side NORMALIZES apparent size away,
    which destroys the very cue this prior encodes. In training the value is precomputed by
    PoseEstimationDataset (`batch['k_value']`) from the pre-crop, pre-augmentation keypoints;
    this helper exists so eval code can build the same quantity from a detector bbox.

    kp2d_orig: (..., N, 2) px in the original frame.  K_orig: (..., 3, 3) original intrinsics.
    """
    x, y = kp2d_orig[..., 0], kp2d_orig[..., 1]
    bw = (x.amax(-1) - x.amin(-1)).abs()
    bh = (y.amax(-1) - y.amin(-1)).abs()
    area = torch.clamp(torch.maximum(bw, bh) ** 2, min=1.0)          # px^2
    fx, fy = K_orig[..., 0, 0], K_orig[..., 1, 1]
    return torch.sqrt(fx * fy * real_size_mm * real_size_mm / area)  # (...,) mm


class RotationHead(nn.Module):
    """
    Predicts the robot->camera rotation as a 6D vector (Zhou et al.) from DINOv3 APPEARANCE +
    keypoint geometry. The far-camera failure (realsense) is the solver landing in a wrong
    rotation basin: 2D reprojection is degenerate at distance, but appearance (which way the
    robot faces) resolves it. Used as the kinematic solver's R_init to pick the right basin
    (oracle R-init = +0.11 realsense ADD-AUC). Heavy on the global/appearance feature.

    use_rootnet_depth: replace the regressed t_z with the RootNet decomposition
    z = softplus(gamma) * k_value / 1000 (meters). WHY: the plain t branch regresses a residual
    around a FIXED t_base=[0,0,1.1] with no apparent-size cue at all, which is the direct cause
    of KUKA's 56 mm t-error (Panda hides it behind render-and-compare; KUKA has no usable mesh,
    so the feed-forward head must carry the scale->depth physics itself). x,y stay appearance
    regression so the direct-pose property is preserved (no keypoint-uv back-projection, hence
    no re-entry of the KUKA link-confusion tail).
    """
    def __init__(self, feat_dim=768, hidden=512, dropout=0.1, predict_t=False,
                 use_rootnet_depth=False, n_kp=NUM_KP):
        super().__init__()
        self.predict_t = predict_t
        self.use_rootnet_depth = use_rootnet_depth
        self.n_kp = n_kp
        self.geo_mlp = nn.Sequential(nn.Linear(geo_dim(n_kp), 256), nn.GELU(), nn.Dropout(dropout))
        self.conf_proj = nn.Sequential(nn.Linear(n_kp, 64), nn.GELU())
        self.global_proj = nn.Sequential(nn.Linear(feat_dim, 256), nn.GELU())
        self.kp_proj = nn.Sequential(nn.Linear(feat_dim, 128), nn.GELU())
        self.trunk = nn.Sequential(
            nn.Linear(256 + 64 + 256 + n_kp * 128, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.rot_out = nn.Linear(hidden, 6)
        # translation (robot-origin position in camera frame, meters). t_z (depth) is the lever;
        # apparent robot size in the bearings/features -> distance. Predicts a residual around a
        # base depth so the net starts near plausible scale.
        self.t_out = nn.Linear(hidden, 3) if predict_t else None
        self.t_base = torch.tensor([0.0, 0.0, 1.1])
        # RootNet depth: predict only the DIMENSIONLESS correction gamma to the geometric prior
        # k_value. Init so gamma == 1.0 exactly (softplus(0.5413)=1), i.e. the head starts at the
        # pure geometry ("the robot is a 1m x 1m box filling its bbox") and learns the O(1)
        # correction from there -- far easier than regressing raw metric depth.
        self.gamma_out = nn.Linear(hidden, 1) if (predict_t and use_rootnet_depth) else None
        if self.gamma_out is not None:
            nn.init.normal_(self.gamma_out.weight, std=1e-3)
            nn.init.constant_(self.gamma_out.bias, 0.5413248)   # softplus^-1(1.0)

    def forward(self, geo, conf, gfeat, kpfeat, k_value=None):
        x = torch.cat([self.geo_mlp(geo), self.conf_proj(conf), self.global_proj(gfeat),
                       self.kp_proj(kpfeat).flatten(1)], dim=1)
        h = self.trunk(x)
        d6 = self.rot_out(h)
        if self.predict_t:
            t = self.t_out(h) + self.t_base.to(h.device)
            if self.gamma_out is not None and k_value is not None:
                gamma = F.softplus(self.gamma_out(h)).squeeze(-1)          # (B,) > 0
                z = gamma * k_value.to(h.dtype).to(h.device) / 1000.0      # mm -> m, z > 0
                t = torch.cat([t[:, :2], z.unsqueeze(-1)], dim=-1)         # option (a): replace z only
            return d6, t
        return d6


def rot6d_to_matrix(d6):
    """(B,6) 6D rotation -> (B,3,3). Zhou et al. 2019."""
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def kabsch_batch(P, Q, w):
    """Batched rigid (R,t) minimizing ||R P + t - Q||. P,Q (B,N,3), w (B,N). -> R (B,3,3), t (B,3)."""
    w = w / w.sum(1, keepdim=True).clamp(min=1e-6)
    mp = (w.unsqueeze(-1) * P).sum(1, keepdim=True); mq = (w.unsqueeze(-1) * Q).sum(1, keepdim=True)
    H = ((w.unsqueeze(-1) * (P - mp)).transpose(1, 2)) @ (Q - mq)   # (B,3,3)
    U, S, Vt = torch.linalg.svd(H)
    d = torch.sign(torch.linalg.det(Vt.transpose(1, 2) @ U.transpose(1, 2)))
    D = torch.eye(3, device=P.device).unsqueeze(0).repeat(P.shape[0], 1, 1)
    D[:, 2, 2] = d
    R = Vt.transpose(1, 2) @ D @ U.transpose(1, 2)
    t = mq.squeeze(1) - torch.bmm(R, mp.transpose(1, 2)).squeeze(2)   # (B,3)
    return R, t


class AngleHeadTransformer(nn.Module):
    """
    DETR-style alternative to the MLP head. Each detected keypoint becomes a token
    (sampled feature + bearing + joint-id + confidence); a Transformer ENCODER models
    inter-joint relations; N_ANG learnable angle queries CROSS-ATTEND (decoder) to the
    keypoint tokens and each emits sin/cos. Attention is a better inductive bias for the
    kinematic chain than a flat concat-MLP.
    """
    def __init__(self, feat_dim=768, d=256, n_kp=NUM_KP, n_ang=NUM_ANG,
                 enc_layers=3, dec_layers=3, heads=8, dropout=0.1):
        super().__init__()
        self.n_ang = n_ang
        self.feat_proj = nn.Linear(feat_dim, d)
        self.bearing_proj = nn.Linear(2, d)
        self.conf_proj = nn.Linear(1, d)
        self.kp_id = nn.Embedding(n_kp, d)
        enc = nn.TransformerEncoderLayer(d, heads, dim_feedforward=4 * d, dropout=dropout,
                                         activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, enc_layers)
        self.angle_queries = nn.Parameter(torch.randn(n_ang, d) * 0.02)
        dec = nn.TransformerDecoderLayer(d, heads, dim_feedforward=4 * d, dropout=dropout,
                                         activation='gelu', batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(dec, dec_layers)
        self.out = nn.Linear(d, 2)

    def forward(self, kpfeat, bearings, conf):
        # kpfeat (B,7,C), bearings (B,7,2), conf (B,7)
        B = kpfeat.shape[0]
        ids = torch.arange(NUM_KP, device=kpfeat.device)
        tok = (self.feat_proj(kpfeat) + self.bearing_proj(bearings)
               + self.conf_proj(conf.unsqueeze(-1)) + self.kp_id(ids).unsqueeze(0))
        mem = self.encoder(tok)                                 # (B,7,d)
        q = self.angle_queries.unsqueeze(0).expand(B, -1, -1)   # (B,n_ang,d)
        dec = self.decoder(q, mem)                              # (B,n_ang,d)
        sc = F.normalize(self.out(dec), dim=-1)                 # (B,n_ang,2)
        ang = torch.atan2(sc[..., 0], sc[..., 1])
        return ang, sc


class AnglePredictor(nn.Module):
    """Frozen backbone + frozen keypoint head + trainable angle head ('mlp' or 'transformer')."""
    def __init__(self, dino_model_name, heatmap_size, fix_joint7_zero=True, head_type='mlp',
                 with_rotation=False, with_translation=False, n_hyp=4, angle_backbone='dino_frozen',
                 use_rootnet_depth=False, num_kp=NUM_KP, num_ang=NUM_ANG):
        super().__init__()
        # num_kp / num_ang: keypoint & predicted-angle counts. Default 7/6 (Panda/KUKA/Baxter-left,
        # unchanged). Whole-body Baxter uses 17/14 (both arms; no fix_joint7 since the hand keypoints
        # make w2 observable). Only the default 'mlp' angle head + rot head honor num_kp/num_ang.
        self.num_kp = num_kp
        self.num_ang = num_ang
        self.heatmap_size = heatmap_size if isinstance(heatmap_size, int) else heatmap_size[0]
        self.fix_joint7_zero = fix_joint7_zero
        self.head_type = head_type
        self.backbone = DINOv3Backbone(dino_model_name, unfreeze_blocks=0)
        _cfg = self.backbone.model.config
        _vcfg = getattr(_cfg, "vision_config", _cfg)  # Siglip config nests hidden_size under vision_config
        feat_dim = _vcfg.hidden_size
        self.keypoint_head = ViTKeypointHead(
            input_dim=feat_dim, num_joints=num_kp, heatmap_size=(self.heatmap_size, self.heatmap_size))
        self.kp_patch = int(os.environ.get('KP_PATCH_K', '3')) if head_type == 'mlp_patch' else 0  # k×k appearance patch
        self.n_hyp = n_hyp if head_type in ('mlp_mcl', 'mlp_mixsel') else 1
        # P1b: separate TRAINABLE angle-feature backbone. The frozen DINOv3 detector stays kp2d/conf-only;
        # a decoupled ImageNet-init ResNet50 trunk supplies the angle head's gfeat/kpfeat so the angle
        # feature can co-train WITHOUT touching the keypoint tokens (frozen-kp integrity is structural).
        # afeat = the feature dim the angle/rot heads consume (2048 for resnet50, else DINOv3 hidden).
        self.angle_backbone = angle_backbone
        if angle_backbone == 'resnet50':
            import torchvision as tv
            _rn = tv.models.resnet50(weights=tv.models.ResNet50_Weights.IMAGENET1K_V2)
            self.angle_feat = nn.Sequential(*list(_rn.children())[:-2])   # conv trunk -> (B,2048,h,w)
            afeat = 2048
        else:
            self.angle_feat = None
            afeat = feat_dim
        if head_type == 'transformer':
            self.angle_head = AngleHeadTransformer(feat_dim=feat_dim)
        elif head_type == 'mlp_patch':
            self.angle_head = AngleHead(feat_dim=feat_dim, kp_in=feat_dim * self.kp_patch ** 2)
        elif head_type == 'mlp_mcl':
            self.angle_head = AngleHeadMCL(feat_dim=feat_dim, n_hyp=n_hyp)
        elif head_type == 'mlp_mixsel':
            self.angle_head = AngleHeadMixSel(feat_dim=feat_dim, n_hyp=n_hyp)
        elif head_type == 'pare':
            self.angle_head = AngleHeadPARE(feat_dim=feat_dim)
        elif head_type == 'ief':
            self.angle_head = AngleHeadIEF(feat_dim=afeat, kp_in=afeat,
                                           n_iter=int(os.environ.get('IEF_ITERS', '3')))
        else:
            self.angle_head = AngleHead(feat_dim=afeat, kp_in=afeat, n_kp=num_kp, n_ang=num_ang)
        self.with_rotation = with_rotation
        self.with_translation = with_translation
        self.use_rootnet_depth = use_rootnet_depth
        # Rotation head ALWAYS lives on the FROZEN DINOv3 features (feat_dim, 768), even under a
        # resnet50 angle_backbone: P1b only retrained the ANGLE feature path; the deployed rot-head
        # checkpoint (rot_crop_*) was trained on DINOv3 gfeat/kpfeat and is NOT retrained, so it must
        # be built with 768 and fed DINOv3 features (see forward). For dino_frozen afeat==feat_dim, so
        # this is byte-identical to the prior construction.
        self.rot_head = RotationHead(feat_dim=feat_dim, predict_t=with_translation,
                                     use_rootnet_depth=use_rootnet_depth,
                                     n_kp=num_kp) if with_rotation else None

    def freeze_detector(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.keypoint_head.parameters():
            p.requires_grad = False
        self.backbone.eval()
        self.keypoint_head.eval()

    def forward(self, image, camera_K, kp_jitter=0.0, kp_drop=0.0, kp2d_override=None,
                k_value=None):
        # k_value: (B,) RootNet geometric depth prior in mm, computed in the ORIGINAL image frame
        # (see compute_k_value). Only consumed when the rot-head was built with use_rootnet_depth;
        # None -> the head falls back to the legacy t_out + t_base regression, unchanged.
        with torch.no_grad():
            tokens = self.backbone(image)                 # (B,Np,C)
            heatmaps = self.keypoint_head(tokens)         # (B,7,H,W)
            _win = int(os.environ.get('DECODE_WINDOW', '0'))   # 0 = global soft-argmax (default/locked)
            if _win > 0:
                from model_v4 import windowed_soft_argmax_2d
                kp2d = windowed_soft_argmax_2d(heatmaps, win=_win)   # robust to distractor 2nd-modes
            else:
                kp2d = soft_argmax_2d(heatmaps)           # (B,7,2) @ heatmap res
            conf = heatmaps.flatten(2).max(dim=2)[0]      # (B,7)
            if kp2d_override is not None:                 # diagnostic: inject GT/external 2D (heatmap res)
                kp2d = kp2d_override.to(kp2d.dtype)
        if kp_drop > 0.0:
            # OCCLUSION augmentation: randomly "occlude" keypoints (low conf + displaced position) to
            # induce the under-determination that makes p(angles|image) multimodal. Forces the MCL
            # hypotheses to diversify into the modes (synth has no real occlusion to do this).
            mask = torch.rand(conf.shape, device=conf.device) < kp_drop          # (B,7) occluded
            conf = conf * torch.where(mask, torch.full_like(conf, 0.05), torch.ones_like(conf))
            kp2d = kp2d + mask.unsqueeze(-1).float() * torch.randn_like(kp2d) * 25.0
        if kp_jitter > 0.0:
            # train-time keypoint-noise augmentation: the head AMPLIFIES detector 2D error on the
            # gauge-sensitive base joint (realsense J0 44 deg vs 14 deg oracle-2D). Jittering the
            # 2D fed to geo+sampling teaches robustness so J0 leans on appearance, not exact bearings.
            kp2d = kp2d + torch.randn_like(kp2d) * kp_jitter
        out_iters = None                                  # IEF deep-sup per-iterate sin/cos (None otherwise)
        # P1b: TRAINABLE angle-feature map (grad on) from the decoupled ResNet50 trunk. kp2d/conf come
        # from the frozen DINOv3 detector above; only the angle feature is co-trained here.
        angle_fmap = None
        if self.angle_backbone == 'resnet50':
            x256 = F.interpolate(image, size=256, mode='bilinear', align_corners=False)
            angle_fmap = self.angle_feat(x256)               # (B,2048,h,w) trainable
        if self.head_type == 'transformer':
            kpfeat = sample_kp_features(tokens, kp2d, self.heatmap_size)
            bearings = to_bearings(kp2d, camera_K)            # (B,7,2)
            ang, sc = self.angle_head(kpfeat, bearings, conf)
        else:
            if self.angle_backbone == 'resnet50':
                gfeat = angle_fmap.mean(dim=(2, 3))                              # GAP -> (B,2048)
                kpfeat = sample_kp_features_map(angle_fmap, kp2d, self.heatmap_size)  # (B,7,2048)
            elif self.kp_patch:
                kpfeat = sample_kp_patch(tokens, kp2d, self.heatmap_size, k=self.kp_patch)
                gfeat = tokens.mean(dim=1)
            else:
                kpfeat = sample_kp_features(tokens, kp2d, self.heatmap_size)
                gfeat = tokens.mean(dim=1)
            geo = keypoints_to_geo(kp2d, camera_K)
            if self.head_type == 'mlp_mixsel':
                ang, sc, sel_logits = self.angle_head(geo, conf, gfeat, kpfeat)  # (B,K,6),(B,K,6,2),(B,K)
            elif self.head_type == 'pare':
                ang, sc = self.angle_head(tokens, geo)                # per-joint attention over patch tokens
            elif self.head_type == 'ief':
                r = self.angle_head(geo, conf, gfeat, kpfeat)
                if len(r) == 3:
                    ang, sc, out_iters = r                            # deep-sup: (B,6),(B,6,2),(B,n_iter,6,2)
                else:
                    ang, sc = r                                       # final-only
            else:
                ang, sc = self.angle_head(geo, conf, gfeat, kpfeat)   # (B,6),(B,6,2)
        mcl = self.head_type == 'mlp_mcl'              # ang (B,K,6), sc (B,K,6,2)
        mixsel = self.head_type == 'mlp_mixsel'
        if self.fix_joint7_zero:
            zpad = torch.zeros(*ang.shape[:-1], 1, device=ang.device)
            ang_full = torch.cat([ang, zpad], dim=-1)  # (B,7) / (B,K,7)
        else:
            ang_full = ang
        out = {
            'joint_angles': ang_full,   # (B,7) or (B,K,7) radians
            'sin_cos': sc,              # (B,6,2) or (B,K,6,2)
            'keypoints_2d': kp2d,       # (B,7,2)
            'heatmaps_2d': heatmaps,
            'confidence': conf,
            'is_mcl': mcl,
            'global_feat': gfeat if self.head_type != 'transformer' else tokens.mean(dim=1),
        }
        if out_iters is not None:
            out['sin_cos_iters'] = out_iters          # (B,n_iter,6,2) IEF deep supervision
        if mixsel:
            # APPEARANCE selector picks the hypothesis (not the solver). Expose full hypotheses for the
            # training winner-assignment; collapse to the selected one for eval/downstream consumers.
            out['is_mixsel'] = True
            out['sel_logits'] = sel_logits          # (B,K)
            out['sin_cos_all'] = sc                 # (B,K,6,2)  (training uses this)
            out['joint_angles_all'] = ang_full      # (B,K,7)
            _bi = torch.arange(sc.shape[0], device=sc.device)
            _sel = sel_logits.argmax(dim=-1)        # (B,)
            out['joint_angles'] = ang_full[_bi, _sel]   # (B,7) selected
            out['sin_cos'] = sc[_bi, _sel]              # (B,6,2) selected
        if self.with_rotation:
            # The rotation head consumes the FROZEN DINOv3 gfeat/kpfeat (768) regardless of
            # angle_backbone: its checkpoint was trained on DINOv3 features and is NOT retrained for
            # the P1b resnet50 angle path (only the angle head/feat are decoupled). `tokens` is the
            # frozen detector output computed above and always available. dino_frozen path unchanged.
            geo_r = keypoints_to_geo(kp2d, camera_K)
            gfeat_r = tokens.mean(dim=1)
            kpfeat_r = sample_kp_features(tokens, kp2d, self.heatmap_size)
            r = self.rot_head(geo_r, conf, gfeat_r, kpfeat_r, k_value=k_value)
            d6, tpred = r if self.with_translation else (r, None)
            out['rot6d'] = d6
            out['rot_matrix'] = rot6d_to_matrix(d6)              # (B,3,3) robot->camera
            if tpred is not None:
                out['trans'] = tpred                             # (B,3) camera-frame meters
        return out
