"""Per-joint OBSERVABILITY-AWARE ANGLE FUSION — robot-agnostic solver-path helpers.

Motivation (Baxter oracle decomposition, 2026-07-29): GT-crop cascade solver 0.50 / oracle-2D
0.52 / oracle-angle 0.94.  The whole deficit is joint-ANGLE information that is ABSENT from a
single-view keypoint projection: for a joint whose axis points ~along the viewing ray (wrist
rolls) or whose distal keypoints are occluded, wiggling the joint barely moves any observed
pixel, so the reprojection objective is flat there and the solver drifts theta to whatever the
(weak, uniform) init-anchor allows.  The learned appearance head, in contrast, DOES carry
information about those joints (it reads shading/texture, not projected geometry) — the Panda
"solver washes out the head" argument does NOT apply to an unobservable joint because there is
no 2D evidence to wash it out.

So: measure per-joint observability from the FK Jacobian and FUSE — trust the solver where the
joint is observable (w~1, recovers current behaviour -> do-no-harm on Panda/KUKA), fall back to
the head where it is not (w~0, transfers head accuracy to the unobservable tail).

Observability (per frame, per joint j):
    O_j = || d(project(FK(theta))) / d(theta_j) ||   (px per rad)
summed over the CONFIDENT keypoints only (conf-gate mask), R,t,K held fixed at the current
estimate.  No training, no per-robot constant.  Units are px/rad; because observable joints sit
in the hundreds and unobservable joints near zero, ONE global knee c (px/rad) separates them
across robots.

Two use modes (both opt-in, defaults untouched):
  * post-hoc fusion    : theta_final_j = w_j*theta_solve_j + (1-w_j)*theta_head_j,  w_j=O_j/(O_j+c)
  * solve regularizer  : per-joint anchor weight (1-w_j) on ||theta_j - theta_head_j||^2 inside the
                         solve (mathematically cleaner; the anchor already exists uniformly).
"""
import torch


def project_uv(fk, R, t, K):
    """fk (B,N,3) robot frame, R (B,3,3), t (B,3), K (B,3,3) -> uv (B,N,2) pixels."""
    cam = torch.einsum('bij,bnj->bni', R, fk) + t.unsqueeze(1)
    z = cam[..., 2].clamp(min=1e-3)
    u = cam[..., 0] / z * K[:, 0, 0:1] + K[:, 0, 2:3]
    v = cam[..., 1] / z * K[:, 1, 1:2] + K[:, 1, 2:3]
    return torch.stack([u, v], -1)


def observability(fk_fn, theta, R, t, K, conf_mask):
    """Per-joint observability O (B,A) in px/rad.

    O_j = sqrt( sum_{k in confident} (du_k/dtheta_j)^2 + (dv_k/dtheta_j)^2 ), the norm of the
    j-th column of the projected FK Jacobian restricted to confident keypoints. R,t,K are held
    fixed (detached) so this is purely "how much does joint j move the observed pixels".

    fk_fn: theta(B,A)->(B,N,3) differentiable FK. conf_mask (B,N): weight per keypoint (a
    conf-gate 0/1 mask; any non-negative weighting works). Returns O (B,A) on theta.device.
    """
    B, A = theta.shape
    Rf, tf, Kf = R.detach().float(), t.detach().float(), K.detach().float()
    th = theta.detach().float().clone().requires_grad_(True)
    uv = project_uv(fk_fn(th), Rf, tf, Kf)               # (B,N,2)
    m = conf_mask.detach().float()                        # (B,N)
    N = uv.shape[1]
    sq = torch.zeros(B, A, device=theta.device, dtype=torch.float32)
    for n in range(N):
        for c in range(2):
            g = torch.autograd.grad(uv[:, n, c].sum(), th, retain_graph=True)[0]   # (B,A)
            sq = sq + (g ** 2) * m[:, n:n + 1]
    return sq.clamp(min=0.0).sqrt()                       # (B,A) px/rad


def fusion_weights(O, c):
    """Saturating knee w = O/(O+c) in [0,1). c>0 in px/rad. Larger O -> trust solver."""
    return O / (O + float(c))


def _wrap(x):
    return torch.atan2(torch.sin(x), torch.cos(x))


def fuse_angles(theta_solve, theta_head, w):
    """theta_final = theta_head + w * wrap(theta_solve - theta_head).  Circular so a wrist roll
    that the solver drifted across +-pi still fuses toward the head on the short arc.  w=1 returns
    the solver angle exactly (bit-identical do-no-harm when every joint is observable)."""
    return _wrap(theta_head + w * _wrap(theta_solve - theta_head))


def fuse(fk_fn, theta_solve, theta_head, R, t, K, conf_mask, c):
    """Convenience: compute O at (theta_solve,R,t,K), return (theta_fused, w, O)."""
    O = observability(fk_fn, theta_solve, R, t, K, conf_mask)
    w = fusion_weights(O, c)
    return fuse_angles(theta_solve, theta_head, w), w, O
