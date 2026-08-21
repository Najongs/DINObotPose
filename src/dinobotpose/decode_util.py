"""
DARK-style distribution-aware sub-pixel heatmap decoding (Zhang et al. 2020, arXiv:1910.06278).
Pure inference-time keypoint decode fix — no training. Targets the far/small-robot 2D precision gap
(orb): Gaussian-modulate the heatmap, then Taylor-refine the hard-argmax peak using the first and
second derivatives of the log-heatmap. Reported to degrade much less at low input resolution than
plain argmax — exactly the small-target regime.
"""
import torch
import torch.nn.functional as F


def _gaussian_blur(hm, ksize=11, sigma=2.5):
    B, N, H, W = hm.shape
    ax = torch.arange(ksize, device=hm.device, dtype=hm.dtype) - (ksize - 1) / 2
    g = torch.exp(-(ax ** 2) / (2 * sigma ** 2)); g = g / g.sum()
    k = (g[:, None] * g[None, :]).view(1, 1, ksize, ksize).repeat(N, 1, 1, 1)
    return F.conv2d(hm, k, padding=ksize // 2, groups=N)


def dark_decode(heatmaps, blur=True, sigma=2.5):
    """(B,N,H,W) heatmaps -> (B,N,2) sub-pixel (x,y). Non-differentiable (inference decode)."""
    hm = heatmaps.detach()
    if blur:
        hm = _gaussian_blur(hm.clamp(min=0), sigma=sigma)
    B, N, H, W = hm.shape
    flat = hm.reshape(B, N, -1)
    idx = flat.argmax(-1)
    px = (idx % W).long().clamp(1, W - 2)
    py = (idx // W).long().clamp(1, H - 2)
    logh = (hm.clamp(min=1e-10)).log()

    def at(dy, dx):
        yy = (py + dy).clamp(0, H - 1); xx = (px + dx).clamp(0, W - 1)
        bi = torch.arange(B, device=hm.device)[:, None]
        ni = torch.arange(N, device=hm.device)[None, :]
        return logh[bi, ni, yy, xx]                         # (B,N)

    c = at(0, 0)
    Dx = (at(0, 1) - at(0, -1)) / 2
    Dy = (at(1, 0) - at(-1, 0)) / 2
    Dxx = at(0, 1) - 2 * c + at(0, -1)
    Dyy = at(1, 0) - 2 * c + at(-1, 0)
    Dxy = (at(1, 1) - at(1, -1) - at(-1, 1) + at(-1, -1)) / 4
    # offset = -Hess^-1 @ grad ; Hess=[[Dxx,Dxy],[Dxy,Dyy]]
    det = Dxx * Dyy - Dxy * Dxy
    ok = det.abs() > 1e-6
    inv00 = torch.where(ok, Dyy / det, torch.zeros_like(det))
    inv01 = torch.where(ok, -Dxy / det, torch.zeros_like(det))
    inv11 = torch.where(ok, Dxx / det, torch.zeros_like(det))
    ox = -(inv00 * Dx + inv01 * Dy)
    oy = -(inv01 * Dx + inv11 * Dy)
    # clamp the correction to <=1px (Taylor validity); reject NaN/degenerate
    ox = ox.clamp(-1, 1).nan_to_num(); oy = oy.clamp(-1, 1).nan_to_num()
    x = px.float() + ox; y = py.float() + oy
    return torch.stack([x, y], dim=-1)
