"""
Self-bbox (oracle-free) crop pipeline eval. Productionizes the train+test crop win without GT
keypoints: stage-1 (full-frame) detector runs a FIRST PASS on the full image -> detected 2D ->
square robot bbox -> roi_align crop+resize to image_size (K principal-point shift + focal scale) ->
crop pipeline (crop detector + crop angle head + crop rot-head) SECOND PASS -> kinematic solver.

Answers the honesty question: does the crop ADD gain survive a DETECTED (not GT) bbox? The crop
training used bbox jitter (margin 1.3-1.5, scale .9-1.1, center +-.1) so the head is robust to an
imperfect bbox. Mirrors dataset.py's GT-crop math but sourced from the first-pass detector.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src', 'dinobotpose'))
import argparse, glob, math, os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.ops import roi_align
from tqdm import tqdm

TRAIN = os.path.abspath(os.path.join(os.path.dirname(__file__), '../TRAIN'))
from model_angle import AnglePredictor                       # noqa
from dataset import PoseEstimationDataset                    # noqa
from solve import solve_batch                 # noqa
from refine_utils import wrapped_abs_deg, add_auc, scale_K    # reuse helpers


def build_predictor(model_name, image_size, detector_ckpt, device,
                    angle_ckpt=None, rot_ckpt=None, head_type='mlp', n_hyp=2,
                    angle_backbone='dino_frozen'):
    with_rot = rot_ckpt is not None
    m = AnglePredictor(model_name, image_size, head_type=head_type, n_hyp=n_hyp,
                       with_rotation=with_rot, with_translation=with_rot,
                       angle_backbone=angle_backbone).to(device).eval()
    sd = torch.load(detector_ckpt, map_location=device)
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    m.load_state_dict({k: v for k, v in sd.items()
                       if k in m.state_dict() and v.shape == m.state_dict()[k].shape}, strict=False)
    if angle_ckpt:
        m.angle_head.load_state_dict(torch.load(angle_ckpt, map_location=device))
        if angle_backbone == 'resnet50':
            # P1b: the trainable ResNet angle trunk is saved by train_angle.py as a sibling
            # *_angle_feat.pth (e.g. best_angle_head.pth -> best_angle_feat.pth). Load it too.
            feat_ckpt = angle_ckpt.replace('angle_head', 'angle_feat')
            if feat_ckpt == angle_ckpt or not os.path.exists(feat_ckpt):
                raise FileNotFoundError(
                    f"--angle-backbone resnet50 needs the ResNet trunk weights at '{feat_ckpt}' "
                    f"(sibling of the angle ckpt '{angle_ckpt}'), but it was not found. "
                    f"Ensure train_angle.py saved best_angle_feat.pth alongside best_angle_head.pth.")
            m.angle_feat.load_state_dict(torch.load(feat_ckpt, map_location=device))
    if rot_ckpt:
        m.rot_head.load_state_dict(torch.load(rot_ckpt, map_location=device))
    return m


def detected_bbox(kp2d, conf, img_size, margin, conf_thr):
    """Square robot bbox (B,4) x0y0x1y1 in img_size space from confident detected keypoints.
    Mirrors dataset.py: center=midpoint, side=max(dx,dy)*margin. Fallback to full frame."""
    B = kp2d.shape[0]
    boxes = kp2d.new_zeros(B, 4)
    for b in range(B):
        m = conf[b] > conf_thr
        if int(m.sum()) < 2:
            m = conf[b] > 0.0  # fallback: use whatever we have
        if int(m.sum()) < 2:
            boxes[b] = torch.tensor([0., 0., float(img_size), float(img_size)], device=kp2d.device)
            continue
        pts = kp2d[b][m]
        x0, x1 = pts[:, 0].min(), pts[:, 0].max()
        y0, y1 = pts[:, 1].min(), pts[:, 1].max()
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        side = torch.clamp(torch.max(x1 - x0, y1 - y0) * margin, min=16.0)
        boxes[b, 0] = cx - side / 2; boxes[b, 1] = cy - side / 2
        boxes[b, 2] = cx + side / 2; boxes[b, 3] = cy + side / 2
    return boxes


def project_points(kp_cam, K):
    """Project camera-frame 3D points (B,N,3) to pixel 2D (B,N,2) with intrinsics K (B,3,3)."""
    z = kp_cam[..., 2:3].clamp(min=1e-4)
    uv = kp_cam[..., :2] / z                                   # normalized image plane
    fx, fy = K[:, 0, 0].unsqueeze(1), K[:, 1, 1].unsqueeze(1)
    cx, cy = K[:, 0, 2].unsqueeze(1), K[:, 1, 2].unsqueeze(1)
    u = uv[..., 0] * fx + cx; v = uv[..., 1] * fy + cy
    return torch.stack([u, v], dim=-1)                         # (B,N,2)


def clamp_boxes(boxes, img_size, min_side=24.0):
    """Sanity-clamp boxes: replace NaN/inf, enforce a min side, keep the center on-frame. Prevents a
    degenerate bbox (side~0 or huge) from blowing up crop_K's scale -> solver divergence."""
    boxes = torch.nan_to_num(boxes, nan=0.0, posinf=float(img_size), neginf=0.0)
    cx = (boxes[:, 0] + boxes[:, 2]) / 2; cy = (boxes[:, 1] + boxes[:, 3]) / 2
    side = (boxes[:, 2] - boxes[:, 0]).clamp(min=min_side, max=float(img_size) * 2)
    cx = cx.clamp(0.0, float(img_size)); cy = cy.clamp(0.0, float(img_size))
    out = boxes.clone()
    out[:, 0] = cx - side / 2; out[:, 1] = cy - side / 2
    out[:, 2] = cx + side / 2; out[:, 3] = cy + side / 2
    return out


def crop_K(K, boxes, out_size):
    """Adjust K for crop [x0,y0]+side then resize side->out_size. scale=out_size/side."""
    Kc = K.clone()
    for b in range(K.shape[0]):
        x0, y0 = boxes[b, 0], boxes[b, 1]
        side = (boxes[b, 2] - boxes[b, 0]).clamp(min=1.0)
        s = out_size / side
        Kc[b, 0, 0] = K[b, 0, 0] * s; Kc[b, 1, 1] = K[b, 1, 1] * s
        Kc[b, 0, 2] = (K[b, 0, 2] - x0) * s; Kc[b, 1, 2] = (K[b, 1, 2] - y0) * s
    return Kc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage1-detector', required=True, help='full-frame detector for the bbox first pass')
    ap.add_argument('--stage1-angle', default=None, help='full-frame angle head (for --bbox-from-solved)')
    ap.add_argument('--stage1-rot', default=None, help='full-frame rot-head (for --bbox-from-solved)')
    ap.add_argument('--bbox-guard', action='store_true',
                    help='divergence guard: fall back to detected-kp bbox when the pass-1 solve diverged '
                         '(high reproj / off-frame projection), and clamp all bboxes. Use with --bbox-from-solved.')
    ap.add_argument('--bbox-reproj-thr', type=float, default=1e9, help='OPTIONAL extra gate: pass-1 reproj px above which the solved bbox is deemed diverged (OFF by default; geometric off-frame/NaN check is the primary guard)')
    ap.add_argument('--bbox-union', action='store_true',
                    help='UNION the solved-FK bbox with the confidently-detected-keypoint bbox. Solver '
                         'bias mildly mis-crops (the -0.04 vs GT-crop gap); detected keypoints are direct '
                         'observations, so unioning guarantees the crop never clips a confident keypoint '
                         'while still using the solved skeleton to fill the occluded base. Use with --bbox-from-solved.')
    ap.add_argument('--bbox-from-solved', action='store_true',
                    help='build the pass-1 bbox from the SOLVED full-frame pose: run the kinematic '
                         'solver on the full-frame detection, project ALL 7 FK keypoints (incl. the '
                         'occluded base that the raw detector misses) -> bbox. Training-free fix for '
                         'the keypoint-bbox failure (occluded base dropped -> mis-centered crop -> 0.55).')
    ap.add_argument('--crop-detector', required=True, help='crop-trained detector for the second pass')
    ap.add_argument('--crop-angle', required=True, help='crop-trained angle head')
    ap.add_argument('--crop-head-type', default='mlp', choices=['mlp', 'transformer', 'mlp_patch', 'mlp_mcl', 'mlp_mixsel', 'pare', 'ief'], help='architecture of the crop-angle head (must match how it was trained)')
    ap.add_argument('--crop-n-mix', type=int, default=2, help='n_hyp for mlp_mcl/mlp_mixsel crop-angle head')
    ap.add_argument('--angle-backbone', default='dino_frozen', choices=['dino_frozen', 'resnet50'],
                    help='P1b: crop-angle feature source. dino_frozen (default) = frozen DINOv3 pooled tokens. '
                         'resnet50 = separate trainable ResNet50 trunk; loads the sibling *_angle_feat.pth next to --crop-angle.')
    ap.add_argument('--rot-head', default=None, help='crop-native rot-head (R_init)')
    ap.add_argument('--head-direct', action='store_true',
                    help='ATTRIBUTION ARM (reference-style feed-forward): skip the kinematic solver; '
                         'pose = rot-head (R,t) applied to FK(theta_head). Requires a rot head that '
                         'predicts translation. Isolates the solver re-derivation contribution.')
    ap.add_argument('--val-dir', required=True)
    ap.add_argument('--model-name', default='facebook/dinov3-vitb16-pretrain-lvd1689m')
    ap.add_argument('--image-size', type=int, default=512)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--max-frames', type=int, default=1000)
    ap.add_argument('--frac-range', nargs=2, type=float, default=None,
                    help='restrict to ds.samples[LO*N:HI*N] BEFORE striding (anti-leak held-out eval; '
                         'use 0.7 1.0 to match selftrain contiguous last-30%% split)')
    ap.add_argument('--iters', type=int, default=200)
    ap.add_argument('--joint-mean-npy', default=None,
                    help='per-joint mean angles (.npy) for --wrist-fill mean')
    ap.add_argument('--conf-gate', type=float, default=0.05)
    ap.add_argument('--edge-gate', type=float, default=0.0, help='committee off-frame fix: drop (conf=0) any keypoint whose position maps within this many px of / outside the image frame -> solver ignores off-screen hallucinations and infers those joints from FK + prior. 0=off.')
    ap.add_argument('--edge-gate-oracle', action='store_true', help='ceiling probe: drop keypoints whose GT 2D is truly off-frame (perfect presence signal). Isolates the drop approach from the border heuristic.')
    ap.add_argument('--wrist-fill', default='none', choices=['none', 'gt', 'mean'], help='1악장 fill probe: when the wrist keypoints (link6/7/hand) are off-frame, DROP them and FILL the distal joints J5/J6 with GT (ceiling) or dataset-mean (dumb-prior floor). Measures how good a wrist FILL needs to be.')
    ap.add_argument('--freeze-head-theta', action='store_true', help='P0 DECOUPLE (RoboPEPP-style): freeze theta at the HEAD prediction and solve ONLY camera 6DOF R,t. Stops off-frame hallucinated keypoints from dragging the joint angles into a wrong basin (they can only affect R,t). Pair with --edge-gate to also keep them out of the R,t solve.')
    ap.add_argument('--oracle-2d', action='store_true',
                    help='AUDIT PROBE (opt-in, default off = bit-identical): inject GT 2D keypoints (mapped into the deployed crop) with conf=1 on-frame / 0 off-frame, then solve. Upper bound of the 2D-keypoint parameterization given the auto-crop -> separates 2D-detector-limited from monocular-depth-ambiguity.')
    ap.add_argument('--oracle-angle', action='store_true',
                    help='known-joint ceiling: use GT joint angles (theta) and solve only camera R,t '
                         '(theta frozen). Measures the cost of joint-angle prediction vs a known-joint upper bound.')
    ap.add_argument('--oracle-except', default='', help='with --oracle-angle: comma-separated joint indices (0=J1..5=J6) to KEEP the PREDICTED value instead of GT. e.g. "4" = oracle all joints but predict J5 (achievable-ceiling: J5 is a write-off observability dead-end).')
    ap.add_argument('--oracle-angle-noise-mae', type=float, default=0.0, help='with --oracle-angle: inject controlled Gaussian noise on GT theta, per-batch renormalized so the realized wrapped-abs MAE (deg) equals this target. Builds the freeze-6DOF ADD-AUC vs angle-MAE curve. 0 = pure oracle.')
    ap.add_argument('--oracle-noise-seed', type=int, default=0, help='seed for --oracle-angle-noise-mae (average curve over a few seeds).')
    ap.add_argument('--margin', type=float, default=1.5)
    ap.add_argument('--bbox-conf', type=float, default=0.1, help='conf threshold for bbox keypoints')
    ap.add_argument('--bbox-refine-iters', type=int, default=0,
                    help='iterative bbox refinement: after the stage1 full-frame bbox, re-detect with '
                         'the CROP detector on the zoomed crop (robot bigger -> sharper keypoints -> '
                         'tighter bbox) and re-crop, N times. RoboPEPP-style coarse-to-fine.')
    ap.add_argument('--oracle-bbox', action='store_true',
                    help='DEBUG: build the bbox from GT keypoints (not the detector) to isolate the '
                         'roi_align/K crop pipeline from detected-bbox error; should reproduce the '
                         'dataset.py GT-crop ADD (~0.768 realsense) if the crop math is correct')
    ap.add_argument('--dump-npz', default=None, help='save (fid, theta, kp_cam, gt3d, found) for cross-env render-compare')
    ap.add_argument('--occlude-ratio', type=float, default=0.0,
                    help='RoboPEPP-protocol synthetic occlusion: paint deterministic black rect/circle masks covering this fraction of the GT-keypoint RoI (seeded per frame+ratio so downstream RC sees identical occluders)')
    ap.add_argument('--cov-pnp', action='store_true',
                    help='anisotropic heatmap-covariance (Mahalanobis) weighting in the final solve — continuous upgrade of the scalar conf weighting for occluded/diffuse keypoints')
    ap.add_argument('--prior-adaptive', type=float, default=0.0,
                    help='occlusion-adaptive configuration prior weight (masked-state prior, analytic per-joint Gaussian; scales with fraction of low-conf keypoints)')
    ap.add_argument('--dark-decode', action='store_true',
                    help='DARK sub-pixel heatmap decode (Taylor refine of the argmax peak) instead of soft-argmax — targets far/small-robot 2D precision (orb)')
    ap.add_argument('--dark-sigma', type=float, default=2.5, help='DARK Gaussian modulation sigma (match training heatmap sigma)')
    ap.add_argument('--ms-local', type=int, default=0,
                    help='head-seeded local multi-start for the theta solve: N candidates = angle-head init + Gaussian(ms-sigma) perturbations, keep min-reproj. Fixes the monocular basin-finding for robots whose angle head cannot init within the solver basin (real-data robots without a synth angle prior).')
    ap.add_argument('--ms-sigma', type=float, default=45.0, help='local multi-start perturbation std in degrees (around the angle-head init)')
    ap.add_argument('--border-margin', type=float, default=0.0,
                    help='BORDER gate (0=off, default): drop keypoints decoded within this many px of the crop border from the minimal-set PnP init AND the refine weights. Targets the off-frame trigger (border-pinned peaks are SHARP, so --conf-gate structurally cannot reject them).')
    ap.add_argument('--resolve-reproj', type=float, default=0.0,
                    help='MULTI-START re-solve (0=off, default): frames whose solved reprojection exceeds this many px are re-solved from alternative PnP inits (7/6/5-point sets, no learned R_init override so candidates can reach different rotation basins). A candidate is adopted ONLY if it lowers the reprojection residual.')
    ap.add_argument('--kp-jitter', type=float, default=0.0,
                    help='inject Gaussian 2D-localization noise (px std) into the decoded keypoints before the solver — PnP/solver robustness sweep (G1). cov_inv is kept from the clean heatmap, so this probes whether anisotropic whitening absorbs added noise.')
    ap.add_argument('--robust-scale', type=float, default=0.0,
                    help='adaptive robust-loss knee = robust_scale x per-frame median residual '
                         '(0=off -> fixed 8px Geman-McClure, bit-identical baseline)')
    ap.add_argument('--shed-k', type=float, default=0.0,
                    help='soft floor: drop keypoints with residual > shed_k x per-frame median and '
                         're-solve below the min_kp floor, adopting only if reprojection on the '
                         'retained keypoints improves (0=off, do-no-harm)')
    ap.add_argument('--z-bound', type=str, default='',
                    help='physical-plausibility depth bound "lo,hi" in meters (e.g. "0.3,5.0", off by '
                         'default). Frames whose median solved skeleton depth lands outside are re-solved '
                         'at a plausible depth and adopted on physical validity (targets the z-collapse '
                         'failure mode reproj-keyed guards structurally miss). In-bound frames untouched.')
    ap.add_argument('--z-target', type=float, default=0.0,
                    help='plausible re-init depth (m) for the --z-bound re-solve. >0 = fixed scene depth '
                         '(e.g. the split median SOLVED depth over healthy frames, a GT-free prior); '
                         '<=0 = per-frame clamp of the collapsed depth into the bound.')
    ap.add_argument('--z-anchor-w', type=float, default=0.5,
                    help='depth-anchor weight holding the --z-bound re-solve at --z-target (default 0.5)')
    ap.add_argument('--fids-file', type=str, default='',
                    help='restrict the eval to the frame names (one per line) in this file. Deterministic '
                         'frame subset for targeted re-solves; applied AFTER --frac-range, BEFORE striding.')
    ap.add_argument('--per-frame-out', type=str, default='',
                    help='write per-frame "fid<TAB>ADD(m)<TAB>median_depth(m)" rows to this file '
                         '(for targeted-recovery / do-no-harm analysis).')
    ap.add_argument('--seed', type=int, default=-1,
                    help='opt-in determinism (default -1 = off, deployed behavior). >=0 seeds '
                         'numpy/torch and re-seeds cv2 RANSAC per PnP call so a z-bound ON vs OFF '
                         'comparison is bitwise-identical on in-bound frames (do-no-harm proof).')
    args = ap.parse_args()
    if args.seed >= 0:
        # NOTE: deliberately NOT calling torch.use_deterministic_algorithms — it forces
        # pathologically slow CUDA kernels (~100x). The z-bound do-no-harm proof only needs the
        # cv2 RANSAC RNG pinned per PnP call (below); residual torch CUDA float noise is <1e-4 m.
        import cv2 as _cv2
        np.random.seed(args.seed); torch.manual_seed(args.seed); _cv2.setRNGSeed(args.seed)
        from solve import set_pnp_seed
        set_pnp_seed(args.seed)
    _zb = None
    if args.z_bound.strip():
        _lo, _hi = (float(x) for x in args.z_bound.split(','))
        _zb = (_lo, _hi)
    # NOTE dump 필드는 ADDITIVE only — 기존 소비 코드(rc_dumps_oas 등)가 읽는 키를 절대 제거/개명하지 말 것.
    # head_theta/kp2d/gtkp2d 는 "등가 입력오차 3.65px가 검출기 2D 탓인지 각도head 탓인지" 판별용
    # (2026-07-22 mediocre-band 분석). solve 이전 값이어야 의미가 있다.
    # ⚠️ 좌표계 주의: kp2d 는 CROP-IS 공간, gtkp2d(=batch['keypoints']) 는 FULL-FRAME IS 공간이라
    # 직접 비교하면 안 된다. kp2d_full 이 crop->full-frame 으로 매핑된 검출 2D (line ~298 과 동일 식).
    _DUMP = {'fid': [], 'theta': [], 'kp_cam': [], 'gt3d': [], 'found': [], 'feat': [], 'reproj': [], 'conf': [], 'gtoff': [],
             'head_theta': [], 'kp2d': [], 'gtkp2d': [], 'kp2d_full': [], 'boxes': []} if args.dump_npz else None

    device = torch.device('cuda'); assert torch.cuda.is_available()
    IS = args.image_size
    # bbox pass: plain detector, OR a full predictor (detector+angle+rot) when solving for the bbox
    det1 = build_predictor(args.model_name, IS, args.stage1_detector, device,
                           angle_ckpt=args.stage1_angle if args.bbox_from_solved else None,
                           rot_ckpt=args.stage1_rot if args.bbox_from_solved else None)     # bbox pass
    cropm = build_predictor(args.model_name, IS, args.crop_detector, device,
                            angle_ckpt=args.crop_angle, rot_ckpt=args.rot_head,
                            head_type=args.crop_head_type, n_hyp=args.crop_n_mix,
                            angle_backbone=args.angle_backbone)                              # crop pass
    print(f"stage1: {args.stage1_detector}\ncrop det: {args.crop_detector}\ncrop angle: {args.crop_angle}\nrot: {args.rot_head}\nval: {args.val_dir}")

    # full-frame dataset (NO crop) -> we crop ourselves from detected bbox
    # SigLIP/SigLIP2 expect mean=std=0.5 ([-1,1]); DINOv3 uses ImageNet stats. Must match the backbone
    # the heads/detector were trained under (train_heatmap/angle/rotation set the same).
    if "siglip" in args.model_name:
        _nm = _ns = [0.5, 0.5, 0.5]
        print("==> SigLIP backbone detected: using mean=std=0.5 normalization")
    else:
        _nm = _ns = None
    ds = PoseEstimationDataset(args.val_dir,
                               keypoint_names=['link0', 'link2', 'link3', 'link4', 'link6', 'link7', 'hand'],
                               image_size=(IS, IS), heatmap_size=(IS, IS),
                               augment=False, include_angles=True, sigma=2.5, crop_to_robot=False,
                               norm_mean=_nm, norm_std=_ns)
    if args.frac_range:
        lo, hi = args.frac_range; Nall = len(ds.samples)
        ds.samples = ds.samples[int(lo * Nall):int(hi * Nall)]
        print(f"frac-range {lo}-{hi}: {Nall} -> {len(ds.samples)} held-out frames")
    if args.fids_file:
        with open(args.fids_file) as _f:
            _keep = set(l.strip() for l in _f if l.strip())
        Nall = len(ds.samples)
        ds.samples = [s for s in ds.samples if s['name'] in _keep]
        print(f"fids-file {args.fids_file}: {Nall} -> {len(ds.samples)} selected frames "
              f"({len(_keep)} names requested)")
    if args.max_frames and args.max_frames < len(ds):
        stride = max(1, len(ds.samples) // args.max_frames)
        ds.samples = ds.samples[::stride][:args.max_frames]
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    raw_err = torch.zeros(6); ref_err = torch.zeros(6); n = 0
    adds = []
    for batch in tqdm(loader, desc="selfbbox eval"):
        img = batch['image'].to(device)
        if args.occlude_ratio > 0:
            from occl_util import paste_occluders_batch_
            paste_occluders_batch_(img, batch['keypoints'].numpy(), batch['valid_mask'].numpy(),
                                   args.occlude_ratio, batch['name'])
        gt = batch['angles'].to(device)[:, :6]
        gt3d = batch['keypoints_3d'].to(device)
        K = scale_K(batch['camera_K'], batch['original_size'], IS).to(device)
        bidx = torch.arange(img.shape[0], device=device).view(-1, 1).float()
        with torch.no_grad():
            # PASS 1: full-frame detector -> bbox  (or GT keypoints if --oracle-bbox)
            if args.oracle_bbox:
                gkp = batch['keypoints'].to(device).float()        # (B,7,2) in IS space
                gconf = batch['valid_mask'].to(device).float()     # 1 where keypoint valid
                boxes = detected_bbox(gkp, gconf, IS, args.margin, 0.5)
            elif args.bbox_from_solved:
                # PASS 1: full-frame detect + SOLVE -> project ALL 7 FK keypoints -> bbox.
                # The solved skeleton fills in the occluded base (link0) the raw detector drops,
                # since base position is fixed by the camera pose (R,t) even when base-yaw is ambiguous.
                o1 = det1(img, K)
                R1 = o1.get('rot_matrix') if args.stage1_rot else None
                # solve_batch optimizes via internal autograd (loss.backward) -> must run with grad
                # enabled even though we're inside the no_grad eval block.
                with torch.enable_grad():
                    _, kp_cam1, reproj1 = solve_batch(o1['keypoints_2d'], o1['confidence'], K, fix_joint7=True,
                                                      iters=args.iters, lr=2e-2, img_size=IS, device=device,
                                                      prior_w=0.0, theta_init=o1['joint_angles'],
                                                      conf_gate=args.conf_gate, R_init=R1)
                kp_cam1 = kp_cam1.detach(); reproj1 = reproj1.detach()
                uv = project_points(kp_cam1, K)                      # (B,7,2) all points, no conf drop
                if args.bbox_union:
                    # UNION: solved 7pts (always in, fills occluded base) + confidently-detected kp
                    # (direct observations, gated by conf). Counters solver-bias mis-cropping: a slightly
                    # biased solve no longer shrinks/shifts the crop off a keypoint the detector saw clearly.
                    upts = torch.cat([uv, o1['keypoints_2d']], dim=1)                       # (B,14,2)
                    uconf = torch.cat([torch.ones(uv.shape[:2], device=device),
                                       o1['confidence']], dim=1)                            # solved=1, det=conf
                    boxes = detected_bbox(upts, uconf, IS, args.margin, args.bbox_conf)
                else:
                    boxes = detected_bbox(uv, torch.ones(uv.shape[:2], device=device), IS, args.margin, 0.0)
                if args.bbox_guard:
                    # DIVERGENCE GUARD: a diverged pass-1 solve projects to a GARBAGE skeleton (points
                    # off-frame / NaN / absurd spread) -> garbage bbox -> blown-up crop. Detect that
                    # GEOMETRICALLY (NOT via reproj: realsense full-frame solves have naturally high
                    # reproj on foreshortened frames, so a reproj gate nukes good frames too) and fall
                    # back to the detected-kp bbox for those frames; clamp every bbox for safety.
                    det_boxes = detected_bbox(o1['keypoints_2d'], o1['confidence'], IS, args.margin, args.bbox_conf)
                    off = ((uv < -IS) | (uv > 2 * IS)).any(dim=2).any(dim=1)    # any point wildly off-frame
                    span = (uv.amax(dim=1) - uv.amin(dim=1)).amax(dim=1)        # bbox side in px
                    huge = span > 3.0 * IS                                      # absurd spread
                    bad = off | huge | torch.isnan(uv).any(dim=2).any(dim=1)
                    if args.bbox_reproj_thr < 1e6:                              # optional extra reproj gate (off by default)
                        bad = bad | (reproj1 > args.bbox_reproj_thr)
                    boxes = torch.where(bad.unsqueeze(1), det_boxes, boxes)
                    n_fb = int(bad.sum())
                    if n_fb:
                        globals().setdefault('_GUARD_FB', [0])[0] += n_fb
                boxes = clamp_boxes(boxes, IS)
            else:
                o1 = det1(img, K)
                boxes = detected_bbox(o1['keypoints_2d'], o1['confidence'], IS, args.margin, args.bbox_conf)
            # ITERATIVE coarse-to-fine: re-detect on the crop with the crop detector -> tighter bbox
            for _ in range(args.bbox_refine_iters):
                rois = torch.cat([bidx, boxes], dim=1)
                ci = roi_align(img, rois, output_size=(IS, IS), spatial_scale=1.0, aligned=True)
                oc = cropm(ci, crop_K(K, boxes, float(IS)))
                kpc = oc['keypoints_2d']                            # (B,7,2) in crop-IS space
                # map crop-space kp -> full-frame IS space, recompute bbox there
                side = (boxes[:, 2] - boxes[:, 0]).clamp(min=1.0).view(-1, 1)
                kpf = kpc * (side / IS).unsqueeze(-1) + boxes[:, :2].unsqueeze(1)
                boxes = detected_bbox(kpf, oc['confidence'], IS, args.margin, args.bbox_conf)
            rois = torch.cat([bidx, boxes], dim=1)
            crop_img = roi_align(img, rois, output_size=(IS, IS), spatial_scale=1.0, aligned=True)
            Kc = crop_K(K, boxes, float(IS))
            # FINAL PASS: crop pipeline on the refined crop
            o2 = cropm(crop_img, Kc)
        init_ang = o2['joint_angles']; kp2d = o2['keypoints_2d']; conf = o2['confidence']
        _head_theta = init_ang.detach().clone()   # solve 이전 head θ (oracle/wrist-fill 덮어쓰기 전 스냅샷)
        if args.oracle_2d:
            # AUDIT: replace detected crop-2D with GT 2D mapped into the SAME (auto) crop. conf=1 on-frame,
            # 0 off-frame. Isolates 2D-keypoint quality from the pose solve (detector vs depth-ambiguity).
            _gkp = batch['keypoints'].to(device).float()                       # (B,7,2) full-frame IS
            _side = (boxes[:, 2] - boxes[:, 0]).clamp(min=1.0).view(-1, 1, 1)
            kp2d = (_gkp - boxes[:, None, :2]) * (float(IS) / _side)           # -> crop-IS space
            _onf = ((_gkp[..., 0] >= 0) & (_gkp[..., 0] <= IS) &
                    (_gkp[..., 1] >= 0) & (_gkp[..., 1] <= IS))                # (B,7) GT on-frame
            conf = _onf.float()
        if args.dark_decode and not args.oracle_2d:
            from decode_util import dark_decode
            kp2d = dark_decode(o2['heatmaps_2d'], sigma=args.dark_sigma)   # sub-pixel re-decode
        if args.kp_jitter > 0:
            # G1: additive 2D-localization noise (px std), deterministic per batch for reproducibility
            _jg = torch.Generator(device=device).manual_seed(1234 + int(kp2d.shape[0]))
            kp2d = kp2d + torch.randn(kp2d.shape, generator=_jg, device=device, dtype=kp2d.dtype) * args.kp_jitter
        if args.edge_gate_oracle:
            # ORACLE presence: drop keypoints whose GT 2D is truly outside the frame (perfect presence
            # signal). Measures the CEILING of the drop-off-frame approach, isolating it from the
            # blunt border heuristic. If the tail recovers here, a learned presence head is the fix.
            _gkp = batch['keypoints'].to(device).float()                          # (B,7,2) GT 2D, IS frame
            _off = ((_gkp[..., 0] < 0) | (_gkp[..., 0] > IS) |
                    (_gkp[..., 1] < 0) | (_gkp[..., 1] > IS))
            conf = conf.clone(); conf[_off] = 0.0
            globals().setdefault('_EDGE_DROP', [0, 0]); _g = globals()['_EDGE_DROP']
            _g[0] += int(_off.sum()); _g[1] += _off.shape[0]
        if args.edge_gate > 0:
            # OFF-FRAME keypoint rejection (committee fix): a keypoint whose position maps OUTSIDE the
            # original image frame is an off-screen hallucination (the detector confidently clamps it to
            # the edge). Confidence can't catch it (it's a sharp peak); the frame boundary is a different
            # axis. Map crop-space kp2d -> frame coords via the crop box, zero conf on out-of-frame points
            # so the solver drops them and infers those joints from FK + the visible chain (+ prior).
            _side = (boxes[:, 2] - boxes[:, 0]).clamp(min=1.0)                       # (B,)
            _frame_kp = boxes[:, None, :2] + kp2d * (_side / IS).view(-1, 1, 1)      # (B,7,2) frame coords
            _m = args.edge_gate
            _off = ((_frame_kp[..., 0] < _m) | (_frame_kp[..., 0] > IS - _m) |
                    (_frame_kp[..., 1] < _m) | (_frame_kp[..., 1] > IS - _m))        # (B,7)
            conf = conf.clone(); conf[_off] = 0.0
            globals().setdefault('_EDGE_DROP', [0, 0]); _g = globals()['_EDGE_DROP']
            _g[0] += int(_off.sum()); _g[1] += _off.shape[0]                          # total drops, frames
        R_init = o2.get('rot_matrix') if args.rot_head else None
        cov_inv = None
        if args.cov_pnp and not args.oracle_2d:
            from solve import heatmap_cov_inv
            cov_inv = heatmap_cov_inv(o2['heatmaps_2d'], kp2d)
        if args.oracle_angle:
            # known-joint upper bound: replace predicted theta with GT, freeze it, solve only R,t.
            # --oracle-except keeps the PREDICTED value for listed joints (e.g. J5=idx4 write-off) ->
            # measures the ACHIEVABLE ceiling when the unobservable joints stay predicted.
            pred_ang = init_ang.clone()
            init_ang = init_ang.clone()
            init_ang[:, :6] = gt
            if args.oracle_angle_noise_mae > 0:
                # controlled angle-MAE curve: GT + Gaussian noise, per-batch renormalized so mean|noise|
                # (rad) = target(deg) -> realized wrapped-abs MAE == target. Deterministic per batch.
                _bc = globals().setdefault('_NOISE_BC', [0])
                _ng = torch.Generator(device=device).manual_seed(args.oracle_noise_seed * 99991 + _bc[0]); _bc[0] += 1
                _raw = torch.randn(init_ang[:, :6].shape, generator=_ng, device=device, dtype=init_ang.dtype)
                _raw = _raw / _raw.abs().mean().clamp(min=1e-6)
                init_ang[:, :6] = gt + _raw * math.radians(args.oracle_angle_noise_mae)
            for _j in [int(x) for x in args.oracle_except.split(',') if x.strip() != '']:
                init_ang[:, _j] = pred_ang[:, _j]
        if args.wrist_fill != 'none':
            # 1악장: drop off-frame keypoints AND fill distal joints J5/J6 (idx4,5) for wrist-off-frame
            # frames. gt=ceiling (perfect fill), mean=dumb-prior floor. Dropped kp -> J5/J6 stay at fill.
            _gkp = batch['keypoints'].to(device).float()
            _offkp = ((_gkp[..., 0] < 0) | (_gkp[..., 0] >= IS) | (_gkp[..., 1] < 0) | (_gkp[..., 1] >= IS))  # (B,7)
            conf = conf.clone(); conf[_offkp] = 0.0
            _wristoff = _offkp[:, 4:7].any(dim=1)                          # (B,)
            init_ang = init_ang.clone()
            if args.wrist_fill == 'gt':
                for _j in (4, 5):
                    init_ang[_wristoff, _j] = gt[_wristoff, _j]
            else:
                # dataset joint mean, computed once from the training split (see --joint-mean-npy)
                if args.joint_mean_npy is None:
                    raise SystemExit('--wrist-fill mean requires --joint-mean-npy <file.npy>')
                _mn = torch.tensor(np.load(args.joint_mean_npy),
                                   device=device, dtype=init_ang.dtype)
                for _j in (4, 5):
                    init_ang[_wristoff, _j] = _mn[_j]
        if args.head_direct:
            # reference-style arm: FK(theta_head) posed by the rot head's (R,t); no solve.
            from solve import panda_forward_kinematics
            assert 'trans' in o2, '--head-direct requires a rot head trained with translation'
            _Rh = o2['rot_matrix'].to(device); _th_ = o2['trans'].to(device)
            _fkr = panda_forward_kinematics(init_ang)                      # (B,N,3) robot frame
            refined = init_ang
            kp_cam = torch.einsum('bij,bnj->bni', _Rh, _fkr) + _th_.unsqueeze(1)
            reproj2 = torch.zeros(init_ang.shape[0], device=device)
        elif args.ms_local > 0:
            _sig = math.radians(args.ms_sigma)
            _gen = torch.Generator(device=device).manual_seed(0)
            refined = kp_cam = reproj2 = _best = None
            for _s in range(args.ms_local):
                _ti = init_ang if _s == 0 else init_ang + torch.randn(init_ang.shape, generator=_gen, device=device, dtype=init_ang.dtype) * _sig
                _th, _kc, _rp = solve_batch(kp2d, conf, Kc, fix_joint7=True, iters=args.iters,
                                            lr=2e-2, img_size=IS, device=device, prior_w=0.0,
                                            theta_init=_ti, conf_gate=args.conf_gate, R_init=R_init,
                                            cov_inv=cov_inv, prior_adaptive=args.prior_adaptive,
                                            robust_scale=args.robust_scale, shed_k=args.shed_k,
                                            z_bound=_zb, z_target=args.z_target,
                                            z_anchor_w=args.z_anchor_w)
                if _best is None:
                    _best, refined, kp_cam, reproj2 = _rp, _th, _kc, _rp
                else:
                    _b = _rp < _best
                    _best = torch.where(_b, _rp, _best)
                    refined = torch.where(_b.unsqueeze(1), _th, refined)
                    kp_cam = torch.where(_b.unsqueeze(1).unsqueeze(2), _kc, kp_cam)
                    reproj2 = _best
        else:
            globals().setdefault('_RESOLVE', {})
            globals().setdefault('_SHED', {})
            globals().setdefault('_ZBOUND', {})
            refined, kp_cam, reproj2 = solve_batch(kp2d, conf, Kc, fix_joint7=True, iters=args.iters,
                                             lr=2e-2, img_size=IS, device=device, prior_w=0.0,
                                             theta_init=init_ang, conf_gate=args.conf_gate, R_init=R_init,
                                             cov_inv=cov_inv, prior_adaptive=args.prior_adaptive,
                                             freeze_theta=(args.oracle_angle or args.freeze_head_theta),
                                             border_margin=args.border_margin,
                                             resolve_reproj_thr=args.resolve_reproj,
                                             _resolve_stats=globals()['_RESOLVE'],
                                             robust_scale=args.robust_scale, shed_k=args.shed_k,
                                             _shed_stats=globals()['_SHED'],
                                             z_bound=_zb, z_target=args.z_target,
                                             z_anchor_w=args.z_anchor_w,
                                             _zbound_stats=globals()['_ZBOUND'])
        raw_err += wrapped_abs_deg(init_ang[:, :6], gt).sum(0).cpu()
        ref_err += wrapped_abs_deg(refined[:, :6], gt).sum(0).cpu()
        valid = (gt3d.abs().sum(-1) > 0)
        per_j = (kp_cam - gt3d).norm(dim=-1)
        if _DUMP is not None:
            names = batch['name']; th = refined.detach().cpu().numpy(); kc = kp_cam.detach().cpu().numpy()
            g3 = gt3d.detach().cpu().numpy(); fv = valid.detach().cpu().numpy()
            ft = o2['global_feat'].detach().cpu().numpy(); rp = reproj2.detach().cpu().numpy()
            # presence diagnostics: per-kp heatmap confidence + whether GT 2D is truly off-frame.
            # dataset.py:609-611 gives off-frame keypoints an ALL-ZERO heatmap target (and
            # train_heatmap.py applies no valid mask), so conf should already encode presence.
            _cf = conf.detach().cpu().numpy()
            _gk = batch['keypoints'].to(device).float()
            _go = (((_gk[..., 0] < 0) | (_gk[..., 0] >= IS) |
                    (_gk[..., 1] < 0) | (_gk[..., 1] >= IS)).detach().cpu().numpy())
            # 검출 2D(솔버에 실제로 들어간 값, DARK 재디코드 후) + 동일 IS 프레임의 GT 2D
            _ht = _head_theta.cpu().numpy(); _k2 = kp2d.detach().cpu().numpy(); _gk2 = _gk.detach().cpu().numpy()
            # crop-IS -> full-frame IS (line ~298 과 동일). --crop 경로가 아니면 boxes 미정의이므로 항등.
            if 'boxes' in dir() or 'boxes' in locals():
                _side = (boxes[:, 2] - boxes[:, 0]).clamp(min=1.0).view(-1, 1)
                _k2f = (kp2d * (_side / IS).unsqueeze(-1) + boxes[:, :2].unsqueeze(1)).detach().cpu().numpy()
                _bx = boxes.detach().cpu().numpy()
            else:
                _k2f = _k2; _bx = np.zeros((_k2.shape[0], 4), dtype=np.float32)
            for b in range(img.shape[0]):
                _DUMP['fid'].append(names[b]); _DUMP['theta'].append(th[b]); _DUMP['kp_cam'].append(kc[b])
                _DUMP['gt3d'].append(g3[b]); _DUMP['found'].append(fv[b])
                _DUMP['feat'].append(ft[b]); _DUMP['reproj'].append(float(rp[b]))
                _DUMP['conf'].append(_cf[b]); _DUMP['gtoff'].append(_go[b])
                _DUMP['head_theta'].append(_ht[b]); _DUMP['kp2d'].append(_k2[b]); _DUMP['gtkp2d'].append(_gk2[b])
                _DUMP['kp2d_full'].append(_k2f[b]); _DUMP['boxes'].append(_bx[b])
        for b in range(img.shape[0]):
            if valid[b].any():
                _a = float(per_j[b][valid[b]].mean().item())
                adds.append(_a)
                globals().setdefault('_PF', []).append((str(batch['name'][b]), _a,
                                                         float(kp_cam[b][valid[b]][:, 2].median().item())))
        n += img.shape[0]

    if _DUMP is not None:
        import numpy as _np
        _np.savez(args.dump_npz, fid=_np.array(_DUMP['fid']), theta=_np.array(_DUMP['theta']),
                  kp_cam=_np.array(_DUMP['kp_cam']), gt3d=_np.array(_DUMP['gt3d']), found=_np.array(_DUMP['found']),
                  feat=_np.array(_DUMP['feat']), reproj=_np.array(_DUMP['reproj']),
                  conf=_np.array(_DUMP['conf']), gtoff=_np.array(_DUMP['gtoff']),
                  head_theta=_np.array(_DUMP['head_theta']), kp2d=_np.array(_DUMP['kp2d']),
                  gtkp2d=_np.array(_DUMP['gtkp2d']), kp2d_full=_np.array(_DUMP['kp2d_full']),
                  boxes=_np.array(_DUMP['boxes']))
        print(f"[dump] {len(_DUMP['fid'])} frames -> {args.dump_npz}", flush=True)
    raw = (raw_err / n).numpy(); ref = (ref_err / n).numpy(); adds = np.array(adds)
    print(f"\n{'='*54}\n  SELF-BBOX CROP  ({n} frames)  {os.path.basename(args.val_dir)}\n{'='*54}")
    print(f"  {'joint':<6}{'raw MLP':>10}{'refined':>10}{'delta':>9}")
    for j in range(6):
        print(f"  J{j:<5}{raw[j]:>10.2f}{ref[j]:>10.2f}{ref[j]-raw[j]:>+9.2f}")
    print('-'*54)
    if len(adds):
        print(f"  [Pose] ADD-AUC@100mm: {add_auc(adds):.4f} | mean ADD {adds.mean()*1000:.1f}mm | "
              f"median {np.median(adds)*1000:.1f}mm ({len(adds)} frames)")
        print(f"  [Tail] >100mm: {100*(adds>0.1).mean():.1f}%  | p99 {np.percentile(adds,99)*1000:.0f}mm "
              f"| robust_scale={args.robust_scale} shed_k={args.shed_k}")
        _sh = globals().get('_SHED', {})
        if _sh.get('flagged', 0):
            print(f"  [Shed] flagged {_sh['flagged']}/{_sh.get('total',0)} frames, adopted {_sh.get('adopted',0)}")
    _zbs = globals().get('_ZBOUND', {})
    if _zb is not None:
        print(f"  [z-bound {_zb}] z_target={args.z_target} anchor_w={args.z_anchor_w} | "
              f"flagged {_zbs.get('flagged',0)}/{_zbs.get('total',0)} frames, "
              f"recovered(in-bounds) {_zbs.get('recovered',0)}")
    _pf = globals().get('_PF', [])
    if _pf:
        _pfo = getattr(args, 'per_frame_out', '') or ''
        if _pfo:
            with open(_pfo, 'w') as _f:
                for _fid, _a, _z in _pf:
                    _f.write(f"{_fid}\t{_a:.6f}\t{_z:.6f}\n")
            print(f"  [per-frame] {len(_pf)} rows -> {_pfo}")
    if args.bbox_guard:
        print(f"  [guard] bbox fell back to detected on {globals().get('_GUARD_FB', [0])[0]} frames")
        if args.edge_gate > 0:
            _ed = globals().get('_EDGE_DROP', [0, 1])
            print(f"  [edge-gate {args.edge_gate}px] dropped {_ed[0]} keypoints over {_ed[1]} frames = {_ed[0]/max(_ed[1],1):.2f}/frame")
    if args.resolve_reproj > 0:
        _rs = globals().get('_RESOLVE', {})
        _tot = max(_rs.get('total', 0), 1)
        print(f"  [re-solve >{args.resolve_reproj}px] flagged {_rs.get('flagged',0)}/{_tot} "
              f"({100*_rs.get('flagged',0)/_tot:.1f}%) | adopted {_rs.get('adopted',0)} "
              f"({100*_rs.get('adopted',0)/_tot:.1f}% of frames)")
    print('='*54)


if __name__ == '__main__':
    main()
