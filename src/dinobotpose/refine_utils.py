"""
Headline pipeline test: MLP angle predictor -> kinematic refinement.

Runs the MLP AnglePredictor to get (a) joint angles (init) and (b) detected 2D keypoints +
confidence, then refines the angles with the Stage-2 kinematic solver (solve_batch, theta_init
mode). Reports raw-MLP vs refined per-joint + mean angle MAE (deg). Stage-2 showed refinement
cuts a good init ~8°->5.3° on oracle 2D; this measures the real gain on detected keypoints.
"""
import argparse, glob, math, os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

TRAIN = os.path.abspath(os.path.join(os.path.dirname(__file__), '../TRAIN'))
sys.path.append(TRAIN); sys.path.append(os.path.dirname(__file__))
from model_angle import AnglePredictor, to_bearings  # noqa
from dataset import PoseEstimationDataset
from solve import solve_batch, solve_batch_heatmap  # kinematic refiners
from model_v4 import panda_forward_kinematics
import obs_fusion   # per-joint observability-aware angle fusion (opt-in)


def wrapped_abs_deg(pred, gt):
    return torch.atan2(torch.sin(pred - gt), torch.cos(pred - gt)).abs() * 180.0 / math.pi


def add_auc(adds_m, thr=0.1):
    """RoboPEPP-style ADD AUC@thr (m). adds_m: 1D np array of per-frame ADD (m)."""
    if len(adds_m) == 0:
        return 0.0
    d = 1e-5
    ts = np.arange(0.0, thr, d)
    counts = (adds_m[None, :] <= ts[:, None]).sum(1) / float(len(adds_m))
    return float(np.trapz(counts, dx=d) / thr)


def scale_K(camera_K, original_size, hm):
    K = camera_K.clone().float()
    for b in range(K.shape[0]):
        ow, oh = float(original_size[b][0]), float(original_size[b][1])
        sx, sy = hm / ow, hm / oh
        K[b, 0, 0] *= sx; K[b, 1, 1] *= sy
        K[b, 0, 2] *= sx; K[b, 1, 2] *= sy
    return K


def geometric_K(val_dir, camera_K, original_size, S):
    """TRUE metric intrinsics for the SOLVER / any metric geometry, at `S` scale.

    Whether frame JSONs carry `meta.K` is TREE-DEPENDENT, which is why the identity-K solver
    bug stayed hidden for so long:
      - datasets/synthetic/{kuka,baxter}_*  : NO meta.K -> PoseEstimationDataset falls back to
        eye(3). Feeding that to the solver puts fx=1 where the truth is ~555 (320x off), and
        the classic focal/depth ambiguity collapses solved depth.
      - Converted_dataset/DREAM_to_DREAM{,_syn} (Panda): meta.K IS present and real
        (synth 320, rs/kinect/orb 615.5, azure 399.7) -> must NEVER be overwritten.
    So this reconstructs ONLY when the input is the identity fallback, and otherwise scales and
    returns the dataset's own K untouched. Safe to call on every path.

    The eye(3) fallback is LOAD-BEARING for the MODEL (checkpoints learned their bearing
    features from fx=fy=1): dataset K -> model, geometric_K -> solver.
    The crop block shifts the principal point in place (camera_K[0,2] -= bx0), so on top of
    eye(3) the returned K[0,2], K[1,2] ARE exactly (-bx0, -by0); combined with
    _camera_settings.json this rebuilds the true crop intrinsics.
    Reconstruction verified to <0.09 px reprojection over 60 frames (Eval/iiwa7_rc_eval.py).

    ASSUMES one camera per directory: `_camera_settings.json` is read once and applied to the
    whole batch. Unsafe for a directory mixing several cameras/intrinsics.
    """
    import json
    is_identity = (abs(float(camera_K[0, 0, 0]) - 1.0) < 1e-6 and
                   abs(float(camera_K[0, 1, 1]) - 1.0) < 1e-6)
    if not is_identity:
        return scale_K(camera_K, original_size, S)   # real meta.K (e.g. Panda) — pass through
    p = os.path.join(val_dir, '_camera_settings.json')
    if not os.path.exists(p):
        raise FileNotFoundError(
            f'dataset camera_K is the eye(3) fallback but {p} is missing — cannot recover true '
            f'intrinsics for the solver')
    it = json.load(open(p))['camera_settings'][0]['intrinsic_settings']
    B = camera_K.shape[0]
    Kt = torch.zeros(B, 3, 3)
    Kt[:, 0, 0] = it['fx']; Kt[:, 1, 1] = it['fy']; Kt[:, 2, 2] = 1.0
    Kt[:, 0, 2] = it['cx'] + camera_K[:, 0, 2]      # cx0 - bx0
    Kt[:, 1, 2] = it['cy'] + camera_K[:, 1, 2]      # cy0 - by0
    return scale_K(Kt, original_size, S)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--detector', default=None)
    ap.add_argument('--mlp-head', default=None)
    ap.add_argument('--val-dir', default='../Dataset/Converted_dataset/DREAM_to_DREAM_syn/panda_synth_test_dr')
    ap.add_argument('--model-name', default='facebook/dinov3-vitb16-pretrain-lvd1689m')
    ap.add_argument('--image-size', type=int, default=512)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--max-frames', type=int, default=1000)
    ap.add_argument('--iters', type=int, default=200)
    ap.add_argument('--refiner', default='reproj', choices=['reproj', 'heatmap'],
                    help='reproj=fit argmax keypoints; heatmap=maximize heatmap response (bidirectional)')
    ap.add_argument('--conf-gate', type=float, default=0.05,
                    help='hard-reject keypoints below this confidence from PnP+refinement '
                         '(handles occluded/off-frame keypoints; +0.018 ADD-AUC on azure)')
    ap.add_argument('--rot-head', default=None,
                    help='optional rotation head: predict R and seed the solver R_init '
                         '(fixes the far-camera wrong-rotation-basin; realsense +0.117 ADD-AUC)')
    ap.add_argument('--crop', action='store_true',
                    help='crop image to robot bbox (RoboPEPP-style); use with a crop-trained detector+angle head')
    ap.add_argument('--crop-margin', type=float, default=1.5)
    ap.add_argument('--obs-fusion', action='store_true',
                    help='ALSO report observability-fused ADD (post-solve fuse toward head) at each '
                         '--obs-c, for cross-robot do-no-harm (all Panda joints observable -> w~1)')
    ap.add_argument('--obs-c', type=float, nargs='+', default=[20, 40, 80],
                    help='fusion knee grid c (px/rad); reported alongside the baseline in one pass')
    ap.add_argument('--obs-dump', default='',
                    help='opt-in: save per-joint observability profile {O,head,solve,gt} npz '
                         '(for Eval/obs_profile.py). Default off -> path untouched, bit-identical.')
    args = ap.parse_args()

    def latest(pat):
        f = sorted(glob.glob(pat), key=os.path.getmtime); return f[-1] if f else None
    det = args.detector or latest(f'{TRAIN}/outputs_heatmap/stage1_unfrozen_*/best_heatmap.pth')
    # glob only timestamped plain-mlp runs (angle_<ts>); exclude angle_patch_* (head_type
    # mismatch -> won't load into the 'mlp' AnglePredictor here).
    mlp_h = args.mlp_head or latest(f'{TRAIN}/outputs_angle/angle_[0-9]*/best_angle_head.pth')
    print(f"detector: {det}\nmlp head: {mlp_h}\nval: {args.val_dir}")

    device = torch.device('cuda'); assert torch.cuda.is_available()
    # with_translation builds the R+t head structure so the (R+t) checkpoint loads; we feed only
    # R_init to the solver — the learned t fails sim2real on far cameras (587mm) and HURTS, so it
    # is intentionally NOT used as t_init. R is the keeper.
    mlp = AnglePredictor(args.model_name, args.image_size, head_type='mlp',
                         with_rotation=args.rot_head is not None,
                         with_translation=args.rot_head is not None).to(device).eval()
    sd = torch.load(det, map_location=device); sd = {k.replace('module.', ''): v for k, v in sd.items()}
    mlp.load_state_dict({k: v for k, v in sd.items() if k in mlp.state_dict()
                         and v.shape == mlp.state_dict()[k].shape}, strict=False)
    if args.rot_head:
        mlp.rot_head.load_state_dict(torch.load(args.rot_head, map_location=device))
        print(f"rot head: {args.rot_head}")
    mlp.angle_head.load_state_dict(torch.load(mlp_h, map_location=device))

    ds = PoseEstimationDataset(args.val_dir, keypoint_names=['link0','link2','link3','link4','link6','link7','hand'],
                               image_size=(args.image_size, args.image_size),
                               heatmap_size=(args.image_size, args.image_size),
                               augment=False, include_angles=True, sigma=2.5,
                               crop_to_robot=args.crop, crop_margin=args.crop_margin)
    if args.max_frames and args.max_frames < len(ds):
        # STRIDE for a representative sample; ds.samples[:N] is one biased contiguous trajectory segment
        stride = max(1, len(ds.samples) // args.max_frames)
        ds.samples = ds.samples[::stride][:args.max_frames]
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    raw_err = torch.zeros(6); ref_err = torch.zeros(6); n = 0
    adds = []  # per-frame ADD (m) from refined camera-frame pose
    adds_fus = {c: [] for c in args.obs_c} if args.obs_fusion else {}
    _D = {k: [] for k in ('O', 'head', 'solve', 'gt')} if args.obs_dump else None
    for batch in tqdm(loader, desc="refine eval"):
        img = batch['image'].to(device)
        gt_full = batch['angles'].to(device)
        gt = batch['angles'].to(device)[:, :6]
        gt3d = batch['keypoints_3d'].to(device)  # (B,7,3) camera frame, meters
        K = scale_K(batch['camera_K'], batch['original_size'], args.image_size).to(device)
        with torch.no_grad():
            out = mlp(img, K)
        init_ang = out['joint_angles']           # (B,7) radians
        kp2d = out['keypoints_2d']               # (B,7,2)
        conf = out['confidence']                 # (B,7)
        if args.refiner == 'heatmap':
            refined, kp_cam, _ = solve_batch_heatmap(out['heatmaps_2d'], K, fix_joint7=True,
                                                     iters=args.iters, lr=1e-2,
                                                     img_size=args.image_size, device=device,
                                                     theta_init=init_ang)
        else:
            R_init = out.get('rot_matrix') if args.rot_head else None
            if args.obs_fusion or args.obs_dump:
                refined, kp_cam, _, R_s, t_s = solve_batch(kp2d, conf, K, fix_joint7=True, iters=args.iters,
                                                           lr=2e-2, img_size=args.image_size, device=device,
                                                           prior_w=0.0, theta_init=init_ang,
                                                           conf_gate=args.conf_gate, R_init=R_init,
                                                           return_pose=True)
            else:
                refined, kp_cam, _ = solve_batch(kp2d, conf, K, fix_joint7=True, iters=args.iters,
                                                 lr=2e-2, img_size=args.image_size, device=device,
                                                 prior_w=0.0, theta_init=init_ang,
                                                 conf_gate=args.conf_gate, R_init=R_init)
        raw_err += wrapped_abs_deg(init_ang[:, :6], gt).sum(0).cpu()
        ref_err += wrapped_abs_deg(refined[:, :6], gt).sum(0).cpu()
        # ADD: per-joint 3D distance (camera frame), per-frame mean over valid GT joints
        valid = (gt3d.abs().sum(-1) > 0)         # (B,7)
        per_j = (kp_cam - gt3d).norm(dim=-1)     # (B,7) meters
        for b in range(img.shape[0]):
            m = valid[b]
            if m.any():
                adds.append(float(per_j[b][m].mean().item()))
        if args.obs_fusion or args.obs_dump:
            cmask = (conf >= args.conf_gate).float() if args.conf_gate > 0 else torch.ones_like(conf)
            O = obs_fusion.observability(panda_forward_kinematics, refined, R_s, t_s, K, cmask)
        if args.obs_dump:
            _D['O'].append(O.detach().cpu().numpy())
            _D['head'].append(init_ang.detach().cpu().numpy())
            _D['solve'].append(refined.detach().cpu().numpy())
            _D['gt'].append(gt_full.detach().cpu().numpy())
        if args.obs_fusion:
            for c in args.obs_c:
                th_f = obs_fusion.fuse_angles(refined, init_ang, obs_fusion.fusion_weights(O, c))
                kp_f = torch.einsum('bij,bnj->bni', R_s, panda_forward_kinematics(th_f)) + t_s.unsqueeze(1)
                pj = (kp_f - gt3d).norm(dim=-1)
                for b in range(img.shape[0]):
                    if valid[b].any():
                        adds_fus[c].append(float(pj[b][valid[b]].mean().item()))
        n += img.shape[0]

    raw = (raw_err / n).numpy(); ref = (ref_err / n).numpy()
    adds = np.array(adds)
    print(f"\n{'='*54}\n  MLP -> KINEMATIC REFINE  ({n} frames)  {os.path.basename(args.val_dir)}\n{'='*54}")
    print(f"  {'joint':<6}{'raw MLP':>10}{'refined':>10}{'delta':>9}")
    for j in range(6):
        print(f"  J{j:<5}{raw[j]:>10.2f}{ref[j]:>10.2f}{ref[j]-raw[j]:>+9.2f}")
    print(f"  {'MEAN':<6}{raw.mean():>10.2f}{ref.mean():>10.2f}{ref.mean()-raw.mean():>+9.2f}")
    print('-'*54)
    if len(adds):
        print(f"  [Pose] ADD-AUC@100mm: {add_auc(adds):.4f} | "
              f"mean ADD {adds.mean()*1000:.1f}mm | median {np.median(adds)*1000:.1f}mm "
              f"({len(adds)} frames)")
    if args.obs_fusion:
        base = add_auc(adds)
        for c in args.obs_c:
            af = np.array(adds_fus[c])
            print(f"  [obs-fusion c={c:<5g}] ADD-AUC@100mm: {add_auc(af):.4f} "
                  f"({add_auc(af)-base:+.4f}) | mean {af.mean()*1000:.1f}mm  (do-no-harm gate -0.003)")
    if args.obs_dump and _D['O']:
        out = {k: np.concatenate(v, 0) for k, v in _D.items()}
        out['names'] = np.array(['joint_1','joint_2','joint_3','joint_4','joint_5','joint_6','joint_7'])
        os.makedirs(os.path.dirname(args.obs_dump) or '.', exist_ok=True)
        np.savez_compressed(args.obs_dump, **out)
        print(f"  [obs-dump] saved {out['O'].shape[0]} frames -> {args.obs_dump}")
    print('='*54)


if __name__ == '__main__':
    main()
