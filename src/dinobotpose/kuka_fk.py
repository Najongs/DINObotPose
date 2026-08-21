"""KUKA iiwa7 keypoint-path ADD-AUC (7-DOF, DREAM kuka synth).

Reuses the SOTA kinematic solver (solve_pose_kinematic.solve_batch) WITHOUT editing it, by
monkeypatching the module's FK + joint limits + mean to the data-fit iiwa7 torch FK
(model_v4.iiwa7_forward_kinematics, verified 0.003mm vs DREAM). Mirrors meca_add_eval.py.

iiwa7 keypoints are link_1..7 (after each joint). joint_7 does NOT move any link origin
(self-axis rotation) -> unobservable from keypoints, so we fix_joint7=True and the head predicts
joint_1..6 (all observable: link_7 sees joint_6). This is the test the L2-diagnosis predicted:
the kinematic solver + conf-gate + cov-PnP should REJECT the detector's link-identity swaps
(pred link_i snapping onto link_j) as reprojection outliers and recover pose.

  --oracle-2d : feed GT 2D keypoints (upper bound of the keypoint PARAMETERIZATION).
  --oracle-R  : pin R_init/t_init via Kabsch(FK(gt_angles), gt3d) — isolates the rot-head.
  --head-direct : trust head angles, get (R,t) by PnP/reprojection (no joint refinement).
"""
import argparse, os, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np, torch
from torch.utils.data import DataLoader
from tqdm import tqdm

HERE = os.path.dirname(__file__)
TRAIN = os.path.abspath(os.path.join(HERE, '../TRAIN'))
sys.path.append(TRAIN); sys.path.append(HERE)
from model_angle import AnglePredictor
from model_v4 import iiwa7_forward_kinematics, _IIWA7_JOINT_LIMITS
from dataset import PoseEstimationDataset
from refine_utils import scale_K, add_auc, wrapped_abs_deg, geometric_K
import solve as spk
import obs_fusion   # per-joint observability-aware angle fusion (opt-in)

FK = iiwa7_forward_kinematics
KP_NAMES = [f'iiwa7_link_{i}' for i in range(1, 8)]
ANGLE_JOINTS = [f'iiwa7_joint_{i}' for i in range(1, 8)]


def kabsch_batch(A, B):
    """Rigid R,t mapping A(B,N,3) onto B(B,N,3). Returns R(B,3,3), t(B,3)."""
    ca = A.mean(1, keepdim=True); cb = B.mean(1, keepdim=True)
    H = (A - ca).transpose(1, 2) @ (B - cb)
    U, S, Vt = torch.linalg.svd(H)
    d = torch.sign(torch.linalg.det(Vt.transpose(1, 2) @ U.transpose(1, 2)))
    D = torch.eye(3, device=A.device, dtype=A.dtype).unsqueeze(0).repeat(A.shape[0], 1, 1)
    D[:, 2, 2] = d
    R = Vt.transpose(1, 2) @ D @ U.transpose(1, 2)
    t = cb.squeeze(1) - torch.einsum('bij,bj->bi', R, ca.squeeze(1))
    return R, t


def _patch_solver_for_iiwa7():
    spk.panda_forward_kinematics = iiwa7_forward_kinematics
    lims = _IIWA7_JOINT_LIMITS
    def _iiwa7_limits(device, dtype):
        lo = torch.tensor([l for l, _ in lims], device=device, dtype=dtype)
        hi = torch.tensor([h for _, h in lims], device=device, dtype=dtype)
        return lo, hi
    spk.make_limits = _iiwa7_limits
    spk.PANDA_JOINT_MEAN = torch.tensor([(l + h) / 2 for l, h in lims], dtype=torch.float32)  # (7,)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--detector', required=True)
    ap.add_argument('--angle-head', required=True)
    ap.add_argument('--rot-head', default=None, help='trained rot head -> R_init/t_init (basin-pin)')
    ap.add_argument('--val-dir', required=True)
    ap.add_argument('--model-name', default='facebook/dinov3-vitb16-pretrain-lvd1689m')
    ap.add_argument('--image-size', type=int, default=512)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--max-frames', type=int, default=1500)
    ap.add_argument('--iters', type=int, default=250)
    ap.add_argument('--crop-margin', type=float, default=1.5)
    ap.add_argument('--oracle-2d', action='store_true')
    ap.add_argument('--oracle-R', action='store_true', help='pin R_init/t_init via Kabsch(FK(gt), gt3d)')
    ap.add_argument('--head-direct', action='store_true',
                    help='trust head angles; (R,t) via learned-R+reprojection-t or PnP')
    ap.add_argument('--direct-pose', action='store_true',
                    help='trust head angles AND rot-head R+t DIRECTLY (no solve/re-solve). '
                         'SUPERSEDED 2026-07-22: this mode existed because the solver appeared to '
                         'diverge, but that was an identity-K bug (dataset camera_K = eye(3) fed '
                         'into PnP => 320x-wrong focal => collapsed depth), NOT depth ambiguity or '
                         'link-confusion. With true intrinsics the solver beats direct-pose by a '
                         'wide margin. Kept only for reproducing the old baseline.')
    ap.add_argument('--cov-pnp', action='store_true')
    ap.add_argument('--conf-gate', type=float, default=0.05)
    ap.add_argument('--robust-scale', type=float, default=0.0,
                    help='adaptive robust-loss knee = robust_scale x per-frame median residual '
                         '(0=off -> fixed 8px Geman-McClure, bit-identical baseline)')
    ap.add_argument('--shed-k', type=float, default=0.0,
                    help='soft floor: after an initial solve, drop keypoints with residual > '
                         'shed_k x per-frame median and re-solve below the min_kp floor, adopting '
                         'only if reprojection on the retained keypoints improves (0=off)')
    ap.add_argument('--obs-fusion', action='store_true',
                    help='ALSO report observability-fused ADD at each --obs-c (cross-robot do-no-harm; '
                         'iiwa7 joint_1..6 all observable -> w~1 -> fusion is a near-no-op)')
    ap.add_argument('--obs-c', type=float, nargs='+', default=[20, 40, 80])
    ap.add_argument('--obs-dump', default='',
                    help='opt-in: save per-joint observability profile {O,head,solve,gt} npz '
                         '(for Eval/obs_profile.py). Default off -> path untouched, bit-identical.')
    args = ap.parse_args()

    device = torch.device('cuda'); assert torch.cuda.is_available(); IS = args.image_size
    _patch_solver_for_iiwa7()

    m = AnglePredictor(args.model_name, IS, fix_joint7_zero=True, head_type='mlp',
                       with_rotation=bool(args.rot_head), with_translation=bool(args.rot_head)).to(device).eval()
    sd = torch.load(args.detector, map_location=device); sd = {k.replace('module.', ''): v for k, v in sd.items()}
    m.load_state_dict({k: v for k, v in sd.items() if k in m.state_dict() and v.shape == m.state_dict()[k].shape}, strict=False)
    m.angle_head.load_state_dict(torch.load(args.angle_head, map_location=device))
    if args.rot_head:
        m.rot_head.load_state_dict(torch.load(args.rot_head, map_location=device))

    ds = PoseEstimationDataset(args.val_dir, keypoint_names=KP_NAMES, image_size=(IS, IS), heatmap_size=(IS, IS),
                               augment=False, include_angles=True, sigma=2.5,
                               crop_to_robot=True, crop_margin=args.crop_margin, angle_joint_names=ANGLE_JOINTS)
    if args.max_frames and args.max_frames < len(ds):
        ds.samples = ds.samples[::max(1, len(ds.samples)//args.max_frames)][:args.max_frames]
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    raw_err = torch.zeros(6); ref_err = torch.zeros(6); n = 0; adds = []
    adds_fus = {c: [] for c in args.obs_c} if args.obs_fusion else {}
    _D = {k: [] for k in ('O', 'head', 'solve', 'gt')} if args.obs_dump else None
    _SHED = {}
    for batch in tqdm(loader, desc='kuka-add'):
        img = batch['image'].to(device)
        gt_full = batch['angles'].to(device)
        gt = batch['angles'].to(device)[:, :6]              # joint_1..6 (joint_7 unobservable)
        gt3d = batch['keypoints_3d'].to(device)             # (B,7,3) camera frame, meters
        # DISCIPLINE: dataset K -> MODEL (checkpoints learned bearing features from the eye(3)
        # DREAM fallback), TRUE metric K -> any geometric solve. The kuka/baxter synth trees
        # carry no meta.K, so passing the dataset K into PnP/refine put a 320x-wrong focal in
        # and collapsed solved depth — that, not "link confusion", was the solver divergence.
        K = scale_K(batch['camera_K'], batch['original_size'], IS).to(device)
        K_true = geometric_K(args.val_dir, batch['camera_K'], batch['original_size'], IS).to(device)
        with torch.no_grad():
            o = m(img, K)
        init_ang = o['joint_angles']                         # (B,7), joint_7=0
        if args.oracle_2d:
            kp2d = batch['keypoints'].to(device).float()
            conf = batch['valid_mask'].to(device).float().clamp(min=1e-3)
        else:
            kp2d = o['keypoints_2d']; conf = o['confidence']
        cov_inv = spk.heatmap_cov_inv(o['heatmaps_2d'], kp2d) if (args.cov_pnp and not args.oracle_2d) else None
        if args.direct_pose:
            # trust head angles + rot-head R,t directly. No optimization: the reprojection
            # angle-refine diverges (link-confusion pulls the fit) and the t-re-solve diverges
            # in depth (2D reprojection ~ scale/depth invariant). rot-head's direct t wins.
            fk_h = FK(init_ang.double()).float()
            R_h = o['rot_matrix'].float(); t_h = o['trans'].float()
            refined = init_ang
            kp_cam = torch.einsum('bij,bnj->bni', R_h, fk_h) + t_h.unsqueeze(1)
        elif args.head_direct:
            fk_h = FK(init_ang.double()).float()             # (B,7,3) robot frame
            if args.rot_head:
                R_h = o['rot_matrix'].float()
                t_h = o['trans'].float().clone().detach().requires_grad_(True)
                opt_t = torch.optim.Adam([t_h], lr=5e-3)
                cam0 = torch.einsum('bij,bnj->bni', R_h, fk_h)
                w = conf.clamp(min=1e-3)
                for _ in range(80):
                    cam = cam0 + t_h.unsqueeze(1); z = cam[..., 2].clamp(min=1e-3)
                    u = cam[..., 0] / z * K_true[:, 0, 0:1] + K_true[:, 0, 2:3]
                    v = cam[..., 1] / z * K_true[:, 1, 1:2] + K_true[:, 1, 2:3]
                    loss = (((torch.stack([u, v], -1) - kp2d).norm(dim=-1)) * w).sum() / w.sum()
                    opt_t.zero_grad(); loss.backward(); opt_t.step()
                t_h = t_h.detach()
            else:
                Rn, tn, _ = spk.pnp_init(kp2d.detach().cpu().numpy(), fk_h.detach().cpu().numpy(),
                                         K_true.detach().cpu().numpy(), conf.detach().cpu().numpy(),
                                         min_kp=6, pnp_drop=1)
                R_h = torch.from_numpy(Rn).float().to(device); t_h = torch.from_numpy(tn).float().to(device)
            refined = init_ang
            kp_cam = torch.einsum('bij,bnj->bni', R_h, fk_h) + t_h.unsqueeze(1)
        else:
            R_init = t_init = None
            if args.oracle_R:
                fk_gt = FK(batch['angles'].to(device).double()).float()
                R_init, t_init = kabsch_batch(fk_gt, gt3d)
            elif args.rot_head:
                R_init = o.get('rot_matrix'); t_init = o.get('trans')
            with torch.enable_grad():
                out_solve = spk.solve_batch(kp2d, conf, K_true, fix_joint7=True, iters=args.iters,
                                            lr=2e-2, img_size=IS, device=device, prior_w=0.0,
                                            theta_init=init_ang, cov_inv=cov_inv,
                                            conf_gate=args.conf_gate, R_init=R_init, t_init=t_init,
                                            robust_scale=args.robust_scale, shed_k=args.shed_k,
                                            _shed_stats=_SHED, return_pose=args.obs_fusion or bool(args.obs_dump))
            if args.obs_fusion or args.obs_dump:
                refined, kp_cam, reproj, R_s, t_s = out_solve
            else:
                refined, kp_cam, reproj = out_solve
        raw_err += wrapped_abs_deg(init_ang[:, :6], gt).sum(0).cpu()
        ref_err += wrapped_abs_deg(refined[:, :6], gt).sum(0).cpu()
        valid = (gt3d.abs().sum(-1) > 0)
        per_j = (kp_cam - gt3d).norm(dim=-1)
        for b in range(img.shape[0]):
            if valid[b].any(): adds.append(float(per_j[b][valid[b]].mean().item()))
        if (args.obs_fusion or args.obs_dump) and not (args.oracle_2d or args.direct_pose or args.head_direct):
            cmask = (conf >= args.conf_gate).float() if args.conf_gate > 0 else torch.ones_like(conf)
            O = obs_fusion.observability(FK, refined, R_s, t_s, K_true, cmask)
            if args.obs_dump:
                _D['O'].append(O.detach().cpu().numpy())
                _D['head'].append(init_ang.detach().cpu().numpy())
                _D['solve'].append(refined.detach().cpu().numpy())
                _D['gt'].append(gt_full.detach().cpu().numpy())
        if args.obs_fusion and not (args.oracle_2d or args.direct_pose or args.head_direct):
            for c in args.obs_c:
                th_f = obs_fusion.fuse_angles(refined, init_ang, obs_fusion.fusion_weights(O, c))
                kp_f = torch.einsum('bij,bnj->bni', R_s, FK(th_f)) + t_s.unsqueeze(1)
                pj = (kp_f - gt3d).norm(dim=-1)
                for b in range(img.shape[0]):
                    if valid[b].any(): adds_fus[c].append(float(pj[b][valid[b]].mean().item()))
        n += img.shape[0]

    raw = (raw_err / n).numpy(); ref = (ref_err / n).numpy(); adds = np.array(adds)
    tag = 'ORACLE-2D' if args.oracle_2d else 'detected-2D'
    mode = ('direct-pose' if args.direct_pose else 'head-direct' if args.head_direct
            else 'oracle-R' if args.oracle_R else 'rot-head' if args.rot_head else 'solver')
    print(f"\n{'='*60}\n  KUKA iiwa7 keypoint-path ADD  [{tag} | {mode}]  ({n} frames)\n{'='*60}")
    print(f"  {'joint':<6}{'raw head':>10}{'refined':>10}{'delta':>9}   (deg MAE; joint_1..6)")
    for j in range(6):
        print(f"  J{j:<5}{raw[j]:>10.2f}{ref[j]:>10.2f}{ref[j]-raw[j]:>+9.2f}")
    print('-'*60)
    tail = float((adds > 0.1).mean()) * 100.0
    print(f"  [Pose] ADD-AUC@100mm: {add_auc(adds):.4f} | mean ADD {adds.mean()*1000:.1f}mm | "
          f"median {np.median(adds)*1000:.1f}mm ({len(adds)} frames)")
    print(f"  [Tail] >100mm: {tail:.1f}%  | p99 {np.percentile(adds,99)*1000:.0f}mm  "
          f"| robust_scale={args.robust_scale} shed_k={args.shed_k}")
    if args.obs_fusion and adds_fus and len(adds_fus[args.obs_c[0]]):
        base = add_auc(adds)
        for c in args.obs_c:
            af = np.array(adds_fus[c])
            print(f"  [obs-fusion c={c:<5g}] ADD-AUC: {add_auc(af):.4f} ({add_auc(af)-base:+.4f}) "
                  f"| mean {af.mean()*1000:.1f}mm  (do-no-harm gate -0.003)")
    if _SHED.get('flagged', 0):
        print(f"  [Shed] flagged {_SHED['flagged']}/{_SHED.get('total',0)} frames, "
              f"adopted {_SHED.get('adopted',0)}")
    if args.obs_dump and _D['O']:
        out = {k: np.concatenate(v, 0) for k, v in _D.items()}
        out['names'] = np.array([f'joint_{i}' for i in range(1, 8)])
        os.makedirs(os.path.dirname(args.obs_dump) or '.', exist_ok=True)
        np.savez_compressed(args.obs_dump, **out)
        print(f"  [obs-dump] saved {out['O'].shape[0]} frames -> {args.obs_dump}")
    print('='*60)


if __name__ == '__main__':
    main()
