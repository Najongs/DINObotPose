"""KUKA iiwa7 FULLY-AUTOMATIC bbox ADD-AUC — the auto-bbox analogue of kuka_add_eval.py.

kuka_add_eval.py feeds the crop pipeline a GROUND-TRUTH keypoint crop. This script removes that
oracle: it replicates Panda's two-pass self-bbox flow (Eval/selfbbox_eval.py) for the 7-DOF iiwa7:

  PASS 1  full-frame detector (+ stage-1 angle/rot heads) on the 512x512 frame
          -> kinematic solve (iiwa7 FK, TRUE K) -> project ALL 7 FK keypoints -> robot bbox
          (fills the occluded base the raw detector misses).  --bbox-guard falls back to the
          detected-keypoint bbox on diverged frames and clamps every box.
  crop    the bbox is built the way TRAIN/dataset.py builds the GT crop: a SQUARE box in the
          ORIGINAL 640x480 frame (side = max(dx,dy)*margin), so pass-2 is geometry-identical to
          the GT-crop eval — only the BOX SOURCE differs (solved skeleton vs GT keypoints).
          Implemented by mapping that original-space square into the 512 frame (a rectangle) and
          roi_align-ing it back to 512x512; crop_K then reproduces the exact GT-crop intrinsics
          (verified: --oracle-bbox reproduces kuka_add_eval's GT-crop ADD).
  PASS 2  crop detector + crop angle/rot heads + iiwa7 solver — the SAME call as kuka_add_eval's
          rot-head 'solver' mode (iters 250, conf-gate 0.05, TRUE crop-K, R_init/t_init from the
          crop rot head).

--oracle-bbox : build the box from GT keypoints -> must reproduce the GT-crop ADD (crop-math check).
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src', 'dinobotpose'))
import argparse, os, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np, torch
from torch.utils.data import DataLoader
from torchvision.ops import roi_align
from tqdm import tqdm

HERE = os.path.dirname(__file__)
TRAIN = os.path.abspath(os.path.join(HERE, '../TRAIN'))
from model_angle import AnglePredictor
from model_v4 import iiwa7_forward_kinematics
from dataset import PoseEstimationDataset
from refine_utils import scale_K, add_auc, geometric_K, wrapped_abs_deg
import solve as spk
from kuka_fk import _patch_solver_for_iiwa7   # monkeypatch spk FK/limits/mean -> iiwa7

FK = iiwa7_forward_kinematics
KP_NAMES = [f'iiwa7_link_{i}' for i in range(1, 8)]
ANGLE_JOINTS = [f'iiwa7_joint_{i}' for i in range(1, 8)]


def build_predictor(model_name, IS, det_ckpt, device, angle_ckpt=None, rot_ckpt=None):
    with_rot = rot_ckpt is not None
    m = AnglePredictor(model_name, IS, fix_joint7_zero=True, head_type='mlp',
                       with_rotation=with_rot, with_translation=with_rot).to(device).eval()
    sd = torch.load(det_ckpt, map_location=device)
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    m.load_state_dict({k: v for k, v in sd.items()
                       if k in m.state_dict() and v.shape == m.state_dict()[k].shape}, strict=False)
    if angle_ckpt:
        m.angle_head.load_state_dict(torch.load(angle_ckpt, map_location=device))
    if rot_ckpt:
        m.rot_head.load_state_dict(torch.load(rot_ckpt, map_location=device))
    return m


def project_points(kp_cam, K):
    z = kp_cam[..., 2:3].clamp(min=1e-4)
    uv = kp_cam[..., :2] / z
    fx, fy = K[:, 0, 0].unsqueeze(1), K[:, 1, 1].unsqueeze(1)
    cx, cy = K[:, 0, 2].unsqueeze(1), K[:, 1, 2].unsqueeze(1)
    u = uv[..., 0] * fx + cx; v = uv[..., 1] * fy + cy
    return torch.stack([u, v], dim=-1)


def bbox_original(kp512, conf, ow, oh, IS, margin, conf_thr):
    """SQUARE robot bbox in ORIGINAL (ow x oh) px from 512-frame keypoints — mirrors dataset.py's
    crop_to_robot (crop_aspect=1.0): convert 512->original, center=midrange, side=max(dx,dy)*margin."""
    B = kp512.shape[0]
    ox = kp512[..., 0] * (ow / IS); oy = kp512[..., 1] * (oh / IS)
    boxes = kp512.new_zeros(B, 4)
    for b in range(B):
        m = conf[b] > conf_thr
        if int(m.sum()) < 2:
            m = conf[b] > -1.0    # fallback: use all
        xs, ys = ox[b][m], oy[b][m]
        x0, x1 = xs.min(), xs.max(); y0, y1 = ys.min(), ys.max()
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        side = torch.clamp(torch.max(x1 - x0, y1 - y0) * margin, min=16.0)
        boxes[b, 0] = cx - side / 2; boxes[b, 1] = cy - side / 2
        boxes[b, 2] = cx + side / 2; boxes[b, 3] = cy + side / 2
    return boxes


def orig_to_512(box_o, ow, oh, IS):
    b = box_o.clone()
    b[:, 0] *= IS / ow; b[:, 2] *= IS / ow
    b[:, 1] *= IS / oh; b[:, 3] *= IS / oh
    return b


def crop_K_rect(K, box512, out):
    """K adjusted for a (possibly rectangular) crop box512 in 512-space, then resize each side->out."""
    Kc = K.clone()
    for b in range(K.shape[0]):
        x0, y0 = box512[b, 0], box512[b, 1]
        w = (box512[b, 2] - box512[b, 0]).clamp(min=1.0)
        h = (box512[b, 3] - box512[b, 1]).clamp(min=1.0)
        sx, sy = out / w, out / h
        Kc[b, 0, 0] = K[b, 0, 0] * sx; Kc[b, 1, 1] = K[b, 1, 1] * sy
        Kc[b, 0, 2] = (K[b, 0, 2] - x0) * sx; Kc[b, 1, 2] = (K[b, 1, 2] - y0) * sy
    return Kc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage1-detector', required=True)
    ap.add_argument('--stage1-angle', default=None)
    ap.add_argument('--stage1-rot', default=None)
    ap.add_argument('--crop-detector', required=True)
    ap.add_argument('--crop-angle', required=True)
    ap.add_argument('--rot-head', default=None, help='crop rot head -> R_init/t_init (matches kuka_add_eval solver)')
    ap.add_argument('--val-dir', required=True)
    ap.add_argument('--model-name', default='facebook/dinov3-vitb16-pretrain-lvd1689m')
    ap.add_argument('--image-size', type=int, default=512)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--max-frames', type=int, default=0, help='0 = all frames')
    ap.add_argument('--iters', type=int, default=250)
    ap.add_argument('--margin', type=float, default=1.5)
    ap.add_argument('--conf-gate', type=float, default=0.05)
    ap.add_argument('--bbox-conf', type=float, default=0.1)
    ap.add_argument('--bbox-from-solved', action='store_true')
    ap.add_argument('--bbox-union', action='store_true',
                    help='box from UNION of ON-FRAME solved kp and confidently-detected kp (Panda '
                         'selfbbox convention). Uniform box-construction A/B vs --bbox-from-solved.')
    ap.add_argument('--bbox-guard', action='store_true')
    ap.add_argument('--oracle-bbox', action='store_true', help='box from GT keypoints (crop-math sanity)')
    ap.add_argument('--oracle-angle', action='store_true', help='known-joint ceiling: inject GT theta, freeze it, solve only R,t (mirrors selfbbox_eval --oracle-angle)')
    ap.add_argument('--freeze-head-theta', action='store_true', help='P0 DECOUPLE (RoboPEPP-style): freeze theta at the HEAD prediction (no GT) and solve ONLY camera R,t (mirrors selfbbox_eval --freeze-head-theta)')
    ap.add_argument('--occlude-ratio', type=float, default=0.0, help='DIAGNOSTIC (opt-in, default off = bit-identical): paste distractor occluders covering this fraction of the robot, using the same generator and per-frame seeding as the Panda occlusion protocol, so the two robots\' curves are directly comparable.')
    ap.add_argument('--cov-pnp', action='store_true')
    args = ap.parse_args()

    device = torch.device('cuda'); assert torch.cuda.is_available(); IS = args.image_size
    _patch_solver_for_iiwa7()

    det1 = build_predictor(args.model_name, IS, args.stage1_detector, device,
                           angle_ckpt=args.stage1_angle if args.bbox_from_solved else None,
                           rot_ckpt=args.stage1_rot if args.bbox_from_solved else None)
    cropm = build_predictor(args.model_name, IS, args.crop_detector, device,
                            angle_ckpt=args.crop_angle, rot_ckpt=args.rot_head)
    print(f"stage1 det: {args.stage1_detector}\ncrop det: {args.crop_detector}\nval: {args.val_dir}")
    print(f"mode: {'ORACLE-bbox' if args.oracle_bbox else 'bbox-from-solved' if args.bbox_from_solved else 'detector-bbox'} "
          f"| guard={args.bbox_guard} | iters={args.iters} conf_gate={args.conf_gate}")

    # full-frame dataset (NO crop) -> we crop ourselves from the pass-1 bbox
    ds = PoseEstimationDataset(args.val_dir, keypoint_names=KP_NAMES, image_size=(IS, IS),
                               heatmap_size=(IS, IS), augment=False, include_angles=True, sigma=2.5,
                               crop_to_robot=False, angle_joint_names=ANGLE_JOINTS)
    if args.max_frames and args.max_frames < len(ds):
        ds.samples = ds.samples[::max(1, len(ds.samples) // args.max_frames)][:args.max_frames]
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    print(f"frames: {len(ds)}")

    raw_err = torch.zeros(6); ref_err = torch.zeros(6); n = 0; adds = []
    n_fb = 0
    for batch in tqdm(loader, desc='kuka-autobbox'):
        img = batch['image'].to(device)
        if args.occlude_ratio > 0:
            from occl_util import paste_occluders_batch_
            paste_occluders_batch_(img, batch['keypoints'].numpy(), batch['valid_mask'].numpy(),
                                   args.occlude_ratio, batch['name'])
        gt = batch['angles'].to(device)[:, :6]
        gt3d = batch['keypoints_3d'].to(device)
        ow = float(batch['original_size'][0][0]); oh = float(batch['original_size'][0][1])
        K_model = scale_K(batch['camera_K'], batch['original_size'], IS).to(device)      # eye3-based (model)
        K_true = geometric_K(args.val_dir, batch['camera_K'], batch['original_size'], IS).to(device)
        bidx = torch.arange(img.shape[0], device=device).view(-1, 1).float()
        with torch.no_grad():
            if args.oracle_bbox:
                gkp = batch['keypoints'].to(device).float()
                # dataset.py builds the GT crop from IN-FRAME keypoints only (0<=x<IS & 0<=y<IS)
                inb = ((gkp[..., 0] >= 0) & (gkp[..., 0] < IS) &
                       (gkp[..., 1] >= 0) & (gkp[..., 1] < IS)).float()
                box_o = bbox_original(gkp, inb, ow, oh, IS, args.margin, 0.5)
            elif args.bbox_from_solved:
                o1 = det1(img, K_model)
                R1 = o1.get('rot_matrix') if args.stage1_rot else None
                t1 = o1.get('trans') if args.stage1_rot else None
                with torch.enable_grad():
                    _, kp_cam1, reproj1 = spk.solve_batch(o1['keypoints_2d'], o1['confidence'], K_true,
                                                          fix_joint7=True, iters=args.iters, lr=2e-2,
                                                          img_size=IS, device=device, prior_w=0.0,
                                                          theta_init=o1['joint_angles'], conf_gate=args.conf_gate,
                                                          R_init=R1, t_init=t1)
                kp_cam1 = kp_cam1.detach()
                uv = project_points(kp_cam1, K_true)               # (B,7,2) 512-frame, all points
                if args.bbox_union:
                    onf = ((uv[..., 0] >= 0) & (uv[..., 0] < IS) &
                           (uv[..., 1] >= 0) & (uv[..., 1] < IS)).float()
                    upts = torch.cat([uv, o1['keypoints_2d']], dim=1)
                    uconf = torch.cat([onf, o1['confidence']], dim=1)
                    box_o = bbox_original(upts, uconf, ow, oh, IS, args.margin, args.bbox_conf)
                else:
                    box_o = bbox_original(uv, torch.ones(uv.shape[:2], device=device), ow, oh, IS, args.margin, 0.0)
                if args.bbox_guard:
                    det_o = bbox_original(o1['keypoints_2d'], o1['confidence'], ow, oh, IS, args.margin, args.bbox_conf)
                    off = ((uv < -IS) | (uv > 2 * IS)).any(dim=2).any(dim=1)
                    span = (uv.amax(dim=1) - uv.amin(dim=1)).amax(dim=1)
                    bad = off | (span > 3.0 * IS) | torch.isnan(uv).any(dim=2).any(dim=1)
                    box_o = torch.where(bad.unsqueeze(1), det_o, box_o)
                    n_fb += int(bad.sum())
            else:
                o1 = det1(img, K_model)
                box_o = bbox_original(o1['keypoints_2d'], o1['confidence'], ow, oh, IS, args.margin, args.bbox_conf)

            box512 = orig_to_512(box_o, ow, oh, IS)
            rois = torch.cat([bidx, box512], dim=1)
            crop_img = roi_align(img, rois, output_size=(IS, IS), spatial_scale=1.0, aligned=True)
            Kc_model = crop_K_rect(K_model, box512, float(IS))
            Kc_true = crop_K_rect(K_true, box512, float(IS))
            o2 = cropm(crop_img, Kc_model)

        init_ang = o2['joint_angles']; kp2d = o2['keypoints_2d']; conf = o2['confidence']
        if args.oracle_angle:                                     # known-joint ceiling: GT theta, frozen, solve only R,t
            init_ang = init_ang.clone(); init_ang[:, :6] = gt
        cov_inv = spk.heatmap_cov_inv(o2['heatmaps_2d'], kp2d) if args.cov_pnp else None
        R_init = o2.get('rot_matrix') if args.rot_head else None
        t_init = o2.get('trans') if args.rot_head else None
        with torch.enable_grad():
            refined, kp_cam, reproj = spk.solve_batch(kp2d, conf, Kc_true, fix_joint7=True, iters=args.iters,
                                                      lr=2e-2, img_size=IS, device=device, prior_w=0.0,
                                                      theta_init=init_ang, cov_inv=cov_inv, freeze_theta=(args.oracle_angle or args.freeze_head_theta),
                                                      conf_gate=args.conf_gate, R_init=R_init, t_init=t_init)
        raw_err += wrapped_abs_deg(init_ang[:, :6], gt).sum(0).cpu()
        ref_err += wrapped_abs_deg(refined[:, :6], gt).sum(0).cpu()
        valid = (gt3d.abs().sum(-1) > 0)
        per_j = (kp_cam - gt3d).norm(dim=-1)
        for b in range(img.shape[0]):
            if valid[b].any():
                adds.append(float(per_j[b][valid[b]].mean().item()))
        n += img.shape[0]

    raw = (raw_err / n).numpy(); ref = (ref_err / n).numpy(); adds = np.array(adds)
    print(f"\n{'='*60}\n  KUKA iiwa7 AUTO-BBOX ADD  ({n} frames)  {os.path.basename(args.val_dir)}\n{'='*60}")
    print(f"  {'joint':<6}{'raw head':>10}{'refined':>10}{'delta':>9}   (deg MAE; joint_1..6)")
    for j in range(6):
        print(f"  J{j:<5}{raw[j]:>10.2f}{ref[j]:>10.2f}{ref[j]-raw[j]:>+9.2f}")
    print('-' * 60)
    print(f"  [Pose] ADD-AUC@100mm: {add_auc(adds):.4f} | mean ADD {adds.mean()*1000:.1f}mm | "
          f"median {np.median(adds)*1000:.1f}mm ({len(adds)} frames)")
    if args.bbox_guard:
        print(f"  [guard] bbox fell back to detected on {n_fb} frames")
    print('=' * 60)


if __name__ == '__main__':
    main()
