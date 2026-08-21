"""
Train a camera-rotation head (robot->camera 6D) on a frozen Stage-1 detector.

WHY: the residual realsense ADD gap is the kinematic solver landing in a wrong ROTATION basin
(rot_err 47 deg even with good 2D); 2D reprojection is degenerate at distance. depth_diag +
rinit_probe showed an oracle ROTATION init recovers +0.11 realsense ADD-AUC. This head predicts
that rotation from DINOv3 APPEARANCE (which way the robot faces) + keypoint geometry, to seed the
solver's R_init. Only rot_head trains; backbone + keypoint head frozen. GT rotation =
Kabsch(FK(gt_angles) -> keypoints_3d) per frame. Loss = Frobenius matrix error; val = geodesic deg.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'src', 'dinobotpose'))
import argparse, math, os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from model_angle import AnglePredictor, kabsch_batch
from model_v4 import panda_forward_kinematics, iiwa7_forward_kinematics, baxter_left_forward_kinematics
from dataset import PoseEstimationDataset

try:
    import wandb; _HAS_WANDB = True
except Exception:
    _HAS_WANDB = False


def scale_K(camera_K, original_size, hm):
    K = camera_K.clone().float()
    for b in range(K.shape[0]):
        ow, oh = float(original_size[b][0]), float(original_size[b][1])
        K[b, 0, 0] *= hm / ow; K[b, 1, 1] *= hm / oh
        K[b, 0, 2] *= hm / ow; K[b, 1, 2] *= hm / oh
    return K


def geodesic_deg(Rp, Rg):
    """(B,3,3),(B,3,3) -> (B,) geodesic angle in deg."""
    c = ((Rp.transpose(1, 2) @ Rg).diagonal(dim1=1, dim2=2).sum(-1) - 1) / 2
    return torch.acos(c.clamp(-1 + 1e-6, 1 - 1e-6)) * 180 / math.pi


def _check_k_value(kv, tg, tp, rot_head):
    """One-time smoke gate for the RootNet depth head, run on the first training batch.

    THE failure mode this guards against: computing k_value from post-`--crop-to-robot`
    coordinates. The crop is resized to a fixed 512 side, so every robot ends up the same
    apparent size -> k_value becomes near-constant and CARRIES NO DEPTH INFORMATION, silently
    voiding the whole experiment (the head would still train, just on a useless prior).
    A correctly-sourced k_value (original bbox + original K) both varies and tracks GT depth,
    so we assert on its spread and on its correlation with the GT root depth.
    """
    kv = kv.detach().float(); gz = tg.detach().float()[:, 2]; pz = tp.detach().float()[:, 2]
    kd = kv / 1000.0                                        # mm -> m, the raw geometric prior
    q = lambda t: [round(v, 3) for v in torch.quantile(
        t, torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], device=t.device)).tolist()]
    gamma = (pz / kd.clamp(min=1e-6))                       # z = gamma * k_value/1000  =>  invert
    cv = (kd.std() / kd.mean().clamp(min=1e-6)).item()
    kc, gc = kd - kd.mean(), gz - gz.mean()
    corr = (kc * gc).sum() / (kc.norm() * gc.norm()).clamp(min=1e-9)
    print("\n[depth-head smoke] (first batch, n=%d)" % kv.numel())
    print(f"  k_value/1000 [m]  min/q25/med/q75/max = {q(kd)}   mean={kd.mean():.3f} cv={cv:.3f}")
    print(f"  GT root depth [m] min/q25/med/q75/max = {q(gz)}   mean={gz.mean():.3f}")
    print(f"  gamma (=z/k)      min/q25/med/q75/max = {q(gamma)}   mean={gamma.mean():.3f}")
    print(f"  pred z [m]        min/q25/med/q75/max = {q(pz)}")
    print(f"  corr(k_value, GT depth) = {corr:.3f}   gamma_out.grad wired = "
          f"{rot_head.gamma_out.weight.requires_grad}\n")
    assert torch.isfinite(kd).all() and (kd > 0).all(), "k_value must be finite and > 0"
    assert (pz > 0).all(), "softplus must keep predicted z > 0"
    # ORIGINAL-frame assertions. Under crop-then-resize k_value collapses to a constant:
    # cv ~ 0 and corr ~ 0. Real original-frame values vary by tens of percent and track depth.
    assert cv > 0.05, (f"k_value is near-constant (cv={cv:.4f}) -> it was almost certainly "
                       f"computed AFTER --crop-to-robot resized to a fixed side; the "
                       f"apparent-size depth cue is destroyed. Must use the ORIGINAL bbox + K.")
    assert 0.5 < kd.median() < 10.0, (f"k_value median {kd.median():.3f} m is outside any plausible "
                                      f"depth band -> fx/fy are almost certainly wrong. NOTE: DREAM "
                                      f"frame JSONs have no meta.K, so camera_K is eye(3); k_value "
                                      f"must use _camera_settings.json (dataset.py::_intrinsics_for).")
    assert 0.05 < gamma.median() < 20.0, f"gamma should be O(1), got median {gamma.median():.3f}"
    # NOT an assert: measured offline (2026-07-22) corr is only 0.14 (KUKA) / 0.32 (Baxter). For a
    # 7-DOF arm the bbox area is driven by JOINT CONFIGURATION far more than by distance, unlike
    # RootNet's human-body assumption, so the prior is weakly informative here by nature.
    if corr < 0.3:
        print(f"  [WARN] corr(k_value, depth)={corr:.3f} is low. Expected for articulated arms "
              f"(bbox is config-dominated); gamma must absorb the configuration itself. Only "
              f"suspect miswiring if cv is also tiny.\n")


_FK = panda_forward_kinematics   # robot forward-kinematics used to build the GT R,t (swap per --fk-robot)


def gt_pose(gt_angles, kp3d, valid):
    """GT robot->camera (R,t) via Kabsch. gt_angles (B,>=6), kp3d (B,7,3), valid (B,7)."""
    ga = gt_angles.clone()
    if ga.shape[1] > 6: ga[:, 6] = 0.0              # Panda fixes joint7=0 (Meca has 6 angles)
    fk = _FK(ga)                                    # (B,7,3) robot frame
    return kabsch_batch(fk, kp3d, valid.float())    # (B,3,3),(B,3)


def main(args):
    global _FK
    device = torch.device('cuda'); assert torch.cuda.is_available()
    if args.fk_robot in ('kuka', 'iiwa7'):
        _FK = iiwa7_forward_kinematics
        print('==> using iiwa7 FK for GT rotation labels')
    elif args.fk_robot in ('baxter', 'baxter_left'):
        _FK = baxter_left_forward_kinematics
        print('==> using baxter-left FK for GT rotation labels')
    ang_names = args.angle_joint_names.split(',') if getattr(args, 'angle_joint_names', None) else None
    kp_names = (args.keypoint_names.split(',') if args.keypoint_names
                else ['link0', 'link2', 'link3', 'link4', 'link6', 'link7', 'hand'])
    # SigLIP/SigLIP2 expect mean=std=0.5 ([-1,1]); DINOv3 uses ImageNet stats. Must match the backbone.
    if "siglip" in args.model_name:
        norm_mean, norm_std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        print("==> SigLIP backbone detected: using mean=std=0.5 normalization")
    else:
        norm_mean = norm_std = None
    mk = lambda d, aug: PoseEstimationDataset(
        data_dir=d, keypoint_names=kp_names, image_size=(args.image_size, args.image_size),
        heatmap_size=(args.image_size, args.image_size), augment=aug, aug_level='strong',
        include_angles=True, sigma=2.5,
        crop_to_robot=args.crop_to_robot, crop_margin=args.crop_margin,
        crop_aspect=args.crop_aspect,
        norm_mean=norm_mean, norm_std=norm_std,
        angle_joint_names=ang_names)
    train_ds, val_ds = mk(args.train_dir, True), mk(args.val_dir, False)
    if getattr(args, 'max_train', 0) and args.max_train < len(train_ds):
        train_ds.samples = train_ds.samples[::max(1, len(train_ds.samples) // args.max_train)][:args.max_train]
        print(f"==> subsampled train to {len(train_ds)} frames")
    if getattr(args, 'max_val', 0) and args.max_val < len(val_ds):
        val_ds.samples = val_ds.samples[::max(1, len(val_ds.samples) // args.max_val)][:args.max_val]
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    model = AnglePredictor(args.model_name, args.image_size, fix_joint7_zero=True,
                           head_type='mlp', with_rotation=True, with_translation=True,
                           use_rootnet_depth=args.depth_head).to(device)
    ckpt = torch.load(args.detector_ckpt, map_location=device)
    ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
    msd = model.state_dict()
    model.load_state_dict({k: v for k, v in ckpt.items() if k in msd and v.shape == msd[k].shape}, strict=False)
    model.freeze_detector()
    if args.init_head:
        # depth-head adds gamma_out, absent from a legacy rot_head checkpoint -> non-strict then.
        model.rot_head.load_state_dict(torch.load(args.init_head, map_location=device),
                                       strict=not args.depth_head)
        print(f'[warm-start] rot_head <- {args.init_head}')
    print("==> training rotation head only")
    if args.depth_head:
        print(f"==> RootNet depth head ON: t_z = softplus(gamma)*k_value/1000 "
              f"(depth_weight={args.depth_weight})")

    opt = optim.AdamW(model.rot_head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.min_lr)
    if args.use_wandb and _HAS_WANDB:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    best = 1e9
    depth_checked = not args.depth_head       # one-time k_value sanity gate (see _check_k_value)
    start_epoch = 0
    _state = out / 'train_state.pt'
    if getattr(args, 'auto_resume', False) and _state.is_file():
        st = torch.load(_state, map_location=device, weights_only=False)
        model.rot_head.load_state_dict(st['head']); opt.load_state_dict(st['opt']); sched.load_state_dict(st['sched'])
        start_epoch = st['epoch'] + 1; best = st['best']
        print(f"==> RESUMED rot from epoch {st['epoch']} -> {start_epoch}, best {best:.2f}", flush=True)
    for epoch in range(start_epoch, args.epochs):
        model.rot_head.train()
        run = 0.0
        for batch in tqdm(train_loader, desc=f"Ep{epoch}"):
            imgs = batch['image'].to(device)
            gt_ang = batch['angles'].to(device)
            kp3d = batch['keypoints_3d'].to(device)
            K = scale_K(batch['camera_K'], batch['original_size'], args.image_size).to(device)
            valid = (kp3d.abs().sum(-1) > 1e-6) & (kp3d[..., 2] > 0)     # (B,7)
            keep = valid.sum(1) >= 4
            if keep.sum() == 0:
                continue
            Rg, tg = gt_pose(gt_ang, kp3d, valid)
            if args.occlude_aug > 0:
                import sys as _s, os as _o
                _s.path.append(_o.path.join(_o.path.dirname(_o.path.abspath(__file__)), '../Eval'))
                from occl_util import paste_random_occluders_
                paste_random_occluders_(imgs, batch['keypoints'].numpy(), batch['valid_mask'].numpy(), args.occlude_aug)
            kv = batch['k_value'].to(device) if args.depth_head else None
            if kv is not None:
                keep = keep & (kv > 0)                 # k_value undetermined (<2 in-frame kps)
                if keep.sum() == 0:
                    continue
            o = model(imgs, K, k_value=kv)
            if not depth_checked:
                _check_k_value(kv[keep], tg[keep], o['trans'][keep], model.rot_head)
                depth_checked = True
            r_loss = ((o['rot_matrix'][keep] - Rg[keep]) ** 2).sum(dim=(1, 2)).mean()  # Frobenius^2
            t_loss = F.smooth_l1_loss(o['trans'][keep], tg[keep])                       # meters
            loss = r_loss + args.t_weight * t_loss
            if args.depth_head and args.depth_weight > 0:
                # HoRoPose weights root-depth L1 x10: the pooled t_loss alone gives z too weak a
                # gradient for gamma to actually learn the scale->depth mapping.
                loss = loss + args.depth_weight * F.l1_loss(o['trans'][keep][:, 2], tg[keep][:, 2])
            opt.zero_grad(); loss.backward(); opt.step()
            run += loss.item()
        sched.step()

        model.rot_head.eval()
        # NOTE: `terrs` is the 3D norm |t_pred - t_gt|. It is NOT the depth error -- the old
        # variable name `tz_med` wrongly suggested that. dzs/dxys split it into the DEPTH
        # component and the LATERAL component, which is what decides whether a depth-only fix
        # (RootNet --depth-head, which replaces z alone) can move the number at all.
        gerrs, terrs, dzs, dxys = [], [], [], []
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch['image'].to(device)
                gt_ang = batch['angles'].to(device); kp3d = batch['keypoints_3d'].to(device)
                K = scale_K(batch['camera_K'], batch['original_size'], args.image_size).to(device)
                valid = (kp3d.abs().sum(-1) > 1e-6) & (kp3d[..., 2] > 0)
                keep = valid.sum(1) >= 4
                if keep.sum() == 0:
                    continue
                Rg, tg = gt_pose(gt_ang, kp3d, valid)
                kv = batch['k_value'].to(device) if args.depth_head else None
                if kv is not None:
                    keep = keep & (kv > 0)
                    if keep.sum() == 0:
                        continue
                o = model(imgs, K, k_value=kv)
                gerrs.append(geodesic_deg(o['rot_matrix'][keep], Rg[keep]).cpu())
                dt = (o['trans'][keep] - tg[keep]) * 1000                                # mm
                terrs.append(dt.norm(dim=-1).cpu())          # 3D norm (legacy 't-err')
                dzs.append(dt[:, 2].abs().cpu())             # DEPTH component
                dxys.append(dt[:, :2].norm(dim=-1).cpu())    # LATERAL component
        ge = torch.cat(gerrs); te = torch.cat(terrs)
        dz = torch.cat(dzs); dxy = torch.cat(dxys)
        med = ge.median().item(); mean = ge.mean().item()
        tz_med = te.median().item()
        dz_med = dz.median().item(); dxy_med = dxy.median().item()
        print(f"Ep{epoch} | val geo med={med:.2f} mean={mean:.2f} deg | t-err med={tz_med:.1f}mm "
              f"(|dz|={dz_med:.1f} dxy={dxy_med:.1f}) | train_loss={run/max(1,len(train_loader)):.4f}")
        if args.use_wandb and _HAS_WANDB:
            wandb.log({'epoch': epoch, 'val_geo_median': med, 'val_geo_mean': mean,
                       'val_t_err_mm': tz_med, 'val_dz_mm': dz_med, 'val_dxy_mm': dxy_med,
                       'train_loss': run / max(1, len(train_loader)),
                       'lr': opt.param_groups[0]['lr']})
        torch.save(model.rot_head.state_dict(), out / 'last_rot_head.pth')
        # rank by combined pose error (geodesic deg + t-err in cm) so t actually matters
        score = med + tz_med / 10.0
        if score < best:
            best = score; torch.save(model.rot_head.state_dict(), out / 'best_rot_head.pth')
            print(f"  -> new best (geo {med:.2f}deg + t {tz_med:.1f}mm)")
        if getattr(args, 'auto_resume', False):
            torch.save({'head': model.rot_head.state_dict(), 'opt': opt.state_dict(),
                        'sched': sched.state_dict(), 'epoch': epoch, 'best': best}, out / 'train_state.pt')
    print(f"Done. Best pose score = {best:.2f}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--detector-ckpt', required=True)
    p.add_argument('--train-dir', required=True); p.add_argument('--val-dir', required=True)
    p.add_argument('--keypoint-names', default=None,
                   help='comma-separated. Meca500: link0,link1,link2,link3,link4,link5,link6')
    p.add_argument('--fk-robot', default='panda',
                   choices=['panda', 'kuka', 'iiwa7', 'baxter', 'baxter_left'],
                   help='FK used to build GT robot->camera rotation labels')
    p.add_argument('--angle-joint-names', default=None,
                   help='comma-separated sim_state joint names for GT angles (KUKA: iiwa7_joint_1..7). '
                        'Default None = sim_state[:7] (Panda).')
    p.add_argument('--output-dir', default='./outputs_rotation')
    p.add_argument('--model-name', default='facebook/dinov3-vitb16-pretrain-lvd1689m')
    p.add_argument('--image-size', type=int, default=512); p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--epochs', type=int, default=40); p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--min-lr', type=float, default=1e-6); p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--t-weight', type=float, default=1.0, help='weight on the translation SmoothL1 loss')
    p.add_argument('--depth-head', action='store_true',
                   help='RootNet-style depth: t_z = softplus(gamma)*k_value/1000 instead of the '
                        'fixed-1.1m-prior regression. k_value = sqrt(fx*fy*1e6/max(bw,bh)^2) from the '
                        'ORIGINAL image bbox+K (dataset-side). Fixes the KUKA/Baxter t-err bottleneck.')
    p.add_argument('--depth-weight', type=float, default=10.0,
                   help='direct L1 supervision weight on t_z (HoRoPose uses 10x); only with --depth-head')
    p.add_argument('--crop-to-robot', action='store_true', help='robot-bbox crop (match a crop-trained detector)')
    p.add_argument('--occlude-aug', type=float, default=0.0,
                   help='train-time occlusion augmentation (see train_angle.py --occlude-aug)')
    p.add_argument('--init-head', default=None, help='warm-start rot_head from this state dict')
    p.add_argument('--crop-margin', type=float, default=1.5)
    p.add_argument('--crop-aspect', type=float, default=1.0,
                   help='crop rect w/h. 1.0=legacy square. Set to the deploy frame aspect '
                        '(640x480 -> 1.3333) to match Eval/selfbbox_eval.py roi_align crops.')
    p.add_argument('--num-workers', type=int, default=8)
    p.add_argument('--max-train', type=int, default=0, help='subsample training frames (0=all); for warm-start fine-tunes')
    p.add_argument('--max-val', type=int, default=0, help='subsample val frames (0=all)')
    p.add_argument('--use-wandb', action='store_true')
    p.add_argument('--wandb-project', default='dinov3-rotation'); p.add_argument('--wandb-run-name', default=None)
    p.add_argument('--auto-resume', action='store_true', help='resume head+opt+sched+epoch from <out>/train_state.pt')
    main(p.parse_args())
