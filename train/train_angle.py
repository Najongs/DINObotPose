"""
Train the learned angle predictor (Stage 1.5) on top of a frozen Stage-1 detector.

Only AngleHead trains. Input = the model's OWN predicted 2D keypoints (robust to detector
noise). Loss = sin/cos SmoothL1 to GT angles + FK robot-frame consistency. Eval = per-joint
angle MAE (deg) on the synthetic val set. Output checkpoint -> init for the kinematic refiner.
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

from model_angle import AnglePredictor
from model_v4 import panda_forward_kinematics, iiwa7_forward_kinematics, baxter_left_forward_kinematics
_FK_BY_ROBOT = {'panda': panda_forward_kinematics, 'fr3': panda_forward_kinematics,
                'kuka': iiwa7_forward_kinematics, 'iiwa7': iiwa7_forward_kinematics,
                'baxter': baxter_left_forward_kinematics, 'baxter_left': baxter_left_forward_kinematics}
import sys as _sys, os as _os
from geometry import kabsch_batch
from dataset import PoseEstimationDataset

try:
    import wandb
    _HAS_WANDB = True
except Exception:
    _HAS_WANDB = False


def scale_K(camera_K, original_size, hm):
    """camera_K (B,3,3) original-res -> heatmap-res, using original_size (B,2)=(W,H)."""
    K = camera_K.clone().float()
    for b in range(K.shape[0]):
        ow, oh = float(original_size[b][0]), float(original_size[b][1])
        sx, sy = hm / ow, hm / oh
        K[b, 0, 0] *= sx; K[b, 1, 1] *= sy
        K[b, 0, 2] *= sx; K[b, 1, 2] *= sy
    return K


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} | available={torch.cuda.is_available()}")
    assert torch.cuda.is_available(), "Refusing to train on CPU (check GPU UUID selection)."

    kp_names = (args.keypoint_names.split(',') if getattr(args, 'keypoint_names', None)
                else ['link0', 'link2', 'link3', 'link4', 'link6', 'link7', 'hand'])
    ang_names = args.angle_joint_names.split(',') if getattr(args, 'angle_joint_names', None) else None
    fk_fn = _FK_BY_ROBOT[args.fk_robot]
    print(f"==> FK robot: {args.fk_robot} | angle joints: {ang_names or 'sim_state[:7]'}")
    # SigLIP/SigLIP2 expect mean=std=0.5 ([-1,1]); DINOv3 uses ImageNet stats. Must match the
    # backbone the frozen detector was trained under, or features go off-distribution.
    if "siglip" in args.model_name:
        norm_mean, norm_std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        print("==> SigLIP backbone detected: using mean=std=0.5 normalization")
    else:
        norm_mean = norm_std = None  # dataset default = ImageNet
    train_ds = PoseEstimationDataset(
        data_dir=args.train_dir, keypoint_names=kp_names,
        image_size=(args.image_size, args.image_size),
        heatmap_size=(args.image_size, args.image_size),
        augment=True, aug_level='strong', include_angles=True, sigma=2.5,
        crop_to_robot=args.crop_to_robot, crop_margin=args.crop_margin,
        crop_aspect=args.crop_aspect,
        norm_mean=norm_mean, norm_std=norm_std,
        angle_joint_names=ang_names)
    val_ds = PoseEstimationDataset(
        data_dir=args.val_dir, keypoint_names=kp_names,
        image_size=(args.image_size, args.image_size),
        heatmap_size=(args.image_size, args.image_size),
        augment=False, include_angles=True, sigma=2.5,
        crop_to_robot=args.crop_to_robot, crop_margin=args.crop_margin,
        crop_aspect=args.crop_aspect,
        norm_mean=norm_mean, norm_std=norm_std,
        angle_joint_names=ang_names)
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
                           head_type=args.head_type, n_hyp=args.n_mix,
                           angle_backbone=args.angle_backbone).to(device)
    print(f"==> Angle head type: {args.head_type} | angle backbone: {args.angle_backbone}")

    # Load the Stage-1 detector weights into backbone + keypoint_head.
    ckpt = torch.load(args.detector_ckpt, map_location=device)
    ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
    msd = model.state_dict()
    loaded = {k: v for k, v in ckpt.items() if k in msd and v.shape == msd[k].shape}
    model.load_state_dict(loaded, strict=False)
    n_det = sum(1 for k in loaded if k.startswith('backbone.') or k.startswith('keypoint_head.'))
    print(f"==> Loaded {len(loaded)} tensors from detector ({n_det} into backbone+keypoint_head)")
    model.freeze_detector()
    if args.init_head:
        model.angle_head.load_state_dict(torch.load(args.init_head, map_location=device))
        print(f'[warm-start] angle_head <- {args.init_head}')

    # P1b: the DINOv3 detector (backbone + keypoint_head) MUST stay frozen (kp2d/conf only). The
    # separate ResNet50 angle trunk (angle_feat) is the only new trainable module besides the head.
    assert not any(p.requires_grad for p in model.backbone.parameters()), "DINOv3 backbone must be frozen"
    assert not any(p.requires_grad for p in model.keypoint_head.parameters()), "keypoint_head must be frozen"
    param_groups = [{'params': model.angle_head.parameters(), 'lr': args.lr}]
    if args.angle_backbone == 'resnet50':
        assert all(p.requires_grad for p in model.angle_feat.parameters()), "resnet50 angle_feat must be trainable"
        param_groups.append({'params': model.angle_feat.parameters(), 'lr': args.lr * 0.2})  # backbone LR = head×0.2
        print(f"==> resnet50 angle trunk trainable ({sum(p.numel() for p in model.angle_feat.parameters())/1e6:.1f}M params), lr={args.lr*0.2:.1e}")
    opt = optim.AdamW(param_groups, weight_decay=args.weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.min_lr)

    if args.use_wandb and _HAS_WANDB:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    best_mae = 1e9
    # P1 focal tail-reweighting: per-joint mask + detached per-joint EMA normalizer (decoupled from
    # the batch-32 tail so the loss scale is stationary). idx4=J5 mask~0 (observability write-off).
    focal_jw = torch.tensor([float(x) for x in args.joint_weights.split(',')], device=device)  # (6,)
    ema_dj = torch.ones(6, device=device)
    start_epoch = 0
    _state = out / 'train_state.pt'
    if getattr(args, 'auto_resume', False) and _state.is_file():
        st = torch.load(_state, map_location=device, weights_only=False)
        model.angle_head.load_state_dict(st['head']); opt.load_state_dict(st['opt']); sched.load_state_dict(st['sched'])
        start_epoch = st['epoch'] + 1; best_mae = st['best_mae']
        print(f"==> RESUMED angle from epoch {st['epoch']} -> {start_epoch}, best {best_mae:.2f} deg", flush=True)
    for epoch in range(start_epoch, args.epochs):
        model.angle_head.train()
        if model.angle_feat is not None:
            model.angle_feat.train()          # resnet BN uses batch stats during training
        run_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Ep{epoch} [train]")
        for batch in pbar:
            imgs = batch['image'].to(device)
            gt = batch['angles'].to(device).clone()
            if gt.shape[1] > 6: gt[:, 6] = 0.0            # Panda: fix joint7=0 (Meca has only 6 angles)
            has = batch['has_angles'].to(device).bool() if 'has_angles' in batch else torch.ones(len(imgs), dtype=torch.bool, device=device)
            K = scale_K(batch['camera_K'], batch['original_size'], args.image_size).to(device)

            if args.occlude_aug > 0:
                import sys as _s, os as _o
                _s.path.append(_o.path.join(_o.path.dirname(_o.path.abspath(__file__)), '../Eval'))
                from occl_util import paste_random_occluders_
                paste_random_occluders_(imgs, batch['keypoints'].numpy(), batch['valid_mask'].numpy(), args.occlude_aug)
            out_d = model(imgs, K, kp_jitter=args.kp_jitter, kp_drop=args.kp_drop)
            sc = out_d['sin_cos']                       # (B,6,2)
            gt6 = gt[:, :6]
            gt_sc = torch.stack([torch.sin(gt6), torch.cos(gt6)], dim=-1)
            _mixsel = bool(out_d.get('is_mixsel'))
            if _mixsel:
                # P3: MCL winner-take-all on the OBSERVABLE joints (exclude J5=idx4) so the K modes
                # split on the recoverable basin, + appearance-selector CE to the winner, + load-balance
                # (anti mode-collapse). At inference the selector (not the solver) picks the hypothesis.
                sc_all = out_d['sin_cos_all']; sel = out_d['sel_logits']      # (B,K,6,2),(B,K)
                _obs = [0, 1, 2, 3, 5]
                _dw = (sc_all[:, :, _obs, :] - gt_sc[:, None, _obs, :]).pow(2).sum(dim=(-1, -2))  # (B,K)
                winner = _dw.argmin(dim=1)                                    # (B,)
                _bi = torch.arange(sc_all.shape[0], device=device)
                sc_loss = F.smooth_l1_loss(sc_all[_bi, winner][has], gt_sc[has])
                sel_loss = F.cross_entropy(sel[has], winner[has].detach())
                _p = sel[has].softmax(-1).mean(0)                            # (K,)
                _lb = math.log(sel.shape[1]) + (_p * _p.clamp(min=1e-8).log()).sum()  # >=0, 0 at uniform
                loss = sc_loss + args.selector_weight * sel_loss + args.load_balance * _lb
            elif args.tail_gamma > 0 and epoch >= args.focal_warmup_epochs:
                # focal reweight by detached per-joint angular residual (EMA-normalized), clamped.
                with torch.no_grad():
                    ang_pred = out_d['joint_angles'][:, :6]
                    dj = torch.atan2(torch.sin(ang_pred - gt6), torch.cos(ang_pred - gt6)).abs()  # (B,6)
                    if has.any():
                        ema_dj.mul_(0.99).add_(0.01 * dj[has].mean(0))
                    w = focal_jw.view(1, 6) * (dj[has] / ema_dj.view(1, 6).clamp(min=1e-3)).pow(args.tail_gamma)
                    w = w.clamp(max=args.focal_clamp)
                per = F.smooth_l1_loss(sc[has], gt_sc[has], reduction='none').sum(-1)  # (Nhas,6)
                sc_loss = (w * per).sum() / w.sum().clamp(min=1e-6)
                loss = sc_loss
            elif args.head_type == 'ief' and 'sin_cos_iters' in out_d:
                # IEF deep supervision: L1 on EVERY iterate's sin/cos, later iterates weighted higher.
                it = out_d['sin_cos_iters']                         # (B,n_iter,6,2)
                w = torch.linspace(0.5, 1.0, it.shape[1], device=it.device)   # ramp toward final iterate
                per = F.smooth_l1_loss(it[has], gt_sc[has].unsqueeze(1).expand(-1, it.shape[1], -1, -1),
                                       reduction='none').mean(dim=(2, 3))       # (Nhas,n_iter)
                sc_loss = (per * w).sum(1).mean() / w.sum()
                loss = sc_loss
            else:
                sc_loss = F.smooth_l1_loss(sc[has], gt_sc[has])
                loss = sc_loss
            # FK robot-frame consistency (Panda/FR3 only; needs 7-angle panda FK). Skipped for
            # robots without a wired-up FK by passing --fk-weight 0 --reproj-weight 0.
            if not _mixsel and (args.fk_weight > 0 or args.reproj_weight > 0):
                fk_pred = fk_fn(out_d['joint_angles'])
                fk_gt = fk_fn(gt)
            if not _mixsel and args.fk_weight > 0:
                fk_loss = F.mse_loss(fk_pred[has], fk_gt[has])
                loss = loss + args.fk_weight * fk_loss
            # RoboTAG-style cross-dimensional (2D<->3D) consistency: project FK(pred_angles) through
            # the GT camera pose (Kabsch of FK(gt) onto camera-frame GT keypoints) and match GT 2D.
            # Adds the camera-frame reprojection signal the robot-frame fk_loss lacks — sharpens
            # angles where small errors move 2D (near cameras / azure, our RoboTAG-relative weakness).
            if not _mixsel and args.reproj_weight > 0 and 'keypoints_3d' in batch:
                kp3d = batch['keypoints_3d'].to(device)              # (B,7,3) camera frame
                kp2d = batch['keypoints'].to(device).float()        # (B,7,2) @ IS
                vm = batch['valid_mask'].to(device).float()         # (B,7)
                with torch.no_grad():
                    Rg, tg = kabsch_batch(fk_gt.detach(), kp3d)      # GT camera pose
                cam = torch.einsum('bij,bpj->bpi', Rg, fk_pred) + tg.unsqueeze(1)
                z = cam[..., 2].clamp(min=1e-3)
                u = cam[..., 0] / z * K[:, 0, 0:1] + K[:, 0, 2:3]
                v = cam[..., 1] / z * K[:, 1, 1:2] + K[:, 1, 2:3]
                proj = torch.stack([u, v], -1)
                valid_ok = ((kp3d.abs().sum(-1) > 1e-6) & (kp3d[..., 2] > 0)).float() * vm
                rp = (F.smooth_l1_loss(proj / args.image_size, kp2d / args.image_size,
                                       reduction='none').sum(-1) * valid_ok)[has]
                loss = loss + args.reproj_weight * rp.sum() / valid_ok[has].sum().clamp(min=1)

            opt.zero_grad(); loss.backward(); opt.step()
            run_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{opt.param_groups[0]['lr']:.1e}"})
        sched.step()

        # ---- validation: per-joint angle MAE (deg) ----
        model.angle_head.eval()
        if model.angle_feat is not None:
            model.angle_feat.eval()           # resnet BN uses running stats during val
        errs = []
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch['image'].to(device)
                gt = batch['angles'].to(device).clone()
                if gt.shape[1] > 6: gt[:, 6] = 0.0
                has = batch['has_angles'].bool() if 'has_angles' in batch else torch.ones(len(imgs), dtype=torch.bool)
                K = scale_K(batch['camera_K'], batch['original_size'], args.image_size).to(device)
                pred = model(imgs, K)['joint_angles']
                d = pred[:, :6] - gt[:, :6]
                d = torch.atan2(torch.sin(d), torch.cos(d)).abs() * 180 / math.pi
                errs.append(d[has.to(device)].cpu())
        errs = torch.cat(errs, dim=0)            # (M,6)
        per_joint = errs.mean(0)
        mae = per_joint.mean().item()
        print(f"Ep{epoch} | val angle MAE(J0-5)={mae:.2f} deg | per-joint=" +
              ",".join(f"{v:.1f}" for v in per_joint))

        if args.use_wandb and _HAS_WANDB:
            log = {'epoch': epoch, 'train_loss': run_loss / len(train_loader),
                   'val_angle_mae': mae, 'lr': opt.param_groups[0]['lr']}
            for j in range(6):
                log[f'val_mae_J{j}'] = per_joint[j].item()
            wandb.log(log)

        torch.save(model.angle_head.state_dict(), out / 'last_angle_head.pth')
        # P1b: when a separate trainable angle backbone exists (e.g. resnet50), its trained
        # weights ARE the experiment — persist them alongside the head or the ckpt is useless.
        if getattr(model, 'angle_feat', None) is not None:
            torch.save(model.angle_feat.state_dict(), out / 'last_angle_feat.pth')
        if mae < best_mae:
            best_mae = mae
            torch.save(model.angle_head.state_dict(), out / 'best_angle_head.pth')
            if getattr(model, 'angle_feat', None) is not None:
                torch.save(model.angle_feat.state_dict(), out / 'best_angle_feat.pth')
            print(f"  -> new best {best_mae:.2f} deg")
        if getattr(args, 'auto_resume', False):
            torch.save({'head': model.angle_head.state_dict(), 'opt': opt.state_dict(),
                        'sched': sched.state_dict(), 'epoch': epoch, 'best_mae': best_mae}, out / 'train_state.pt')
    print(f"Done. Best val angle MAE = {best_mae:.2f} deg")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--detector-ckpt', required=True, help='Stage-1 detector checkpoint (backbone+keypoint_head)')
    p.add_argument('--train-dir', required=True)
    p.add_argument('--val-dir', required=True)
    p.add_argument('--keypoint-names', default=None,
                   help='comma-separated (substring-matched). Meca500: link0,link1,link2,link3,link4,link5,link6')
    p.add_argument('--fk-robot', default='panda', choices=['panda', 'kuka', 'iiwa7', 'baxter', 'baxter_left'],
                   help='robot forward-kinematics for FK/reproj consistency loss')
    p.add_argument('--angle-joint-names', default=None,
                   help='comma-separated sim_state joint names for GT angles (order). '
                        'KUKA: iiwa7_joint_1,...,iiwa7_joint_7. Default None = sim_state[:7] (Panda).')
    p.add_argument('--output-dir', default='./outputs_angle')
    p.add_argument('--model-name', default='facebook/dinov3-vitb16-pretrain-lvd1689m')
    p.add_argument('--image-size', type=int, default=512)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--min-lr', type=float, default=1e-6)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--fk-weight', type=float, default=10.0)
    p.add_argument('--tail-gamma', type=float, default=0.0, help='P1 focal exponent on per-joint angular residual (0=off=uniform L2)')
    p.add_argument('--joint-weights', type=str, default='1,0.7,1,1,0.1,0.3', help='P1 per-joint mask J1..J6; idx4=J5~0 (observability write-off)')
    p.add_argument('--focal-warmup-epochs', type=int, default=3, help='plain-L2 epochs before focal kicks in (avoids noisy early residuals)')
    p.add_argument('--focal-clamp', type=float, default=5.0, help='max per-element focal weight (anti hard-negative-memorization)')
    p.add_argument('--n-mix', type=int, default=2, help='P3 mlp_mixsel: number of hypotheses/modes')
    p.add_argument('--selector-weight', type=float, default=1.0, help='P3: appearance-selector CE weight')
    p.add_argument('--load-balance', type=float, default=0.5, help='P3: mode load-balance (anti-collapse) weight')
    p.add_argument('--head-type', type=str, default='mlp', choices=['mlp', 'transformer', 'mlp_patch', 'mlp_mcl', 'mlp_mixsel', 'pare', 'ief'])
    p.add_argument('--angle-backbone', type=str, default='dino_frozen', choices=['dino_frozen', 'resnet50'],
                   help='P1b: feature source for the angle head. dino_frozen (default) = frozen DINOv3 pooled tokens (unchanged). '
                        'resnet50 = separate TRAINABLE ImageNet-init ResNet50 trunk (frozen DINOv3 stays kp2d/conf-only).')
    p.add_argument('--crop-to-robot', action='store_true',
                   help='crop image to robot bbox (train+test), RoboPEPP-style; must match detector ckpt')
    p.add_argument('--crop-margin', type=float, default=1.5)
    p.add_argument('--crop-aspect', type=float, default=1.0,
                   help='crop rect w/h. 1.0=legacy square. Set to the deploy frame aspect '
                        '(640x480 -> 1.3333) to match Eval/selfbbox_eval.py roi_align crops.')
    p.add_argument('--occlude-aug', type=float, default=0.0,
                   help='train-time occlusion augmentation: with prob 0.5 paste black occluders covering U(0.05,THIS) of the robot RoI (frozen detector -> head learns to handle degraded conf/keypoints)')
    p.add_argument('--kp-drop', type=float, default=0.0,
                   help='keypoint-level occlusion aug: randomly displace+deconfidence keypoints (model_angle.forward kp_drop)')
    p.add_argument('--reproj-weight', type=float, default=0.0,
                   help='RoboTAG-style camera-frame reprojection consistency (project FK(pred) via GT pose, match GT 2D) — adds the 2D<->3D alignment the robot-frame fk_loss lacks')
    p.add_argument('--init-head', default=None, help='warm-start angle_head from this state dict')
    p.add_argument('--kp-jitter', type=float, default=0.0,
                   help='train-time Gaussian px noise on detected 2D before geo/sampling (J0 noise-robustness)')
    p.add_argument('--num-workers', type=int, default=8)
    p.add_argument('--max-train', type=int, default=0, help='subsample training frames (0=all); for warm-start fine-tunes')
    p.add_argument('--max-val', type=int, default=0, help='subsample val frames (0=all)')
    p.add_argument('--use-wandb', action='store_true')
    p.add_argument('--wandb-project', default='dinov3-angle-predictor')
    p.add_argument('--wandb-run-name', default=None)
    p.add_argument('--auto-resume', action='store_true', help='resume head+opt+sched+epoch from <out>/train_state.pt')
    main(p.parse_args())
