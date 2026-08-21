"""
DINOv3 Heatmap-only Training Script
2D Keypoint Heatmap 학습 (체크포인트 로드 및 스케줄러 초기화 기능 포함)
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'src', 'dinobotpose'))
import argparse
import os
import time
import pickle
import random
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from datetime import timedelta

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from tqdm import tqdm
import yaml
import wandb
import cv2
from PIL import Image

from model import DINOv3Backbone, ViTKeypointHead, soft_argmax_2d
from dataset import PoseEstimationDataset
from norm_utils import get_norm_stats

def visualize_heatmaps(image_tensor, gt_heatmaps, pred_heatmaps, num_images=4):
    images_to_log = []
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(image_tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(image_tensor.device)
    num_to_show = min(image_tensor.shape[0], num_images)
    for i in range(num_to_show):
        img = image_tensor[i] * std + mean
        img = img.permute(1, 2, 0).cpu().numpy()
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        gt_hm = gt_heatmaps[i].sum(dim=0).cpu().numpy()
        gt_hm = np.clip(gt_hm * 255, 0, 255).astype(np.uint8)
        gt_hm_color = cv2.applyColorMap(gt_hm, cv2.COLORMAP_JET)
        pred_hm = pred_heatmaps[i].sum(dim=0).detach().cpu().numpy()
        pred_hm = np.clip(pred_hm * 255, 0, 255).astype(np.uint8)
        pred_hm_color = cv2.applyColorMap(pred_hm, cv2.COLORMAP_JET)
        gt_overlay = cv2.addWeighted(img, 0.6, gt_hm_color, 0.4, 0)
        pred_overlay = cv2.addWeighted(img, 0.6, pred_hm_color, 0.4, 0)
        combined = np.hstack([gt_overlay, pred_overlay])
        combined = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
        images_to_log.append(wandb.Image(combined, caption=f"GT vs Pred (Sample {i})"))
    return images_to_log

def keypoint_metrics(keypoints_detected, keypoints_gt, image_resolution, auc_pixel_threshold=20.0):
    kp_errors = []
    num_gt_inframe = 0
    for kp_proj_detect, kp_proj_gt in zip(keypoints_detected, keypoints_gt):
        if kp_proj_gt[0] >= 0 and kp_proj_gt[1] >= 0:
            num_gt_inframe += 1
            if kp_proj_detect[0] > -999:
                error = np.linalg.norm(kp_proj_detect - kp_proj_gt)
                kp_errors.append(error)
    
    kp_errors = np.array(kp_errors)
    results = {"num_gt_inframe": num_gt_inframe}
    
    if len(kp_errors) > 0:
        results["l2_error_mean_px"] = np.mean(kp_errors)
        
        # 🚀 PCK @ 2.5, 5, 10px 계산
        for thresh in [2.5, 5.0, 10.0]:
            pck = len(np.where(kp_errors < thresh)[0]) / max(1, num_gt_inframe)
            results[f"pck_{thresh}"] = pck
            
        # AUC 계산
        delta_pixel = 0.1
        pck_values = np.arange(0, auc_pixel_threshold, delta_pixel)
        y_values = [len(np.where(kp_errors < v)[0]) for v in pck_values]
        kp_auc = np.trapz(y_values, dx=delta_pixel) / auc_pixel_threshold / max(1, num_gt_inframe)
        results["l2_error_auc"] = kp_auc
    else:
        results["l2_error_mean_px"] = 512.0
        results["l2_error_auc"] = 0.0
        results["pck_2.5"] = 0.0
        results["pck_5.0"] = 0.0
        results["pck_10.0"] = 0.0
        
    return results

class HeatmapModel(nn.Module):
    def __init__(self, model_name, heatmap_size, unfreeze_blocks=2, pretrained=True, num_joints=7):
        super().__init__()
        self.backbone = DINOv3Backbone(model_name, unfreeze_blocks=unfreeze_blocks, pretrained=pretrained)
        if "siglip" in model_name:
            cfg = self.backbone.model.config
            vcfg = getattr(cfg, "vision_config", cfg)  # Siglip2Model -> vision_config.hidden_size
            feature_dim = vcfg.hidden_size
        else:
            config = self.backbone.model.config
            feature_dim = config.hidden_sizes[-1] if "conv" in model_name else config.hidden_size
        self.keypoint_head = ViTKeypointHead(input_dim=feature_dim, num_joints=num_joints, heatmap_size=heatmap_size)
    def forward(self, x):
        features = self.backbone(x)
        return self.keypoint_head(features)

def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main(args):
    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    if local_rank != -1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', timeout=timedelta(minutes=30))
        device = torch.device(f'cuda:{local_rank}')
        rank = dist.get_rank()
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        rank = 0
    is_main, output_dir = rank == 0, Path(args.output_dir)
    random.seed(args.seed + rank); np.random.seed(args.seed + rank); torch.manual_seed(args.seed + rank)

    keypoint_names = (args.keypoint_names.split(',') if getattr(args, 'keypoint_names', None)
                      else ['link0', 'link2', 'link3', 'link4', 'link6', 'link7', 'hand'])
    # Normalization is backbone-specific (read from the model's own AutoImageProcessor):
    # DINOv3=ImageNet, SigLIP2/google-vit=0.5, random-init=0.5. Prevents train/eval mismatch.
    norm_mean, norm_std = get_norm_stats(args.model_name, random_init=args.random_init, verbose=is_main)
    full_train_dataset = PoseEstimationDataset(
        data_dir=args.data_dir[0], keypoint_names=keypoint_names,
        image_size=(args.image_size, args.image_size), heatmap_size=(args.heatmap_size, args.heatmap_size),
        augment=not args.no_augment, fda_real_dir=args.fda_real_dir, fda_prob=args.fda_prob, fda_beta=args.fda_beta,
        occlusion_prob=args.occlusion_prob, occlusion_max_size_frac=args.occlusion_size,
        aug_level=args.aug_level,
        norm_mean=norm_mean, norm_std=norm_std,
        crop_to_robot=args.crop_to_robot, crop_margin=args.crop_margin,
        crop_aspect=args.crop_aspect, crop_aspect_jitter=args.crop_aspect_jitter,
        crop_res_jitter=args.crop_res_jitter,
        crop_res_range=(args.crop_res_min, args.image_size),
        distractor_occ_prob=args.distractor_occ_prob,
        distractor_occ_max_frac=args.distractor_occ_max_frac,
        frame_boundary_prob=args.frame_boundary_prob,
        sigma=2.5
    )
    if args.val_dir:
        val_dataset = PoseEstimationDataset(
            data_dir=args.val_dir, keypoint_names=keypoint_names,
            image_size=(args.image_size, args.image_size), heatmap_size=(args.heatmap_size, args.heatmap_size),
            augment=False, norm_mean=norm_mean, norm_std=norm_std,
            crop_to_robot=args.crop_to_robot, crop_margin=args.crop_margin,
            crop_aspect=args.crop_aspect, sigma=2.5   # val: 지터 없이 배포 기하 그대로
        )
        train_dataset = full_train_dataset
    else:
        val_size = int(len(full_train_dataset) * args.val_split)
        train_dataset, val_dataset = random_split(full_train_dataset, [len(full_train_dataset) - val_size, val_size])

    if local_rank != -1:
        train_sampler, val_sampler = DistributedSampler(train_dataset), DistributedSampler(val_dataset, shuffle=False)
    else: train_sampler = val_sampler = None
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, shuffle=(train_sampler is None), num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, num_workers=args.num_workers, pin_memory=True)

    # Model 초기화
    model = HeatmapModel(args.model_name, (args.heatmap_size, args.heatmap_size), args.unfreeze_blocks,
                         pretrained=not args.random_init, num_joints=len(keypoint_names)).to(device)
    
    # 체크포인트 로드 (가중치만 불러오고 옵티마이저/스케줄러는 초기화)
    if args.checkpoint and os.path.isfile(args.checkpoint):
        if is_main:
            print(f"==> Loading weights from checkpoint: {args.checkpoint}")
        
        checkpoint = torch.load(args.checkpoint, map_location=device)
        
        # 현재 모델의 파라미터 상태 가져오기
        model_state_dict = model.state_dict()
        
        # 'module.' 접두사 제거 (만약 DDP로 저장된 체크포인트인 경우)
        checkpoint_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}
        
        # 🚀 [핵심] 가중치 필터링: 현재 모델에 해당 키가 존재하고, 텐서 모양(shape)이 완벽히 똑같은 경우만 복사!
        filtered_dict = {
            k: v for k, v in checkpoint_dict.items() 
            if k in model_state_dict and v.shape == model_state_dict[k].shape
        }
        
        if is_main:
            print(f"==> Successfully matched {len(filtered_dict)} out of {len(model_state_dict)} layers.")
            missed_keys = set(model_state_dict.keys()) - set(filtered_dict.keys())
            if missed_keys:
                print(f"==> Randomly initializing {len(missed_keys)} new or modified layers (e.g., Large Kernels, FiLM).")
        
        # strict=False로 설정하여 누락된 키가 있어도 에러 없이 넘어가게 로드
        model.load_state_dict(filtered_dict, strict=False)
    
    if local_rank != -1:
        # random-init trains the whole ViT incl. genuinely-unused params (pooler/mask_token that
        # never touch patch-token loss) -> DDP needs find_unused_parameters. Frozen/fine-tune runs
        # freeze those params so they're excluded from reduction and don't need the flag.
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], find_unused_parameters=args.random_init)
    
    # Per-keypoint MSE loss weights. Default = legacy Panda U-shape (base+hand up-weighted).
    # --joint-weights overrides to attack the empirical catastrophic-argmax tail, which is
    # concentrated on DISTAL joints (link6 9.24% / hand 9.09% / link7 7.95% / link3 6.02%)
    # that the legacy U-shape under-weights. Backward-compatible: None -> identical old behavior.
    if getattr(args, 'joint_weights', None):
        jw = [float(x) for x in args.joint_weights.split(',')]
        assert len(jw) == len(keypoint_names), \
            f"--joint-weights has {len(jw)} values but there are {len(keypoint_names)} keypoints"
        joint_weights = torch.tensor(jw).to(device)
        if is_main:
            print(f"==> Per-keypoint loss weights (custom): {dict(zip(keypoint_names, jw))}")
    elif len(keypoint_names) == 7:
        joint_weights = torch.tensor([2.5, 1.5, 1.3, 1.0, 1.3, 1.5, 2.5]).to(device)
    else:
        # Non-7-keypoint robots (e.g. whole-body Baxter, 17): uniform weighting by default.
        joint_weights = torch.ones(len(keypoint_names)).to(device)
    criterion = nn.MSELoss(reduction='none')
    loss_scale = 1000.0
    # Two LR groups: head at --learning-rate, unfrozen backbone blocks at --backbone-lr (lower).
    backbone_params, head_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (backbone_params if '.backbone.' in n or n.startswith('backbone.') else head_params).append(p)
    bb_lr = args.backbone_lr if args.backbone_lr is not None else args.learning_rate
    param_groups = [{'params': head_params, 'lr': args.learning_rate}]
    if backbone_params:
        param_groups.append({'params': backbone_params, 'lr': bb_lr})
        if is_main:
            print(f"==> Optimizer: {len(head_params)} head tensors @ lr={args.learning_rate}, "
                  f"{len(backbone_params)} backbone tensors @ lr={bb_lr}")
    optimizer = optim.AdamW(param_groups, lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)

    if is_main:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=args)
        output_dir.mkdir(parents=True, exist_ok=True)

    best_val_auc = -1.0
    global_step = 0
    warmup_steps = 100
    start_epoch = 0
    state_path = output_dir / 'training_state.pt'
    if getattr(args, 'auto_resume', False) and state_path.is_file():
        st = torch.load(state_path, map_location=device, weights_only=False)  # our own trusted state (has optimizer + py scalars)
        (model.module if hasattr(model, 'module') else model).load_state_dict(st['model'])
        optimizer.load_state_dict(st['optimizer']); scheduler.load_state_dict(st['scheduler'])
        start_epoch = st.get('resume_epoch', st['epoch'] + 1); best_val_auc = st['best_val_auc']; global_step = st['global_step']
        if is_main:
            print(f"==> RESUMED from epoch {st['epoch']} -> start {start_epoch}, best_auc {best_val_auc:.4f}", flush=True)

    for epoch in range(start_epoch, args.epochs):
        if train_sampler: train_sampler.set_epoch(epoch)
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]") if is_main else train_loader
        train_loss_accum = 0.0
        train_joint_losses = torch.zeros(len(keypoint_names)).to(device)
        for batch in pbar:
            if global_step < warmup_steps:
                warmup_lr = args.min_lr + (args.learning_rate - args.min_lr) * (global_step / warmup_steps)
                set_lr(optimizer, warmup_lr)
            imgs, gt_hms = batch['image'].to(device), batch['heatmaps'].to(device)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=args.amp):
                preds = model(imgs)
                raw_loss = criterion(preds, gt_hms).mean(dim=[2, 3])
                loss = (raw_loss * joint_weights.view(1, -1)).mean() * loss_scale
            train_joint_losses += raw_loss.mean(dim=0).detach() * loss_scale
            optimizer.zero_grad(); loss.backward(); optimizer.step()  # bf16 autocast: no GradScaler needed
            train_loss_accum += loss.item()
            global_step += 1
            if is_main: pbar.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{optimizer.param_groups[0]['lr']:.2e}"})
            # mid-epoch state save (survives flaky-GPU kills that land inside an epoch); resume_epoch=epoch
            # means "redo the current epoch with these weights" — preserves all weight/opt progress.
            if getattr(args, 'ckpt_every', 0) and is_main and global_step % args.ckpt_every == 0:
                _sm = model.module if hasattr(model, 'module') else model
                torch.save({'model': _sm.state_dict(), 'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(), 'epoch': epoch, 'resume_epoch': epoch,
                            'best_val_auc': float(best_val_auc), 'global_step': global_step},
                           output_dir / 'training_state.pt')

        train_joint_losses /= len(train_loader)
        model.eval(); val_loss, all_preds, all_gts, viz_batch = 0.0, [], [], None
        val_joint_losses = torch.zeros(len(keypoint_names)).to(device)
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                imgs, gt_hms = batch['image'].to(device), batch['heatmaps'].to(device)
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=args.amp):
                    preds = model(imgs)
                    raw_val_loss = criterion(preds, gt_hms).mean(dim=[2, 3])
                    weighted_val_loss = (raw_val_loss * joint_weights.view(1, -1)).mean() * loss_scale
                preds = preds.float()  # bf16->fp32 for argmax decode / viz (numpy has no bf16)
                val_joint_losses += raw_val_loss.mean(dim=0).detach() * loss_scale
                val_loss += weighted_val_loss.item()
                if i == 0: viz_batch = (imgs, gt_hms, preds)
                
                # 🚀 [수정] soft-argmax를 지우고 원래의 argmax 코드로 원상복구!
                B, N, H, W = preds.shape
                max_idx = preds.view(B, N, -1).argmax(dim=-1)
                pred_coords = torch.stack([max_idx % W, max_idx // W], dim=-1).cpu().numpy()
                
                all_preds.append(pred_coords)
                all_gts.append(batch['keypoints'].numpy())

        val_loss /= len(val_loader) if len(val_loader) > 0 else 1
        val_joint_losses /= len(val_loader) if len(val_loader) > 0 else 1
        if len(all_preds) > 0:
            metrics = keypoint_metrics(np.concatenate(all_preds).reshape(-1, 2), np.concatenate(all_gts).reshape(-1, 2), (args.heatmap_size, args.heatmap_size))
            current_auc, current_l2 = metrics['l2_error_auc'], metrics['l2_error_mean_px']
        else: current_auc, current_l2 = 0.0, 512.0

        if is_main:
            # 🚀 [수정됨] 터미널에 상세 지표(PCK@2.5, 5, 10) 출력
            print(f"Epoch {epoch} | Val Loss: {val_loss:.4f} | AUC: {current_auc:.4f} | L2: {current_l2:.2f}px")
            print(f"PCK@2.5: {metrics.get('pck_2.5', 0):.4f} | PCK@5: {metrics.get('pck_5.0', 0):.4f} | PCK@10: {metrics.get('pck_10.0', 0):.4f}")
            
            log_dict = {'epoch': epoch, 'train_loss_total': train_loss_accum/len(train_loader), 'val_loss_total': val_loss, 'val_auc': current_auc, 'val_l2_px': current_l2, 'learning_rate': optimizer.param_groups[0]['lr']}
            for idx, name in enumerate(keypoint_names):
                log_dict[f'train_joint_loss/{name}'] = train_joint_losses[idx].item()
                log_dict[f'val_joint_loss/{name}'] = val_joint_losses[idx].item()
            if viz_batch is not None: log_dict['visualizations'] = visualize_heatmaps(*viz_batch, num_images=4)
            wandb.log(log_dict)
            save_model = model.module if hasattr(model, 'module') else model
            if current_auc > best_val_auc: best_val_auc = current_auc; torch.save(save_model.state_dict(), output_dir / 'best_heatmap.pth')
            torch.save(save_model.state_dict(), output_dir / 'last_heatmap.pth')
            if getattr(args, 'auto_resume', False):
                torch.save({'model': save_model.state_dict(), 'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(), 'epoch': epoch, 'resume_epoch': epoch + 1,
                            'best_val_auc': float(best_val_auc), 'global_step': global_step},
                           output_dir / 'training_state.pt')
        if global_step >= warmup_steps: scheduler.step()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True, nargs='+')
    parser.add_argument('--val-dir', type=str, default=None)
    parser.add_argument('--val-split', type=float, default=0.2)
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to weights checkpoint to load')
    parser.add_argument('--model-name', type=str, default='facebook/dinov3-vitb16-pretrain-lvd1689m')
    parser.add_argument('--random-init', action='store_true',
                        help='from-scratch: random-init backbone (same arch, no pretrained weights); full backbone trains')
    parser.add_argument('--amp', action='store_true',
                        help='bf16 mixed-precision training (~1.6-2x faster). Default off=fp32. Use uniformly across compared cells.')
    parser.add_argument('--output-dir', type=str, default='./outputs_heatmap')
    parser.add_argument('--image-size', type=int, default=512)
    parser.add_argument('--heatmap-size', type=int, default=512)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--backbone-lr', type=float, default=None, help='LR for unfrozen backbone blocks (default = learning-rate)')
    parser.add_argument('--min-lr', type=float, default=1e-7)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--unfreeze-blocks', type=int, default=2)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--auto-resume', action='store_true', help='resume model+optimizer+scheduler+epoch from <output-dir>/training_state.pt if present (per-epoch checkpoint; survives GPU kills)')
    parser.add_argument('--ckpt-every', type=int, default=0, help='also save training_state.pt every N optimizer steps (mid-epoch; bounds work lost to a flaky-GPU kill). 0 = per-epoch only')
    parser.add_argument('--no-augment', action='store_true', help='Disable general data augmentation')
    parser.add_argument('--aug-level', type=str, default='light', choices=['light', 'strong', 'strong_vp'], help='Augmentation strength (strong = Stage-1 sim-to-real; strong_vp = strong + wider rotation/perspective/shear for diverse-viewpoint splits)')
    parser.add_argument('--occlusion-prob', type=float, default=0.5, help='Probability of occlusion augmentation')
    parser.add_argument('--occlusion-size', type=float, default=0.2, help='Max size of occlusion patch relative to image')
    parser.add_argument('--crop-to-robot', action='store_true', help='square-crop around robot (GT-kp bbox) before resize')
    parser.add_argument('--crop-margin', type=float, default=1.5, help='bbox expansion factor for crop')
    parser.add_argument('--crop-aspect', type=float, default=1.0,
                        help='crop rect w/h. 1.0=legacy square. Set to the deploy frame aspect '
                             '(640x480 -> 1.3333) to match Eval/selfbbox_eval.py, which crops a '
                             'square out of the anisotropically-resized 512x512 frame (= 4:3 in original px).')
    parser.add_argument('--crop-aspect-jitter', type=float, default=0.0,
                        help='log-uniform multiplicative jitter on --crop-aspect (train only)')
    parser.add_argument('--crop-res-jitter', type=float, default=0.0,
                        help='probability of resampling the crop to a lower effective resolution '
                             '(simulates the small-robot upsampled-crop regime)')
    parser.add_argument('--crop-res-min', type=int, default=140,
                        help='lower bound of the --crop-res-jitter target side in px')
    # det_C (2026-07-31) occlusion-realistic + frame-boundary detector aug (opt-in; 0.0 = deploy-invariant)
    parser.add_argument('--distractor-occ-prob', type=float, default=0.0,
                        help='per-sample prob of pasting REALISTIC distractor occluders (image-crop textures) over the robot')
    parser.add_argument('--distractor-occ-max-frac', type=float, default=0.35,
                        help='max distractor occluder size relative to the robot keypoint-bbox diagonal')
    parser.add_argument('--frame-boundary-prob', type=float, default=0.0,
                        help='per-sample prob of an off-center/shrunk crop that pushes part of the robot OFF-FRAME (teaches the detector to suppress off-frame joints, not hallucinate)')
    parser.add_argument('--fda-real-dir', type=str, default=None)
    parser.add_argument('--fda-prob', type=float, default=0.0)
    parser.add_argument('--fda-beta', type=float, default=0.05)
    parser.add_argument('--wandb-project', type=str, default='dinov3-heatmap-only')
    parser.add_argument('--wandb-run-name', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--joint-weights', type=str, default=None,
                        help='comma-separated per-keypoint MSE loss weights (must match keypoint '
                             'count). Default = legacy Panda U-shape 2.5,1.5,1.3,1.0,1.3,1.5,2.5. '
                             'Up-weight distal (link3/6/7/hand) to attack the catastrophic argmax tail.')
    parser.add_argument('--keypoint-names', type=str, default=None,
                        help='comma-separated keypoint suffixes (default = Panda 7). Meca500/FR5 6-DOF: link0,link1,link2,link3,link4,link5,link6')
    main(parser.parse_args())
