"""
DINOv3 Pose Estimation Training Script
DREAM 학습 방식을 참고한 학습 코드
"""

import argparse
import os
import time
import json
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import timedelta

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from tqdm import tqdm
import yaml
import wandb

from model import DINOv3PoseEstimator
from dataset import PoseEstimationDataset, create_dataloaders

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../DREAM')))
import dream
from dream import analysis as dream_analysis

def get_keypoints_from_heatmaps(heatmaps):
    """
    Extract keypoint coordinates from heatmaps using argmax.
    heatmaps: (B, N, H, W)
    Returns: (B, N, 2) [x, y] coordinates
    """
    B, N, H, W = heatmaps.shape
    heatmaps_flat = heatmaps.view(B, N, -1)
    max_indices = torch.argmax(heatmaps_flat, dim=-1)
    
    y = max_indices // W
    x = max_indices % W
    
    return torch.stack([x, y], dim=-1).float()


class PoseLoss(nn.Module):
    """
    Multi-task loss for pose estimation
    - Heatmap loss (MSE)
    - Joint angle loss (MSE or L1)
    """

    def __init__(
        self,
        heatmap_weight: float = 1.0,
        angle_weight: float = 1.0,  # Increased weight for angles
        angle_loss_type: str = 'smooth_l1'
    ):
        super().__init__()
        self.heatmap_weight = heatmap_weight
        self.angle_weight = angle_weight

        self.heatmap_loss = nn.MSELoss()

        if angle_loss_type == 'mse':
            self.angle_loss = nn.MSELoss()
        elif angle_loss_type == 'l1':
            self.angle_loss = nn.L1Loss()
        elif angle_loss_type == 'smooth_l1':
            self.angle_loss = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unknown angle loss type: {angle_loss_type}")

    def forward(self, pred_heatmaps, pred_angles, gt_heatmaps, gt_angles):
        """
        Args:
            pred_heatmaps: (B, N, H, W) predicted heatmaps
            pred_angles: (B, A) predicted joint angles
            gt_heatmaps: (B, N, H, W) ground truth heatmaps
            gt_angles: (B, A) ground truth joint angles

        Returns:
            total_loss, loss_dict
        """
        # Heatmap loss
        loss_heatmap = self.heatmap_loss(pred_heatmaps, gt_heatmaps)

        # Angle loss
        loss_angle = self.angle_loss(pred_angles, gt_angles)

        # Total loss
        total_loss = (
            self.heatmap_weight * loss_heatmap +
            self.angle_weight * loss_angle
        )

        loss_dict = {
            'total': total_loss.item(),
            'heatmap': loss_heatmap.item(),
            'angle': loss_angle.item()
        }

        return total_loss, loss_dict


class Trainer:
    """Training manager for DINOv3 pose estimation"""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[optim.lr_scheduler._LRScheduler],
        device: torch.device,
        output_dir: str,
        config: Dict,
        camera_K: Optional[np.ndarray] = None,
        raw_res: Tuple[int, int] = (640, 480),
        resume_from: Optional[str] = None,
        local_rank: int = -1
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.output_dir = Path(output_dir)
        self.config = config
        self.camera_K = camera_K
        self.raw_res = raw_res
        self.local_rank = local_rank
        self.is_distributed = local_rank != -1
        self.is_main_process = (not self.is_distributed) or (local_rank == 0)

        # Create output directory (only on main process)
        if self.is_main_process:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize wandb (only on main process)
        if self.is_main_process:
            try:
                # Create a unique run name if not specified
                run_name = config.get('wandb_run_name', None)
                if run_name is None:
                    run_name = f"{config.get('model_name', 'dinov3').split('/')[-1]}_{self.output_dir.name}"

                wandb.init(
                    project=config.get('wandb_project', 'dinov3-pose-estimation'),
                    name=run_name,
                    config=config,
                    dir=str(self.output_dir),
                    resume='allow' if resume_from else None,
                    reinit=False  # Prevent reinitialization
                )
                print(f"WandB initialized successfully: {wandb.run.name}")
                print(f"WandB URL: {wandb.run.get_url()}")
            except Exception as e:
                print(f"Warning: Failed to initialize WandB: {e}")
                print("Training will continue without WandB logging")

        # Training state
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        self.train_log = {
            'epochs': [],
            'train_losses': [],
            'val_losses': [],
            'learning_rates': [],
            'timestamps': []
        }

        # Resume from checkpoint
        if resume_from:
            self._load_checkpoint(resume_from)

        # Save config (only on main process)
        if self.is_main_process:
            with open(self.output_dir / 'config.yaml', 'w') as f:
                yaml.dump(config, f)

    def _load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint for resuming training"""
        if self.is_main_process:
            print(f"Loading checkpoint from {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load model state (handle DDP wrapper)
        state_dict = checkpoint['model_state_dict']
        if self.is_distributed and not any(k.startswith('module.') for k in state_dict.keys()):
            # Add 'module.' prefix if loading non-DDP checkpoint into DDP model
            state_dict = {'module.' + k: v for k, v in state_dict.items()}
        elif not self.is_distributed and any(k.startswith('module.') for k in state_dict.keys()):
            # Remove 'module.' prefix if loading DDP checkpoint into non-DDP model
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        self.model.load_state_dict(state_dict)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if self.is_main_process:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Scheduler state restored. Current LR: {current_lr:.2e}")

        self.start_epoch = checkpoint.get('epoch', 0) + 1
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.train_log = checkpoint.get('train_log', self.train_log)

        if self.is_main_process:
            print(f"Resumed from epoch {self.start_epoch}")
            print(f"Best validation loss so far: {self.best_val_loss:.4f}")

    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save training checkpoint (only on main process)"""
        if not self.is_main_process:
            return

        # Get model state dict (unwrap DDP if needed)
        model_state_dict = self.model.module.state_dict() if self.is_distributed else self.model.state_dict()

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_log': self.train_log,
            'config': self.config
        }

        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        # Save current epoch checkpoint
        checkpoint_path = self.output_dir / f'epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

        # Save best model
        if is_best:
            best_path = self.output_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"Saved best model: {best_path}")

        # Maintain last 3 checkpoints
        all_ckpts = sorted(list(self.output_dir.glob('epoch_*.pth')), key=lambda x: int(x.stem.split('_')[1]))
        if len(all_ckpts) > 3:
            for old_ckpt in all_ckpts[:-3]:
                try:
                    old_ckpt.unlink()
                    print(f"Removed old checkpoint: {old_ckpt}")
                except Exception as e:
                    print(f"Error removing old checkpoint {old_ckpt}: {e}")

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()

        # Set sampler epoch for proper shuffling in distributed training
        if self.is_distributed and hasattr(self.train_loader.sampler, 'set_epoch'):
            self.train_loader.sampler.set_epoch(epoch)

        epoch_losses = {
            'total': [],
            'heatmap': [],
            'angle': []
        }

        # Only show progress bar on main process
        if self.is_main_process:
            pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [Train]')
        else:
            pbar = self.train_loader

        batch_idx = 0
        for batch in pbar:
            batch_idx += 1
            # Move data to device
            images = batch['image'].to(self.device)
            gt_heatmaps = batch['heatmaps'].to(self.device)
            gt_angles = batch['angles'].to(self.device)

            # Forward pass
            pred_heatmaps, pred_angles = self.model(images)

            # Compute loss
            loss, loss_dict = self.criterion(
                pred_heatmaps, pred_angles,
                gt_heatmaps, gt_angles
            )

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Record losses
            for key, value in loss_dict.items():
                epoch_losses[key].append(value)

            # Update progress bar (only on main process)
            if self.is_main_process:
                pbar.set_postfix({
                    'loss': f"{loss_dict['total']:.6f}",
                    'hm': f"{loss_dict['heatmap']:.6f}",
                    'ang': f"{loss_dict['angle']:.6f}"
                })

                # Log batch losses to wandb every N batches
                if wandb.run is not None and batch_idx % 10 == 0:
                    try:
                        global_step = epoch * len(self.train_loader) + batch_idx
                        wandb.log({
                            'batch/train_loss': loss_dict['total'],
                            'batch/train_heatmap_loss': loss_dict['heatmap'],
                            'batch/train_angle_loss': loss_dict['angle'],
                        }, step=global_step)
                    except Exception:
                        pass  # Silently ignore batch logging errors

        # Average losses
        avg_losses = {key: np.mean(values) for key, values in epoch_losses.items()}

        # Synchronize losses across processes in distributed training
        if self.is_distributed:
            for key in avg_losses:
                avg_tensor = torch.tensor(avg_losses[key], device=self.device)
                dist.all_reduce(avg_tensor, op=dist.ReduceOp.SUM)
                avg_losses[key] = (avg_tensor / dist.get_world_size()).item()

        return avg_losses

    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()

        epoch_losses = {
            'total': [],
            'heatmap': [],
            'angle': []
        }
        
        # Metrics collection
        all_kp_projs_est = []
        all_kp_projs_gt = []
        all_kp_pos_gt = []

        # Only show progress bar on main process
        if self.is_main_process:
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch} [Val]')
        else:
            pbar = self.val_loader

        for batch in pbar:
            # Move data to device
            images = batch['image'].to(self.device)
            gt_heatmaps = batch['heatmaps'].to(self.device)
            gt_angles = batch['angles'].to(self.device)

            # Forward pass
            pred_heatmaps, pred_angles = self.model(images)

            # Compute loss
            _, loss_dict = self.criterion(
                pred_heatmaps, pred_angles,
                gt_heatmaps, gt_angles
            )

            # Record losses
            for key, value in loss_dict.items():
                epoch_losses[key].append(value)
                
            # Collect data for metrics
            if self.camera_K is not None:
                # Extract coordinates
                pred_kps = get_keypoints_from_heatmaps(pred_heatmaps).cpu().numpy()
                gt_kps = batch['keypoints'].numpy()
                gt_kps_3d = batch['keypoints_3d'].numpy()
                
                # Rescale to raw resolution
                h_size = self.config.get('heatmap_size', 512)
                scale_x = self.raw_res[0] / h_size
                scale_y = self.raw_res[1] / h_size
                
                for i in range(len(pred_kps)):
                    p_kps = pred_kps[i].copy()
                    g_kps = gt_kps[i].copy()
                    p_kps[:, 0] *= scale_x
                    p_kps[:, 1] *= scale_y
                    g_kps[:, 0] *= scale_x
                    g_kps[:, 1] *= scale_y
                    
                    all_kp_projs_est.append(p_kps)
                    all_kp_projs_gt.append(g_kps)
                    all_kp_pos_gt.append(gt_kps_3d[i])

            # Update progress bar (only on main process)
            if self.is_main_process:
                pbar.set_postfix({
                    'loss': f"{loss_dict['total']:.6f}",
                    'hm': f"{loss_dict['heatmap']:.6f}",
                    'ang': f"{loss_dict['angle']:.6f}"
                })

        # Average losses
        avg_losses = {key: np.mean(values) for key, values in epoch_losses.items()}

        # Synchronize basic losses across processes in distributed training FIRST
        # This prevents NCCL timeout when rank 0 computes expensive metrics
        if self.is_distributed:
            for key in list(avg_losses.keys()):
                avg_tensor = torch.tensor(avg_losses[key], device=self.device)
                dist.all_reduce(avg_tensor, op=dist.ReduceOp.SUM)
                avg_losses[key] = (avg_tensor / dist.get_world_size()).item()

        # Compute PCK and ADD if enough data collected (only on main process, AFTER sync)
        if self.is_main_process and self.camera_K is not None and len(all_kp_projs_est) > 0:
            all_kp_projs_est_np = np.array(all_kp_projs_est)
            all_kp_projs_gt_np = np.array(all_kp_projs_gt)
            all_kp_pos_gt_np = np.array(all_kp_pos_gt)

            # 2D PCK / AUC
            kp_metrics = dream_analysis.keypoint_metrics(
                all_kp_projs_est_np.reshape(-1, 2),
                all_kp_projs_gt_np.reshape(-1, 2),
                self.raw_res
            )
            avg_losses['val_pck_auc'] = kp_metrics['l2_error_auc']
            avg_losses['val_kp_mean_err_px'] = kp_metrics['l2_error_mean_px']

            # 3D ADD (PnP)
            pnp_adds = []
            n_inframe_list = []

            # Limit PnP calculation to 50 samples to keep validation fast
            subset_size = min(50, len(all_kp_projs_est))
            subset_indices = np.random.choice(len(all_kp_projs_est), subset_size, replace=False)

            for idx in subset_indices:
                p_est = all_kp_projs_est[idx]
                p_gt = all_kp_projs_gt[idx]
                pos_gt = all_kp_pos_gt[idx]

                n_inframe = 0
                for pt in p_gt:
                    if 0 <= pt[0] < self.raw_res[0] and 0 <= pt[1] < self.raw_res[1]:
                        n_inframe += 1
                n_inframe_list.append(n_inframe)

                pnp_retval, translation, quaternion = dream.geometric_vision.solve_pnp(
                    pos_gt, p_est, self.camera_K
                )
                if pnp_retval:
                    add = dream.geometric_vision.add_from_pose(
                        translation, quaternion, pos_gt, self.camera_K
                    )
                    # Convert to mm
                    pnp_adds.append(add * 1000.0)
                else:
                    pnp_adds.append(-999.0)

            pnp_metrics = dream_analysis.pnp_metrics(pnp_adds, n_inframe_list, add_auc_threshold=100.0) # 100mm threshold
            avg_losses['val_add_auc'] = pnp_metrics['add_auc']
            avg_losses['val_add_mean_mm'] = pnp_metrics['add_mean']
            avg_losses['val_pnp_success_rate'] = pnp_metrics['num_pnp_found'] / max(1, pnp_metrics['num_pnp_possible'])

        return avg_losses

    def train(self, num_epochs: int):
        """Main training loop"""
        if self.is_main_process:
            print("\n" + "=" * 80)
            print("Starting Training")
            print("=" * 80 + "\n")

        start_time = time.time()

        for epoch in range(self.start_epoch, num_epochs):
            epoch_start = time.time()

            # Train
            train_losses = self.train_epoch(epoch)

            # Validate
            val_losses = self.validate(epoch)

            # Synchronize all processes after validation
            if self.is_distributed:
                dist.barrier()

            # Update learning rate
            if self.scheduler:
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]['lr']

            # Log to wandb (only on main process)
            if self.is_main_process and wandb.run is not None:
                try:
                    log_dict = {
                        'epoch': epoch,
                        'train/total_loss': train_losses['total'],
                        'train/heatmap_loss': train_losses['heatmap'],
                        'train/angle_loss': train_losses['angle'],
                        'learning_rate': current_lr,
                        'best_val_loss': self.best_val_loss
                    }
                    # Add all validation metrics
                    for k, v in val_losses.items():
                        log_dict[f'val/{k}'] = v
                        
                    wandb.log(log_dict)
                except Exception as e:
                    print(f"Warning: Failed to log to WandB: {e}")

            # Print epoch summary (only on main process)
            if self.is_main_process:
                epoch_time = time.time() - epoch_start
                print(f"\nEpoch {epoch} Summary:")
                print(f"  Train Loss: {train_losses['total']:.6f} "
                      f"(heatmap: {train_losses['heatmap']:.6f}, angle: {train_losses['angle']:.6f})")
                print(f"  Val Loss:   {val_losses['total']:.6f} "
                      f"(heatmap: {val_losses['heatmap']:.6f}, angle: {val_losses['angle']:.6f})")
                if 'val_pck_auc' in val_losses:
                    print(f"  Val PCK AUC: {val_losses['val_pck_auc']:.4f}, ADD AUC: {val_losses['val_add_auc']:.4f}")
                    print(f"  Val ADD Mean: {val_losses['val_add_mean_mm']:.2f}mm, PnP Succ: {val_losses['val_pnp_success_rate']:.2%}")
                print(f"  Learning Rate: {current_lr:.8f}")
                print(f"  Time: {epoch_time:.2f}s")

            # Save checkpoint (only on main process)
            is_best = val_losses['total'] < self.best_val_loss
            if is_best:
                if self.is_main_process:
                    print(f"  New best model! (previous: {self.best_val_loss:.4f})")
                self.best_val_loss = val_losses['total']

            self._save_checkpoint(epoch, is_best=is_best)

            # Update training log (only on main process)
            if self.is_main_process:
                self.train_log['epochs'].append(epoch)
                self.train_log['train_losses'].append(train_losses)
                self.train_log['val_losses'].append(val_losses)
                self.train_log['learning_rates'].append(current_lr)
                self.train_log['timestamps'].append(time.time() - start_time)

                # Save training log
                with open(self.output_dir / 'training_log.pkl', 'wb') as f:
                    pickle.dump(self.train_log, f)

                print()

            # Synchronize all processes after checkpoint saving
            # This ensures rank 0 finishes saving before other ranks start next epoch
            if self.is_distributed:
                dist.barrier()

        if self.is_main_process:
            total_time = time.time() - start_time
            print("=" * 80)
            print(f"Training completed in {total_time / 3600:.2f} hours")
            print(f"Best validation loss: {self.best_val_loss:.4f}")
            print("=" * 80)

            # Finish wandb run
            if wandb.run is not None:
                try:
                    # Log final summary
                    wandb.summary['best_val_loss'] = self.best_val_loss
                    wandb.summary['total_training_time_hours'] = total_time / 3600
                    wandb.summary['total_epochs'] = num_epochs
                    wandb.finish()
                    print("WandB run finished successfully")
                except Exception as e:
                    print(f"Warning: Error finishing WandB run: {e}")


def setup_distributed():
    """Initialize distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        rank = -1
        world_size = -1
        local_rank = -1

    if local_rank != -1:
        torch.cuda.set_device(local_rank)

        # Set NCCL timeout to 30 minutes (default is 10 minutes)
        # This helps prevent timeout during expensive validation metrics computation
        os.environ.setdefault('NCCL_TIMEOUT', '1800')

        # Additional NCCL settings for stability
        os.environ.setdefault('TORCH_NCCL_ASYNC_ERROR_HANDLING', '1')  # Better error handling
        os.environ.setdefault('NCCL_DEBUG', 'WARN')  # Less verbose logging

        # Initialize process group with timeout
        timeout_minutes = 30
        dist.init_process_group(
            backend='nccl',
            timeout=timedelta(minutes=timeout_minutes)
        )

        if rank == 0:
            print(f"Distributed training initialized with NCCL timeout: {timeout_minutes} minutes")

    return local_rank, rank, world_size


def cleanup_distributed():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


def main(args):
    # Setup distributed training
    local_rank, rank, world_size = setup_distributed()
    is_distributed = local_rank != -1
    is_main_process = (not is_distributed) or (rank == 0)

    # Set random seed
    if args.seed is not None:
        torch.manual_seed(args.seed + rank if is_distributed else args.seed)
        np.random.seed(args.seed + rank if is_distributed else args.seed)

    # Device
    if is_distributed:
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if is_main_process:
        print(f"Using device: {device}")
        if is_distributed:
            print(f"Distributed training: world_size={world_size}, rank={rank}, local_rank={local_rank}")

    # Keypoint names (Panda robot example)
    keypoint_names = [
        'panda_link0', 'panda_link2', 'panda_link3',
        'panda_link4', 'panda_link6', 'panda_link7', 'panda_hand'
    ]

    # Create dataloaders
    if is_main_process:
        print("\nLoading datasets...")

    if args.val_dir:
        # Separate train/val directories
        train_loader, val_loader = create_dataloaders(
            train_dir=args.data_dir,
            val_dir=args.val_dir,
            keypoint_names=keypoint_names,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            image_size=(args.image_size, args.image_size),
            heatmap_size=(args.heatmap_size, args.heatmap_size)
        )
    else:
        # Split single dataset
        dataset = PoseEstimationDataset(
            data_dir=args.data_dir,
            keypoint_names=keypoint_names,
            image_size=(args.image_size, args.image_size),
            heatmap_size=(args.heatmap_size, args.heatmap_size),
            augment=True,
            multi_robot=args.multi_robot,
            robot_types=args.robot_types
        )

        # Split into train/val
        train_size = int(args.train_split * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        # Disable augmentation for validation
        val_dataset.dataset.augment = False

        # Create samplers for distributed training
        if is_distributed:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True
            )
            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False
            )
        else:
            train_sampler = None
            val_sampler = None

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=args.num_workers,
            pin_memory=True
        )

    if is_main_process:
        print(f"Train samples: {len(train_loader.dataset)}")
        print(f"Val samples: {len(val_loader.dataset)}")

    # Create model
    if is_main_process:
        print("\nCreating model...")

    # Load camera intrinsics if available
    camera_K = None
    raw_res = (640, 480)
    cam_settings_path = Path(args.data_dir) / "_camera_settings.json"
    if not cam_settings_path.exists():
        for p in Path(args.data_dir).rglob("_camera_settings.json"):
            cam_settings_path = p
            break
    
    if cam_settings_path.exists():
        try:
            camera_K = dream.utilities.load_camera_intrinsics(str(cam_settings_path))
            raw_res = dream.utilities.load_image_resolution(str(cam_settings_path))
            if is_main_process:
                print(f"Loaded camera intrinsics from {cam_settings_path}")
                print(f"Raw resolution: {raw_res}")
        except Exception as e:
            if is_main_process:
                print(f"Warning: Failed to load camera settings: {e}")

    model = DINOv3PoseEstimator(
        dino_model_name=args.model_name,
        heatmap_size=(args.heatmap_size, args.heatmap_size),
        unfreeze_blocks=args.unfreeze_blocks
    ).to(device)

    # Wrap model with DistributedDataParallel for multi-GPU training
    if is_distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False
        )

    if is_main_process:
        print(f"Model: {args.model_name}")
        model_to_count = model.module if is_distributed else model
        print(f"Number of parameters: {sum(p.numel() for p in model_to_count.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in model_to_count.parameters() if p.requires_grad):,}")

    # Loss function
    criterion = PoseLoss(
        heatmap_weight=args.heatmap_weight,
        angle_weight=args.angle_weight
    )

    # Optimizer
    if args.optimizer == 'adam':
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.learning_rate,
            momentum=0.9,
            weight_decay=args.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    # Learning rate scheduler
    scheduler = None
    if args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.lr_step_size,
            gamma=args.lr_gamma
        )
    elif args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.min_lr  # Minimum learning rate
        )
        if is_main_process:
            print(f"Using CosineAnnealingLR: initial_lr={args.learning_rate}, min_lr={args.min_lr}, T_max={args.epochs}")
    elif args.scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )

    # Training config
    config = {
        'model_name': args.model_name,
        'image_size': args.image_size,
        'heatmap_size': args.heatmap_size,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'optimizer': args.optimizer,
        'learning_rate': args.learning_rate,
        'min_lr': args.min_lr,
        'weight_decay': args.weight_decay,
        'scheduler': args.scheduler,
        'heatmap_weight': args.heatmap_weight,
        'angle_weight': args.angle_weight,
        'keypoint_names': keypoint_names,
        'wandb_project': args.wandb_project,
        'wandb_run_name': args.wandb_run_name
    }

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir=args.output_dir,
        config=config,
        camera_K=camera_K,
        raw_res=raw_res,
        resume_from=args.resume,
        local_rank=local_rank
    )

    # Start training
    try:
        trainer.train(args.epochs)
    finally:
        # Cleanup distributed training
        cleanup_distributed()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train DINOv3 Pose Estimation Model')

    # Data
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to training data directory')
    parser.add_argument('--val-dir', type=str, default=None,
                        help='Path to validation data directory (optional)')
    parser.add_argument('--train-split', type=float, default=0.9,
                        help='Train/val split ratio if val-dir not provided (DREAM uses 0.8, we use 0.9)')
    parser.add_argument('--multi-robot', action='store_true',
                        help='Load data from multiple robot subdirectories for unified model')
    parser.add_argument('--robot-types', type=str, nargs='+', default=None,
                        help='Filter specific robot types (e.g., panda kuka baxter)')

    # Model
    parser.add_argument('--model-name', type=str,
                        default='facebook/dinov2-base',
                        help='DINOv3 model name from HuggingFace')
    parser.add_argument('--image-size', type=int, default=512,
                        help='Input image size')
    parser.add_argument('--heatmap-size', type=int, default=512,
                        help='Output heatmap size')
    parser.add_argument('--unfreeze-blocks', type=int, default=2,
                        help='Number of backbone blocks to unfreeze')

    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')

    # Optimization
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'adamw', 'sgd'],
                        help='Optimizer type')
    parser.add_argument('--learning-rate', '--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        help='Weight decay')

    # Scheduler
    parser.add_argument('--scheduler', type=str, default='step',
                        choices=['step', 'cosine', 'plateau', 'none'],
                        help='Learning rate scheduler')
    parser.add_argument('--lr-step-size', type=int, default=30,
                        help='Step size for StepLR scheduler')
    parser.add_argument('--lr-gamma', type=float, default=0.1,
                        help='Gamma for StepLR scheduler')
    parser.add_argument('--min-lr', type=float, default=1e-8,
                        help='Minimum learning rate for CosineAnnealingLR scheduler')

    # Loss weights
    parser.add_argument('--heatmap-weight', type=float, default=1.0,
                        help='Weight for heatmap loss')
    parser.add_argument('--angle-weight', type=float, default=0.1,
                        help='Weight for angle loss')

    # Output
    parser.add_argument('--output-dir', type=str, default='./outputs',
                        help='Output directory for checkpoints and logs')

    # Resume
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    # Random seed
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    # Wandb
    parser.add_argument('--wandb-project', type=str, default='dinov3-pose-estimation',
                        help='Wandb project name')
    parser.add_argument('--wandb-run-name', type=str, default=None,
                        help='Wandb run name (optional)')

    args = parser.parse_args()

    main(args)
