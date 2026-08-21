"""
DINOv3 Pose Estimation Dataset
DREAM 데이터셋 구조를 따르는 데이터로더
"""

import os
import json
import math
import random
import glob
from pathlib import Path
import numpy as np
from PIL import Image as PILImage
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import albumentations as albu
from typing import Dict, List, Tuple, Optional


def fda_transfer(src_img: np.ndarray, trg_img: np.ndarray, beta: float = 0.01) -> np.ndarray:
    """
    FDA (Fourier Domain Adaptation): Replace low-frequency spectrum of source with target's.
    Low-freq = overall color/tone (domain-specific), High-freq = edges/structure (task-relevant).

    Args:
        src_img: Synthetic image (H, W, 3), uint8
        trg_img: Real image (H, W, 3), uint8
        beta: Low-frequency replacement ratio (0.01~0.05 recommended for DR data)
    Returns:
        FDA-applied image (H, W, 3), uint8
    """
    src = src_img.astype(np.float32)
    trg = trg_img.astype(np.float32)

    # Resize target to match source
    if src.shape[:2] != trg.shape[:2]:
        trg = np.array(PILImage.fromarray(trg.astype(np.uint8)).resize(
            (src.shape[1], src.shape[0]), PILImage.BILINEAR
        )).astype(np.float32)

    result = np.zeros_like(src)
    h, w = src.shape[:2]
    cy, cx = h // 2, w // 2
    bh, bw = max(int(h * beta), 1), max(int(w * beta), 1)

    for ch in range(3):
        fft_src = np.fft.fftshift(np.fft.fft2(src[:, :, ch]))
        fft_trg = np.fft.fftshift(np.fft.fft2(trg[:, :, ch]))

        amp_src = np.abs(fft_src)
        phase_src = np.angle(fft_src)
        amp_trg = np.abs(fft_trg)

        # Replace low-frequency amplitude
        amp_src[cy - bh:cy + bh, cx - bw:cx + bw] = amp_trg[cy - bh:cy + bh, cx - bw:cx + bw]

        fft_result = np.fft.ifftshift(amp_src * np.exp(1j * phase_src))
        result[:, :, ch] = np.real(np.fft.ifft2(fft_result))

    return np.clip(result, 0, 255).astype(np.uint8)


# Robot type constants (must match model.py)
ROBOT_TYPE_NAMES = ['franka_panda', 'meca500', 'fr5', 'franka_research3']


def infer_robot_type_from_path(path: str) -> int:
    """
    Infer robot type from directory path based on naming conventions.

    Args:
        path: Directory path containing the dataset

    Returns:
        Robot type index (0-3) corresponding to ROBOT_TYPE_NAMES
    """
    path_lower = path.lower()

    # Check for specific patterns (order matters - more specific first)
    if 'research3' in path_lower:
        return 3  # franka_research3
    elif 'Fr5' in path_lower:
        return 2  # fr5
    elif 'Meca' in path_lower:
        return 1  # meca500
    elif 'panda' in path_lower or 'dream' in path_lower:
        return 0  # franka_panda
    else:
        # Default to franka_panda if no match
        return 0


class PoseEstimationDataset(Dataset):
    """
    DREAM 스타일의 NDDS 데이터셋을 위한 데이터로더

    데이터 구조:
    - RGB 이미지
    - JSON 어노테이션 (keypoint 위치)
    - (선택적) Joint angle 정보
    """

    def __init__(
        self,
        data_dir: str,
        keypoint_names: List[str],
        image_size: Tuple[int, int] = (512, 512),
        heatmap_size: Tuple[int, int] = (512, 512),
        augment: bool = False,
        normalize: bool = True,
        include_angles: bool = True,
        sigma: float = 5.0,  # Gaussian heatmap sigma
        multi_robot: bool = False,  # Load data from multiple robot subdirectories
        robot_types: Optional[List[str]] = None,  # List of robot types to include
        fda_real_dir: Optional[str] = None,  # Real image directory for FDA augmentation
        fda_beta: float = 0.01,  # FDA low-frequency replacement ratio
        fda_prob: float = 0.5,  # Probability of applying FDA per sample
        occlusion_prob: float = 0.0,  # Probability of synthetic occlusion augmentation
        occlusion_max_holes: int = 6,  # Max number of coarse occlusion patches
        occlusion_max_size_frac: float = 0.2,  # Max occluder size relative to image side
        json_allowlist_path: Optional[str] = None,  # Optional list file (txt/json) to keep only selected json frames
        aug_level: str = "light",  # 'light' (default, legacy) or 'strong' (Stage-1 sim-to-real)
        norm_mean: Optional[List[float]] = None,  # backbone normalization mean (default ImageNet)
        norm_std: Optional[List[float]] = None,   # backbone normalization std  (default ImageNet)
        crop_to_robot: bool = False,  # square-crop around the robot (GT-keypoint bbox) before resize
        crop_margin: float = 1.5,     # bbox expansion factor (random in [1.3, margin] when augmenting)
        crop_aspect: float = 1.0,     # 크롭 직사각형의 w/h. 1.0 = 기존(정사각) 동작 — 기본값 유지 필수.
        # ⚠️ 배포 파이프라인(Eval/selfbbox_eval.py)은 원본 640x480 을 먼저 512x512 로 비등방
        # 리사이즈(x0.8, y1.0667)한 뒤 그 공간에서 정사각 roi_align 크롭을 뜬다. 원본 좌표로 보면
        # 그 크롭은 w/h = (W0/512)/(H0/512) = W0/H0 = 4/3 인 직사각형이다. 반면 학습(이 코드)은
        # 원본 공간에서 정사각(w/h=1)으로 잘라왔다 => crop detector 가 학습에서 한 번도 못 본
        # 4:3 -> 1:1 세로 확대 왜곡을 배포에서 받는다. crop_aspect=W0/H0 로 두면 배포와 일치.
        # (측정: Eval/crop_chain_probe.py — clean 2D 오차 median 1.78px(배포) vs 1.14px(왜곡없음))
        crop_aspect_jitter: float = 0.0,  # crop_aspect 에 곱해지는 log-uniform 지터 (0 = 없음)
        crop_res_jitter: float = 0.0,     # 크롭을 저해상도로 리샘플할 확률 (작은 로봇 유효해상도 모사)
        crop_res_range: Tuple[int, int] = (140, 512),  # 저해상도 리샘플 목표 변 길이 범위(px)
        # ── det_C (2026-07-31) 가림-현실화 + 프레임-경계 증강 (opt-in, 기본 off = 배포 불변) ──
        # distractor_occ_prob>0 : 로봇 위에 '다른 이미지 크롭(질감/물체)' 을 붙여 흑박스가 아닌
        #   현실적 distractor 로 가림 (robot-vs-distractor 학습). 관절은 그 자리에 있으므로 타깃
        #   heatmap 은 유지 → 검출기가 가려진 관절을 문맥으로 추론하도록 학습.
        # frame_boundary_prob>0 : 크롭을 강하게 편심/축소해 로봇 일부를 프레임 밖으로 밀어냄.
        #   프레임 밖 관절은 좌표가 [0,side) 밖 → _create_heatmap 이 zero 타깃을 만듦 → 검출기가
        #   가장자리에서 confident keypoint 를 '환각' 하지 않고 conf 를 낮추도록 학습.
        distractor_occ_prob: float = 0.0,
        distractor_occ_max_frac: float = 0.35,  # occluder 최대 크기 (로봇 kp bbox 대각 대비)
        frame_boundary_prob: float = 0.0,
        angle_joint_names: Optional[List[str]] = None,  # explicit sim_state joint names for GT angles
        # (in order). None -> legacy joints[:7] (correct for Panda). Needed for robots whose sim_state
        # has non-arm joints first (e.g. KUKA 'iiwa7_base_link_iiwa7_joint' -> use iiwa7_joint_1..7).
    ):
        """
        Args:
            data_dir: NDDS 데이터가 있는 디렉토리
            keypoint_names: 키포인트 이름 리스트 (예: ['panda_link0', ...])
            image_size: 네트워크 입력 이미지 크기
            heatmap_size: 출력 heatmap 크기
            augment: 데이터 증강 사용 여부
            normalize: 이미지 정규화 여부
            include_angles: joint angle 정보 포함 여부
            sigma: Gaussian heatmap의 표준편차
            multi_robot: True면 data_dir 하위의 모든 로봇 데이터를 통합하여 로드
            robot_types: multi_robot=True일 때 특정 로봇 타입만 필터링 (예: ['panda', 'kuka'])
            fda_real_dir: Real 이미지 디렉토리 (FDA style source, label 불필요)
            fda_beta: FDA 저주파 교체 비율 (0.01=미세한 톤 변화, 0.05=강한 변환)
            fda_prob: FDA 적용 확률 (0.5 = 50%의 샘플에 적용)
            occlusion_prob: 가려짐 증강(CoarseDropout) 적용 확률
            occlusion_max_holes: 최대 가림 패치 개수
            occlusion_max_size_frac: 가림 패치 최대 크기 비율(이미지 변 길이 대비)
            json_allowlist_path: 선택된 frame json 이름/경로 리스트 파일(txt/json)
        """
        self.data_dir = data_dir
        self.keypoint_names = keypoint_names
        self.image_size = image_size
        self.heatmap_size = heatmap_size
        self.augment = augment
        self.include_angles = include_angles
        self.sigma = sigma
        self.multi_robot = multi_robot
        self.robot_types = robot_types
        self.angle_joint_names = angle_joint_names
        self.fda_beta = fda_beta
        self.fda_prob = fda_prob
        self.occlusion_prob = occlusion_prob
        self.occlusion_max_holes = max(1, int(occlusion_max_holes))
        self.occlusion_max_size_frac = max(0.01, float(occlusion_max_size_frac))
        self.crop_to_robot = crop_to_robot
        self.crop_margin = float(crop_margin)
        self.crop_aspect = float(crop_aspect)
        self.crop_aspect_jitter = float(crop_aspect_jitter)
        self.crop_res_jitter = float(crop_res_jitter)
        self.crop_res_range = tuple(crop_res_range)
        self.distractor_occ_prob = float(distractor_occ_prob)
        self.distractor_occ_max_frac = float(distractor_occ_max_frac)
        self.frame_boundary_prob = float(frame_boundary_prob)
        self.json_allowlist_path = json_allowlist_path
        self._intr_cache = {}          # dir -> (fx, fy) from _camera_settings.json (k_value only)
        self.json_allowlist_keys = self._load_json_allowlist(json_allowlist_path)
        self.aug_level = aug_level
        self.norm_mean = norm_mean if norm_mean is not None else [0.485, 0.456, 0.406]
        self.norm_std = norm_std if norm_std is not None else [0.229, 0.224, 0.225]

        # FDA: Load real image paths for style transfer
        self.fda_real_paths = []
        if fda_real_dir and os.path.isdir(fda_real_dir):
            for ext in ['*.jpg', '*.png', '*.jpeg']:
                self.fda_real_paths.extend(glob.glob(os.path.join(fda_real_dir, '**', ext), recursive=True))
            if self.fda_real_paths:
                print(f"FDA enabled: {len(self.fda_real_paths)} real images from {fda_real_dir} (beta={fda_beta}, prob={fda_prob})")
            else:
                print(f"FDA warning: No images found in {fda_real_dir}")

        # 데이터 파일 리스트 로드
        self.samples = self._load_dataset()

        # 이미지 변환 설정
        if normalize:
            # 기본 ImageNet 통계 (DINOv3). SigLIP/SigLIP2는 mean=std=0.5 ([-1,1])를 사용하므로
            # norm_mean/norm_std로 백본에 맞게 교체 가능.
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.norm_mean, std=self.norm_std)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
            ])

        # 데이터 증강 설정
        if self.augment and self.aug_level == "strong":
            # 🔥 [강함] Stage-1 sim-to-real: heavy photometric + blur + JPEG + occlusion
            #     Geometric kept moderate (keypoints are transformed by albumentations).
            self.augmentation = albu.Compose([
                # Photometric — close the synthetic→real appearance gap
                albu.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
                albu.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=30,
                                        val_shift_limit=20, p=0.5),
                albu.RandomGamma(gamma_limit=(70, 130), p=0.3),
                albu.OneOf([
                    albu.CLAHE(clip_limit=2.0, p=1.0),
                    albu.Sharpen(p=1.0),
                ], p=0.2),
                # Blur / sensor — real cameras
                albu.OneOf([
                    albu.MotionBlur(blur_limit=7, p=1.0),
                    albu.GaussianBlur(blur_limit=(3, 7), p=1.0),
                    albu.Defocus(radius=(1, 3), p=1.0),
                ], p=0.3),
                albu.GaussNoise(std_range=(0.02, 0.08), p=0.3),
                albu.ISONoise(p=0.2),
                albu.ImageCompression(quality_range=(40, 90), p=0.3),
                # Occlusion — robustness to missing/occluded joints (RoboPEPP-style)
                albu.CoarseDropout(
                    num_holes_range=(1, 5),
                    hole_height_range=(0.05, 0.18),
                    hole_width_range=(0.05, 0.18),
                    fill=0,
                    p=0.5,
                ),
                # Geometric — moderate; albumentations re-maps keypoints
                albu.ShiftScaleRotate(
                    shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5
                ),
            ], keypoint_params=albu.KeypointParams(format='xy', remove_invisible=False))
        elif self.augment and self.aug_level == "strong_vp":
            # 🔥 [강함+시점] 'strong' 과 광학 증강은 동일하되 기하 증강을 넓힘: 넓은 회전(±25°)+
            #     원근(perspective)+shear. panda-orb 처럼 시점이 다양하고 로봇이 작게 잡히는
            #     split 의 검출기 2D 한계를 겨냥한 opt-in 레벨. 'strong'/'light' 는 불변.
            self.augmentation = albu.Compose([
                albu.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
                albu.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=30,
                                        val_shift_limit=20, p=0.5),
                albu.RandomGamma(gamma_limit=(70, 130), p=0.3),
                albu.OneOf([
                    albu.CLAHE(clip_limit=2.0, p=1.0),
                    albu.Sharpen(p=1.0),
                ], p=0.2),
                albu.OneOf([
                    albu.MotionBlur(blur_limit=7, p=1.0),
                    albu.GaussianBlur(blur_limit=(3, 7), p=1.0),
                    albu.Defocus(radius=(1, 3), p=1.0),
                ], p=0.3),
                albu.GaussNoise(std_range=(0.02, 0.08), p=0.3),
                albu.ISONoise(p=0.2),
                albu.ImageCompression(quality_range=(40, 90), p=0.3),
                albu.CoarseDropout(
                    num_holes_range=(1, 5),
                    hole_height_range=(0.05, 0.18),
                    hole_width_range=(0.05, 0.18),
                    fill=0,
                    p=0.5,
                ),
                # Geometric — WIDER: viewpoint/roll diversity for orb-like splits
                albu.Affine(scale=(0.85, 1.15), translate_percent=(-0.1, 0.1),
                            rotate=(-25, 25), shear=(-8, 8), p=0.6),
                albu.Perspective(scale=(0.02, 0.08), p=0.35),
            ], keypoint_params=albu.KeypointParams(format='xy', remove_invisible=False))
        elif self.augment:
            # 🚀 [경량] 학습 수렴 우선 — 최소한의 augmentation (legacy default)
            self.augmentation = albu.Compose([
                # 1. 가벼운 노이즈 (강도↓, 확률↓)
                albu.GaussNoise(std_range=(0.01, 0.03), p=0.15),

                # 2. 약한 색상 변화 (brightness/contrast 절반, hue 제거)
                albu.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.2),

                # 3. Occlusion 최소화 (작은 패치 1-2개, 낮은 확률)
                albu.CoarseDropout(
                    num_holes_range=(1, 2),
                    hole_height_range=(0.03, 0.1),
                    hole_width_range=(0.03, 0.1),
                    fill=0,
                    p=0.15,
                ),

                # 4. 기하학적 변환 최소화 (shift/scale만, rotation 거의 없음)
                albu.ShiftScaleRotate(
                    shift_limit=0.05,
                    scale_limit=0.05,
                    rotate_limit=5,
                    p=0.15
                ),

            ], keypoint_params=albu.KeypointParams(format='xy', remove_invisible=False))

    def _intrinsics_for(self, json_path: str):
        """Real (fx, fy) for the frame's dataset, from NDDS `_camera_settings.json`.

        ⚠️ DREAM frame JSONs carry NO `meta.K` (verified: kuka/baxter/panda synth AND panda real),
        so `_load_keypoints_from_json` silently falls back to `camera_K = eye(3)` — i.e. fx=fy=1,
        cx=cy=0. That fallback is LOAD-BEARING: every trained checkpoint learned its bearing
        features from it (the de-facto input is raw pixel coordinates), so `camera_K` is left
        exactly as-is. This lookup is used ONLY for k_value, which is meaningless without metric
        focal lengths. Same source the eval path already uses (Eval/baxter_rc_eval.py:53,
        train.py:37). Cached per directory; returns None when absent.
        """
        d = os.path.dirname(json_path)
        if d in self._intr_cache:
            return self._intr_cache[d]
        fxfy, probe = None, d
        for _ in range(4):                      # walk up a few levels; NDDS puts it at dataset root
            cand = os.path.join(probe, '_camera_settings.json')
            if os.path.exists(cand):
                try:
                    it = json.load(open(cand))['camera_settings'][0]['intrinsic_settings']
                    fxfy = (float(it['fx']), float(it['fy']))
                except Exception:
                    fxfy = None
                break
            nxt = os.path.dirname(probe)
            if nxt == probe:
                break
            probe = nxt
        self._intr_cache[d] = fxfy
        return fxfy

    @staticmethod
    def _normalize_path_token(path_str: str) -> str:
        return os.path.normpath(path_str).replace('\\', '/').lower()

    def _load_json_allowlist(self, allowlist_path: Optional[str]) -> Optional[set]:
        if not allowlist_path:
            return None

        if not os.path.exists(allowlist_path):
            raise FileNotFoundError(f"json allowlist not found: {allowlist_path}")

        keys = set()
        suffix = Path(allowlist_path).suffix.lower()

        def add_key(token: str):
            token = str(token).strip()
            if not token:
                return
            keys.add(self._normalize_path_token(token))
            name = os.path.basename(token)
            if name:
                keys.add(name.lower())
                stem = os.path.splitext(name)[0]
                if stem:
                    keys.add(stem.lower())

        if suffix == '.json':
            with open(allowlist_path, 'r') as f:
                payload = json.load(f)

            if isinstance(payload, list):
                for item in payload:
                    if isinstance(item, str):
                        add_key(item)
                    elif isinstance(item, dict):
                        for field in ('json_path', 'json_name', 'name'):
                            if field in item and item[field]:
                                add_key(item[field])
            elif isinstance(payload, dict):
                for field in ('json_paths', 'json_names', 'items'):
                    val = payload.get(field)
                    if isinstance(val, list):
                        for item in val:
                            if isinstance(item, str):
                                add_key(item)
                            elif isinstance(item, dict):
                                for f2 in ('json_path', 'json_name', 'name'):
                                    if f2 in item and item[f2]:
                                        add_key(item[f2])
            else:
                raise ValueError(f"Unsupported allowlist JSON format: {allowlist_path}")
        else:
            with open(allowlist_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    add_key(line)

        if not keys:
            raise ValueError(f"json allowlist is empty: {allowlist_path}")

        print(f"JSON allowlist loaded: {allowlist_path} ({len(keys)} match keys)")
        return keys

    def _is_json_allowed(self, json_path: str, json_file: str) -> bool:
        if self.json_allowlist_keys is None:
            return True

        p = Path(json_path)
        candidates = {
            self._normalize_path_token(json_path),
            self._normalize_path_token(str(p.resolve())),
            json_file.lower(),
            p.stem.lower(),
        }
        try:
            rel = p.resolve().relative_to(Path(self.data_dir).resolve())
            candidates.add(self._normalize_path_token(str(rel)))
        except Exception:
            pass
        return any(c in self.json_allowlist_keys for c in candidates)

    def _load_dataset(self) -> List[Dict]:
        """NDDS 데이터셋에서 샘플 리스트 로드"""
        samples = []

        if self.multi_robot:
            # Multi-robot mode: load from all subdirectories
            print(f"Loading multi-robot dataset from {self.data_dir}")
            robot_dirs = []

            # DREAM data structure: /data/real/* and /data/synthetic/*
            # Recursively find all robot-specific directories
            def find_robot_dirs(base_path, max_depth=3, current_depth=0):
                """Recursively find directories containing robot data"""
                found_dirs = []
                if current_depth > max_depth:
                    return found_dirs

                try:
                    items = os.listdir(base_path)
                except (PermissionError, FileNotFoundError) as e:
                    print(f"  Warning: Cannot access {base_path}: {e}")
                    return found_dirs

                for item in items:
                    item_path = os.path.join(base_path, item)
                    if not os.path.isdir(item_path):
                        continue

                    # Check if this directory contains data files (has .json files)
                    try:
                        dir_files = os.listdir(item_path)
                        has_json = any(f.endswith('.json') for f in dir_files)
                    except (PermissionError, FileNotFoundError):
                        has_json = False

                    if has_json:
                        # This is a data directory, check if it matches robot filter
                        if self.robot_types:
                            if any(robot_type.lower() in item.lower() for robot_type in self.robot_types):
                                print(f"  Adding data directory: {item_path}")
                                found_dirs.append(item_path)
                        else:
                            print(f"  Adding data directory: {item_path}")
                            found_dirs.append(item_path)
                    else:
                        # Recurse into subdirectories
                        found_dirs.extend(find_robot_dirs(item_path, max_depth, current_depth + 1))

                return found_dirs

            print(f"Searching for robot type(s): {self.robot_types}")
            robot_dirs = find_robot_dirs(self.data_dir)

            print(f"Found {len(robot_dirs)} robot data directories")
            for rdir in robot_dirs:
                print(f"  - {rdir}")
                samples.extend(self._load_from_directory(rdir))
        else:
            # Single directory mode
            samples = self._load_from_directory(self.data_dir)

        print(f"Loaded {len(samples)} samples total")
        return samples

    def _load_from_directory(self, directory: str) -> List[Dict]:
        """단일 디렉토리에서 샘플 로드"""
        samples = []

        # Infer robot type from directory path
        robot_type = infer_robot_type_from_path(directory)

        # NDDS 형식: 각 프레임마다 이미지와 .json 파일이 쌍으로 존재
        for root, dirs, files in os.walk(directory):
            json_files = [f for f in files if f.endswith('.json') and not f.startswith('_')]

            for json_file in json_files:
                json_path = os.path.join(root, json_file)
                if not self._is_json_allowed(json_path, json_file):
                    continue
                base_name = json_file.replace('.json', '')
                img_path = None

                # Try to read image path from JSON meta
                try:
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                        if 'meta' in data and 'image_path' in data['meta']:
                            # Get image path from JSON (can be relative or absolute)
                            json_img_path = data['meta']['image_path']

                            # Fix incorrect relative path: ../dataset/... should be ../../../...
                            if json_img_path.startswith('../dataset/'):
                                json_img_path = json_img_path.replace('../dataset/', '../../../', 1)

                            # If relative path, resolve from JSON directory
                            if not os.path.isabs(json_img_path):
                                img_path = os.path.normpath(os.path.join(root, json_img_path))
                            else:
                                img_path = json_img_path

                            # Verify image exists
                            if not os.path.exists(img_path):
                                print(f"Warning: Image not found: {img_path}")
                                img_path = None
                except Exception as e:
                    print(f"Warning: Failed to read {json_path}: {e}")
                    img_path = None

                # Fallback: look for image in same directory as JSON
                if img_path is None:
                    for ext in ['.rgb.jpg', '.png', '.jpg', '.jpeg']:
                        potential_path = os.path.join(root, base_name + ext)
                        if os.path.exists(potential_path):
                            img_path = potential_path
                            break

                if img_path:
                    # Detect synthetic data (DREAM sim uses cm, real uses m)
                    is_synthetic = 'syn' in directory.lower()
                    samples.append({
                        'image_path': img_path,
                        'annotation_path': json_path,
                        'name': base_name,
                        'source_dir': os.path.basename(directory),
                        'robot_type': robot_type,
                        'is_synthetic': is_synthetic
                    })

        return samples

    def _load_keypoints_from_json(self, json_path: str) -> Dict:
        """JSON 파일에서 keypoint 정보 로드"""
        with open(json_path, 'r') as f:
            data = json.load(f)

        keypoints = {}

        # Initialize keypoint positions array with the correct order
        # This ensures keypoints are in self.keypoint_names order, not JSON order
        keypoint_positions = np.zeros((len(self.keypoint_names), 2), dtype=np.float32)
        keypoint_positions_3d = np.zeros((len(self.keypoint_names), 3), dtype=np.float32)
        keypoint_found = [False] * len(self.keypoint_names)
        keypoint_match_priority = [-1] * len(self.keypoint_names)

        # NDDS 형식에서 keypoint 추출
        if 'objects' in data:
            for obj in data['objects']:
                if 'keypoints' in obj:
                    for kp in obj['keypoints']:
                        kp_name = kp['name']
                        kp_name_l = kp_name.lower()
                        kp_is_collision = '(collision' in kp_name_l
                        # 부분 일치 검사 (예: 'panda_link0'에서 'link0' 찾기).
                        # Exact/non-collision labels must beat collision aliases in original DREAM
                        # robot dumps, where both visual and collision keypoints share substrings.
                        target_idx = -1
                        target_priority = -1
                        for i, name in enumerate(self.keypoint_names):
                            name_l = name.lower()
                            if name_l == kp_name_l:
                                target_idx = i
                                target_priority = 3
                                break
                            if name_l in kp_name_l:
                                priority = 1 if kp_is_collision else 2
                                if priority > target_priority:
                                    target_idx = i
                                    target_priority = priority
                        
                        if target_idx != -1 and target_priority >= keypoint_match_priority[target_idx]:
                            keypoint_positions[target_idx] = [
                                kp['projected_location'][0],
                                kp['projected_location'][1]
                            ]
                            if 'location' in kp:
                                keypoint_positions_3d[target_idx] = [
                                    kp['location'][0],
                                    kp['location'][1],
                                    kp['location'][2]
                                ]
                            keypoint_found[target_idx] = True
                            keypoint_match_priority[target_idx] = target_priority

        # Mark missing keypoints with negative coordinates
        for i, found in enumerate(keypoint_found):
            if not found:
                keypoint_positions[i] = [-1, -1]
                keypoint_positions_3d[i] = [-1, -1, -1]

        keypoints['projections'] = keypoint_positions
        keypoints['locations'] = keypoint_positions_3d

        # Camera intrinsic matrix K (from meta.K)
        if 'meta' in data and 'K' in data['meta']:
            keypoints['camera_K'] = np.array(data['meta']['K'], dtype=np.float32)
        else:
            # Default fallback (should not happen with proper data)
            keypoints['camera_K'] = np.eye(3, dtype=np.float32)

        # Joint angles from sim_state.joints. Default: first 7 (correct for Panda). When
        # angle_joint_names is given, select those joints BY NAME in order (robot-agnostic;
        # KUKA sim_state leads with 'iiwa7_base_link_iiwa7_joint' so positional [:7] drops joint_7).
        if self.include_angles and 'sim_state' in data and 'joints' in data['sim_state']:
            joints = data['sim_state']['joints']
            if self.angle_joint_names:
                pos = {j['name'].split('/')[-1]: j['position'] for j in joints}
                angles = np.array([pos.get(n, 0.0) for n in self.angle_joint_names], dtype=np.float32)
            else:
                angles = np.array([j['position'] for j in joints[:7]], dtype=np.float32)
            keypoints['angles'] = angles  # radians, no normalization (FK needs raw radians)

        return keypoints

    def _create_heatmap(self, keypoints: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """
        Keypoint 위치로부터 Gaussian heatmap 생성 (윈도우 + separable 최적화).
        sigma가 작아 Gaussian이 키포인트 주변 좁은 영역에만 존재하므로, 전체 HxW가 아니라
        반경 R 윈도우에서 separable(outer product)로만 계산 -> ~수백배 빠름 (CPU 병목 제거).
        """
        H, W = size
        num_keypoints = len(keypoints)
        heatmaps = np.zeros((num_keypoints, H, W), dtype=np.float32)

        sigma = max(self.sigma, 1.0)
        two_s2 = 2.0 * sigma ** 2
        # exp(-d2/2σ²) < 0.01  =>  d2 > 2σ²·ln(100); per-axis radius (+1 margin)
        R = int(np.ceil(sigma * np.sqrt(2.0 * np.log(100.0)))) + 1

        for i, (x, y) in enumerate(keypoints):
            # 이미지 범위를 벗어난 키포인트 처리 (Albumentations 이후 대비)
            if x < 0 or y < 0 or x >= W or y >= H:
                continue

            xi, yi = int(round(float(x))), int(round(float(y)))
            x0, x1 = max(0, xi - R), min(W, xi + R + 1)
            y0, y1 = max(0, yi - R), min(H, yi + R + 1)
            if x1 <= x0 or y1 <= y0:
                continue

            gx = np.exp(-((np.arange(x0, x1) - x) ** 2) / two_s2)  # (w,)
            gy = np.exp(-((np.arange(y0, y1) - y) ** 2) / two_s2)  # (h,)
            patch = np.outer(gy, gx).astype(np.float32)            # separable Gaussian
            patch[patch < 0.01] = 0
            heatmaps[i, y0:y1, x0:x1] = patch

        return heatmaps

    def _paste_distractors(self, img_np: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
        """det_C: 현실적 distractor occluder 를 로봇 위에 붙인다 (흑박스 대신 질감/물체 크롭).

        img_np: (H,W,3) uint8, crop 좌표계. keypoints: (N,2) 같은 좌표계.
        관절 위치는 바꾸지 않는다 (가려질 뿐) → 타깃 heatmap 유지 → 검출기가 가려진 관절을
        문맥으로 추론(robot-vs-distractor)하도록 학습. occluder 소스 = 데이터셋의 다른 임의 이미지
        크롭(자연스러운 질감/DR 물체) + 일부는 흑/단색(벤치의 흑박스 일반화 유지).
        """
        H, W = img_np.shape[:2]
        inb = ((keypoints[:, 0] >= 0) & (keypoints[:, 0] < W) &
               (keypoints[:, 1] >= 0) & (keypoints[:, 1] < H))
        if inb.sum() < 1:
            return img_np
        pts = keypoints[inb]
        diag = float(np.hypot(pts[:, 0].max() - pts[:, 0].min(),
                              pts[:, 1].max() - pts[:, 1].min()))
        diag = max(diag, 24.0)
        n_occ = random.randint(1, 3)
        for _ in range(n_occ):
            # occluder 크기 = 로봇 span 의 U(0.12, max_frac)
            frac = random.uniform(0.12, self.distractor_occ_max_frac)
            ow = max(8, int(frac * diag * random.uniform(0.7, 1.4)))
            oh = max(8, int(frac * diag * random.uniform(0.7, 1.4)))
            # 임의의 in-frame 관절 중심 + 지터 (로봇을 실제로 가리도록)
            c = pts[random.randrange(len(pts))]
            cx = int(c[0] + random.uniform(-0.4, 0.4) * ow)
            cy = int(c[1] + random.uniform(-0.4, 0.4) * oh)
            x0 = max(0, cx - ow // 2); y0 = max(0, cy - oh // 2)
            x1 = min(W, x0 + ow);      y1 = min(H, y0 + oh)
            if x1 <= x0 or y1 <= y0:
                continue
            pw, ph = x1 - x0, y1 - y0
            mode = random.random()
            patch = None
            if mode < 0.7:
                # 현실적 질감: 다른 임의 이미지에서 크롭 → 리사이즈
                try:
                    src = self.samples[random.randrange(len(self.samples))]
                    s = PILImage.open(src['image_path']).convert('RGB')
                    sw, sh = s.size
                    if sw > pw + 2 and sh > ph + 2:
                        cw = min(sw, max(pw, int(pw * random.uniform(1.0, 3.0))))
                        ch = min(sh, max(ph, int(ph * random.uniform(1.0, 3.0))))
                        sx = random.randint(0, sw - cw); sy = random.randint(0, sh - ch)
                        s = s.crop((sx, sy, sx + cw, sy + ch)).resize((pw, ph), PILImage.BILINEAR)
                        patch = np.asarray(s)
                except Exception:
                    patch = None
            if patch is None:
                # 흑/단색 (벤치의 흑박스 프로토콜 일반화 유지)
                if random.random() < 0.5:
                    patch = np.zeros((ph, pw, 3), dtype=np.uint8)
                else:
                    col = np.array([random.randint(0, 255) for _ in range(3)], dtype=np.uint8)
                    patch = np.broadcast_to(col, (ph, pw, 3))
            img_np[y0:y1, x0:x1] = patch
        return img_np

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample_info = self.samples[idx]

        # 이미지 로드
        image = PILImage.open(sample_info['image_path']).convert('RGB')
        original_size = image.size  # (W, H)

        # Keypoint 로드
        keypoints_data = self._load_keypoints_from_json(sample_info['annotation_path'])
        keypoints = keypoints_data['projections'].copy()  # (N, 2) [x, y]

        # RootNet geometric depth prior (HoRoPose lib/core/function.py:88-97):
        #     k = sqrt(fx*fy*1000*1000 / max(|bbox_w|,|bbox_h|)^2)   [mm]
        # ⚠️ MUST be computed HERE — on the ORIGINAL-frame keypoint bbox with the ORIGINAL K,
        # i.e. before the crop_to_robot block (which resizes the crop to image_size and thereby
        # NORMALIZES apparent size away) and before augmentation (whose ShiftScaleRotate would
        # rescale the robot without changing its true depth). This is the only apparent-size ->
        # depth cue the rotation head gets; computing it downstream silently destroys it.
        # fx,fy come from _camera_settings.json, NOT from camera_K (which is eye(3) on DREAM —
        # see _intrinsics_for). With fx=fy=1 the prior collapses to ~5mm instead of ~1.3m.
        _W0, _H0 = original_size
        _fxfy = self._intrinsics_for(sample_info['annotation_path'])
        _inb0 = ((keypoints[:, 0] >= 0) & (keypoints[:, 0] < _W0) &
                 (keypoints[:, 1] >= 0) & (keypoints[:, 1] < _H0))
        if _fxfy is not None and _inb0.sum() >= 2:
            _p = keypoints[_inb0]
            _side = max(float(_p[:, 0].max() - _p[:, 0].min()),
                        float(_p[:, 1].max() - _p[:, 1].min()))
            _area = max(_side ** 2, 1.0)
            k_value = float(np.sqrt(_fxfy[0] * _fxfy[1] * 1e6 / _area))
        else:
            k_value = 0.0                              # undetermined -> consumer must mask

        # Robot-centered SQUARE crop (more pixels on small/foreshortened robots -> better keypoints,
        # esp. base-yaw J0). Crop around the in-frame GT keypoints + margin; adjust keypoints & K
        # (principal-point shift). Square -> no aspect distortion. Scale/center jitter when training.
        if self.crop_to_robot:
            W0, H0 = original_size
            inb = ((keypoints[:, 0] >= 0) & (keypoints[:, 0] < W0) &
                   (keypoints[:, 1] >= 0) & (keypoints[:, 1] < H0))
            if inb.sum() >= 2:
                pts = keypoints[inb]
                cx = (pts[:, 0].min() + pts[:, 0].max()) / 2.0
                cy = (pts[:, 1].min() + pts[:, 1].max()) / 2.0
                dx = pts[:, 0].max() - pts[:, 0].min()
                dy = pts[:, 1].max() - pts[:, 1].min()
                marg = random.uniform(1.3, self.crop_margin) if self.augment else self.crop_margin
                # 종횡비 a 인 크롭. a=1.0 이면 아래 식은 기존 정사각 코드와 완전히 동일하고
                # (max(dx, 1.0*dy) == max(dx,dy), side_h == side_w) 난수 소비 순서도 같다 —
                # 기존 설정의 재현성이 보장된다. a>1 이면 배포의 '512공간 정사각 = 원본 4:3' 기하를
                # 원본 공간에서 그대로 재현한다: w = marg*max(dx, a*dy), h = w/a.
                a = self.crop_aspect
                if self.augment and self.crop_aspect_jitter > 0:
                    a *= math.exp(random.uniform(-1.0, 1.0) * math.log(1.0 + self.crop_aspect_jitter))
                side_w = max(max(dx, a * dy) * marg, 16.0)
                if self.augment:
                    side_w *= random.uniform(0.9, 1.1)
                side_h = side_w / a
                # det_C 프레임-경계 증강: 크롭을 강하게 축소+편심 → 로봇 일부가 프레임 밖으로.
                # frame_boundary_prob==0 이면 `> 0` 에서 단락 → random 미소비 → 기존 config 재현성 보존.
                if (self.augment and self.frame_boundary_prob > 0
                        and random.random() < self.frame_boundary_prob):
                    shrink = random.uniform(0.62, 0.85)
                    side_w *= shrink; side_h *= shrink
                    ang = random.uniform(0.0, 2.0 * math.pi)
                    mag = random.uniform(0.22, 0.48)
                    cx += math.cos(ang) * mag * side_w
                    cy += math.sin(ang) * mag * side_h
                elif self.augment:
                    cx += random.uniform(-0.1, 0.1) * side_w
                    cy += random.uniform(-0.1, 0.1) * side_h
                bx0 = int(round(cx - side_w / 2.0)); by0 = int(round(cy - side_h / 2.0))
                bw = max(1, int(round(side_w))); bh = max(1, int(round(side_h)))
                image = image.crop((bx0, by0, bx0 + bw, by0 + bh))  # out-of-bounds -> black pad
                keypoints = keypoints.copy()
                keypoints[:, 0] -= bx0; keypoints[:, 1] -= by0
                if 'camera_K' in keypoints_data:
                    keypoints_data['camera_K'][0, 2] -= bx0
                    keypoints_data['camera_K'][1, 2] -= by0
                original_size = (bw, bh)

        # FDA augmentation (applied before other augmentations)
        if self.fda_real_paths and random.random() < self.fda_prob:
            real_path = random.choice(self.fda_real_paths)
            try:
                real_img = np.array(PILImage.open(real_path).convert('RGB'))
                src_img = np.array(image)
                image = PILImage.fromarray(fda_transfer(src_img, real_img, beta=self.fda_beta))
            except Exception:
                pass  # Skip FDA on error, use original image

        # det_C 현실적 distractor 가림: albumentations(광학/기하) 이전에 붙여 함께 블렌딩된다.
        # distractor_occ_prob==0 이면 `> 0` 단락 → random 미소비 → 기존 config 재현성 보존.
        if (self.augment and self.distractor_occ_prob > 0
                and random.random() < self.distractor_occ_prob and len(keypoints) > 0):
            image = PILImage.fromarray(
                self._paste_distractors(np.array(image), keypoints))

        # 데이터 증강 적용
        if self.augment and len(keypoints) > 0:
            augmented = self.augmentation(
                image=np.array(image),
                keypoints=keypoints
            )
            aug_kps = np.array(augmented['keypoints'])
            # albumentations can DUPLICATE keypoints when several are coincident (e.g. Meca500
            # spherical-wrist kp4≡kp5), breaking the index↔keypoint correspondence. remove_invisible
            # is False so a legitimate count change never happens -> a mismatch means the bug; skip aug.
            if aug_kps.shape[0] == keypoints.shape[0]:
                image = PILImage.fromarray(augmented['image'])
                keypoints = aug_kps

        # 유효 입력 해상도 지터: 작은 로봇은 크롭이 크게 업샘플되어 흐릿한 상태로 들어온다
        # (bbox 대각 73px -> 512 크롭 = ~5x 업샘플). 큰 로봇 크롭을 저해상도로 리샘플해
        # 그 영역의 학습 표본을 늘린다. 이미지 '내용'만 흐려질 뿐 기하는 그대로이고
        # original_size 를 건드리지 않으므로 keypoint 매핑은 불변이다 (transform 이 512로 복원).
        if self.augment and self.crop_res_jitter > 0 and random.random() < self.crop_res_jitter:
            e = int(random.uniform(*self.crop_res_range))
            image = image.resize((max(8, e), max(8, e)), PILImage.BILINEAR)

        # 이미지 크기 변경에 따른 keypoint 좌표 조정
        scale_x = self.heatmap_size[1] / original_size[0]
        scale_y = self.heatmap_size[0] / original_size[1]
        keypoints_scaled = keypoints.copy()
        keypoints_scaled[:, 0] *= scale_x
        keypoints_scaled[:, 1] *= scale_y

        # 이미지 변환
        image_tensor = self.transform(image)

        # Heatmap 생성
        heatmaps = self._create_heatmap(keypoints_scaled, self.heatmap_size)
        heatmaps_tensor = torch.from_numpy(heatmaps).float()

        # Keypoint 좌표를 텐서로
        keypoints_tensor = torch.from_numpy(keypoints_scaled).float()
        keypoints_3d_tensor = torch.from_numpy(keypoints_data['locations']).float()

        # Synthetic data (DREAM sim) uses cm, convert to meters
        if sample_info.get('is_synthetic', False):
            keypoints_3d_tensor = keypoints_3d_tensor / 100.0

        # Create valid mask (True for keypoints with valid coordinates)
        valid_mask = torch.tensor([kp[0] >= 0 and kp[1] >= 0 for kp in keypoints_scaled], dtype=torch.bool)

        # Get robot type from sample info
        robot_type = sample_info.get('robot_type', 0)  # Default to franka_panda if not found

        # Camera intrinsic matrix (original resolution, will be scaled in train_3d.py)
        camera_K = torch.from_numpy(keypoints_data['camera_K']).float()  # (3, 3) - original resolution

        # 🚀 [NEW] Extract depths from 3D keypoints (Z-coordinate in camera frame)
        depths = keypoints_3d_tensor[:, 2]  # (num_joints,) - Z values in meters

        sample = {
            'image': image_tensor,
            'heatmaps': heatmaps_tensor,
            'keypoints': keypoints_tensor,
            'keypoints_3d': keypoints_3d_tensor,
            'depths': depths,  # 🚀 [NEW] Depth ground truth
            'valid_mask': valid_mask,
            'robot_type': torch.tensor(robot_type, dtype=torch.long),
            'name': sample_info['name'],
            'annotation_path': sample_info['annotation_path'],
            'camera_K': camera_K,
            'original_size': torch.tensor([original_size[0], original_size[1]], dtype=torch.float32),  # (W, H)
            # RootNet depth prior in mm, ORIGINAL frame (pre-crop, pre-augment). 0 = undetermined.
            'k_value': torch.tensor(k_value, dtype=torch.float32),
        }

        # Joint angles 포함 (있는 경우)
        if self.include_angles and 'angles' in keypoints_data:
            angles = keypoints_data['angles']
            sample['angles'] = torch.from_numpy(angles).float()
            sample['has_angles'] = torch.tensor(True, dtype=torch.bool)
        else:
            # Dummy angles (모델이 angle 출력을 요구하는 경우)
            sample['angles'] = torch.zeros(7).float()  # 7 joint angles for Panda
            sample['has_angles'] = torch.tensor(False, dtype=torch.bool)

        return sample


def create_dataloaders(
    train_dir: str,
    val_dir: str,
    keypoint_names: List[str],
    batch_size: int = 8,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (512, 512),
    heatmap_size: Tuple[int, int] = (512, 512),
    val_split: float = 1.0,
    **kwargs
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Train/Validation 데이터로더 생성

    Args:
        train_dir: 학습 데이터 디렉토리
        val_dir: 검증 데이터 디렉토리
        keypoint_names: 키포인트 이름 리스트
        batch_size: 배치 크기
        num_workers: 데이터 로딩 워커 수
        image_size: 입력 이미지 크기
        heatmap_size: 출력 heatmap 크기
        val_split: 검증 데이터 사용 비율 (0.0~1.0, default=1.0 for all data)

    Returns:
        train_loader, val_loader
    """
    train_dataset = PoseEstimationDataset(
        data_dir=train_dir,
        keypoint_names=keypoint_names,
        image_size=image_size,
        heatmap_size=heatmap_size,
        augment=True,
        **kwargs
    )

    val_dataset_full = PoseEstimationDataset(
        data_dir=val_dir,
        keypoint_names=keypoint_names,
        image_size=image_size,
        heatmap_size=heatmap_size,
        augment=False,
        **kwargs
    )

    # Use only a fraction of validation data if val_split < 1.0
    if val_split < 1.0:
        val_size = int(len(val_dataset_full) * val_split)
        unused_size = len(val_dataset_full) - val_size
        generator = torch.Generator().manual_seed(42)
        val_dataset, _ = torch.utils.data.random_split(
            val_dataset_full, [val_size, unused_size], generator=generator
        )
    else:
        val_dataset = val_dataset_full

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


if __name__ == "__main__":
    # 테스트 코드
    keypoint_names = [
        'panda_link0', 'panda_link2', 'panda_link3',
        'panda_link4', 'panda_link6', 'panda_link7', 'panda_hand'
    ]

    dataset = PoseEstimationDataset(
        data_dir="/path/to/your/data",
        keypoint_names=keypoint_names,
        augment=True
    )

    print(f"Dataset size: {len(dataset)}")

    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Image shape: {sample['image'].shape}")
        print(f"Heatmaps shape: {sample['heatmaps'].shape}")
        print(f"Keypoints shape: {sample['keypoints'].shape}")
        if 'angles' in sample:
            print(f"Angles shape: {sample['angles'].shape}")
