"""
DINOv3 Pose Estimation Dataset
DREAM 데이터셋 구조를 따르는 데이터로더
"""

import os
import json
import numpy as np
from PIL import Image as PILImage
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import albumentations as albu
from typing import Dict, List, Tuple, Optional


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
        robot_types: Optional[List[str]] = None  # List of robot types to include
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

        # 데이터 파일 리스트 로드
        self.samples = self._load_dataset()

        # 이미지 변환 설정
        if normalize:
            # ImageNet 통계값 사용 (DINOv3가 학습된 방식)
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
            ])

        # 데이터 증강 설정
        if self.augment:
            self.augmentation = albu.Compose([
                albu.GaussNoise(p=0.3),
                albu.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                albu.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.1,
                    rotate_limit=15,
                    p=0.5
                ),
                albu.HueSaturationValue(p=0.3),
            ], keypoint_params=albu.KeypointParams(format='xy', remove_invisible=False))

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

        # NDDS 형식: 각 프레임마다 이미지와 .json 파일이 쌍으로 존재
        for root, dirs, files in os.walk(directory):
            json_files = [f for f in files if f.endswith('.json')]

            for json_file in json_files:
                # 해당 JSON 파일의 이미지 파일 찾기
                base_name = json_file.replace('.json', '')
                img_path = None

                # RGB 이미지 찾기 (DREAM uses .rgb.jpg extension)
                for ext in ['.rgb.jpg', '.png', '.jpg', '.jpeg']:
                    potential_path = os.path.join(root, base_name + ext)
                    if os.path.exists(potential_path):
                        img_path = potential_path
                        break

                if img_path:
                    json_path = os.path.join(root, json_file)
                    samples.append({
                        'image_path': img_path,
                        'annotation_path': json_path,
                        'name': base_name,
                        'source_dir': os.path.basename(directory)
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

        # NDDS 형식에서 keypoint 추출
        if 'objects' in data:
            for obj in data['objects']:
                if 'keypoints' in obj:
                    # Create a mapping from name to position in JSON
                    for kp in obj['keypoints']:
                        kp_name = kp['name']
                        if kp_name in self.keypoint_names:
                            # Find the index in our ordered list
                            idx = self.keypoint_names.index(kp_name)
                            keypoint_positions[idx] = [
                                kp['projected_location'][0],
                                kp['projected_location'][1]
                            ]
                            if 'location' in kp:
                                keypoint_positions_3d[idx] = [
                                    kp['location'][0],
                                    kp['location'][1],
                                    kp['location'][2]
                                ]
                            keypoint_found[idx] = True

        # Mark missing keypoints with negative coordinates
        for i, found in enumerate(keypoint_found):
            if not found:
                keypoint_positions[i] = [-1, -1]
                keypoint_positions_3d[i] = [-1, -1, -1]

        keypoints['projections'] = keypoint_positions
        keypoints['locations'] = keypoint_positions_3d

        # Joint angles (있는 경우)
        if self.include_angles and 'joint_angles' in data:
            # Normalize angles to [-1, 1] range (assuming radians)
            # Most robot joints are within [-pi, pi]
            angles = np.array(data['joint_angles'], dtype=np.float32)
            angles = angles / np.pi  # Scale to [-1, 1] approx
            keypoints['angles'] = angles

        return keypoints

    def _create_heatmap(self, keypoints: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """
        Keypoint 위치로부터 Gaussian heatmap 생성

        Args:
            keypoints: (N, 2) keypoint 좌표 [x, y]
            size: (H, W) heatmap 크기

        Returns:
            heatmaps: (N, H, W) Gaussian heatmap
        """
        H, W = size
        num_keypoints = len(keypoints)
        heatmaps = np.zeros((num_keypoints, H, W), dtype=np.float32)

        for i, (x, y) in enumerate(keypoints):
            # 이미지 크기에 맞게 keypoint 좌표 스케일링
            if x < 0 or y < 0:  # Invalid keypoint
                continue

            # Gaussian 생성
            xx, yy = np.meshgrid(np.arange(W), np.arange(H))
            heatmap = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * self.sigma ** 2))
            heatmaps[i] = heatmap

        return heatmaps

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

        # 데이터 증강 적용
        if self.augment and len(keypoints) > 0:
            augmented = self.augmentation(
                image=np.array(image),
                keypoints=keypoints
            )
            image = PILImage.fromarray(augmented['image'])
            keypoints = np.array(augmented['keypoints'])

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

        sample = {
            'image': image_tensor,
            'heatmaps': heatmaps_tensor,
            'keypoints': keypoints_tensor,
            'keypoints_3d': keypoints_3d_tensor,
            'name': sample_info['name']
        }

        # Joint angles 포함 (있는 경우)
        if self.include_angles and 'angles' in keypoints_data:
            angles = keypoints_data['angles']
            sample['angles'] = torch.from_numpy(angles).float()
        else:
            # Dummy angles (모델이 angle 출력을 요구하는 경우)
            sample['angles'] = torch.zeros(9).float()  # NUM_ANGLES = 9

        return sample


def create_dataloaders(
    train_dir: str,
    val_dir: str,
    keypoint_names: List[str],
    batch_size: int = 8,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (512, 512),
    heatmap_size: Tuple[int, int] = (512, 512),
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

    val_dataset = PoseEstimationDataset(
        data_dir=val_dir,
        keypoint_names=keypoint_names,
        image_size=image_size,
        heatmap_size=heatmap_size,
        augment=False,
        **kwargs
    )

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
