#!/bin/bash

# DINOv3 Pose Estimation Training Example Script
# DREAM 학습 방식을 참고한 실행 스크립트
# Multi-GPU training support with PyTorch DistributedDataParallel

# ============================================================================
# Configuration
# ============================================================================

# Multi-GPU 설정
NUM_GPUS=5  # 사용할 GPU 개수 (single GPU는 1로 설정)
GPU_IDS="0,1,2,3,4"  # 사용할 GPU ID (예: "0,1,2,3")

# 데이터 경로 (multi-robot unified model)
DATA_DIR="/home/najo/NAS/DIP/DINObotPose/DREAM/data"
VAL_DIR=""  # 비워두면 DATA_DIR를 split

# Multi-robot 학습 설정
MULTI_ROBOT="--multi-robot"  # 다중 로봇 데이터 통합 학습
ROBOT_TYPES="--robot-types panda"  # 특정 로봇만 사용 (panda kuka baxter), 비우면 모두 사용

# 출력 디렉토리
OUTPUT_DIR="./outputs/dinov2_base_multi_robot_$(date +%Y%m%d_%H%M%S)"

# 모델 설정
MODEL_NAME='facebook/dinov3-vitb16-pretrain-lvd1689m'  # dinov2-small, dinov2-base, dinov2-large
IMAGE_SIZE=512
HEATMAP_SIZE=512

# 학습 설정
EPOCHS=100
BATCH_SIZE=8
NUM_WORKERS=4

# Optimizer 설정
OPTIMIZER="adam"
LEARNING_RATE=0.0001
WEIGHT_DECAY=0.00001

# Scheduler 설정 (Cosine Annealing)
SCHEDULER="cosine"
MIN_LR=1e-8  # 최소 learning rate (cosine scheduler용)

# StepLR 설정 (SCHEDULER="step"일 때만 사용)
LR_STEP_SIZE=30
LR_GAMMA=0.1

# Loss 가중치
HEATMAP_WEIGHT=1.0
ANGLE_WEIGHT=0.1

# Random seed
SEED=42

# Resume (비워두면 처음부터 학습)
RESUME=""

# ============================================================================
# Run Training
# ============================================================================

echo "=========================================="
echo "DINOv3 Pose Estimation Training"
echo "=========================================="
echo "Model: $MODEL_NAME"
echo "Data: $DATA_DIR"
echo "Output: $OUTPUT_DIR"
echo "Epochs: $EPOCHS"
echo "Batch size per GPU: $BATCH_SIZE"
echo "Number of GPUs: $NUM_GPUS"
echo "GPU IDs: $GPU_IDS"
echo "=========================================="

# 출력 디렉토리 생성
mkdir -p "$OUTPUT_DIR"

# Build training arguments
TRAIN_ARGS="--data-dir $DATA_DIR"

if [ -z "$VAL_DIR" ]; then
    TRAIN_ARGS="$TRAIN_ARGS --train-split 0.9"
else
    TRAIN_ARGS="$TRAIN_ARGS --val-dir $VAL_DIR"
fi

TRAIN_ARGS="$TRAIN_ARGS $MULTI_ROBOT $ROBOT_TYPES"
TRAIN_ARGS="$TRAIN_ARGS --model-name $MODEL_NAME"
TRAIN_ARGS="$TRAIN_ARGS --image-size $IMAGE_SIZE"
TRAIN_ARGS="$TRAIN_ARGS --heatmap-size $HEATMAP_SIZE"
TRAIN_ARGS="$TRAIN_ARGS --epochs $EPOCHS"
TRAIN_ARGS="$TRAIN_ARGS --batch-size $BATCH_SIZE"
TRAIN_ARGS="$TRAIN_ARGS --num-workers $NUM_WORKERS"
TRAIN_ARGS="$TRAIN_ARGS --optimizer $OPTIMIZER"
TRAIN_ARGS="$TRAIN_ARGS --learning-rate $LEARNING_RATE"
TRAIN_ARGS="$TRAIN_ARGS --weight-decay $WEIGHT_DECAY"
TRAIN_ARGS="$TRAIN_ARGS --scheduler $SCHEDULER"
TRAIN_ARGS="$TRAIN_ARGS --min-lr $MIN_LR"
TRAIN_ARGS="$TRAIN_ARGS --lr-step-size $LR_STEP_SIZE"
TRAIN_ARGS="$TRAIN_ARGS --lr-gamma $LR_GAMMA"
TRAIN_ARGS="$TRAIN_ARGS --heatmap-weight $HEATMAP_WEIGHT"
TRAIN_ARGS="$TRAIN_ARGS --angle-weight $ANGLE_WEIGHT"
TRAIN_ARGS="$TRAIN_ARGS --output-dir $OUTPUT_DIR"
TRAIN_ARGS="$TRAIN_ARGS --seed $SEED"

if [ -n "$RESUME" ]; then
    TRAIN_ARGS="$TRAIN_ARGS --resume $RESUME"
fi

# 학습 실행
if [ $NUM_GPUS -eq 1 ]; then
    # Single GPU training
    echo "Running single GPU training..."
    CUDA_VISIBLE_DEVICES=$GPU_IDS python train.py $TRAIN_ARGS
else
    # Multi-GPU distributed training with torchrun
    echo "Running multi-GPU distributed training with $NUM_GPUS GPUs..."
    CUDA_VISIBLE_DEVICES=$GPU_IDS torchrun \
        --standalone \
        --nnodes=1 \
        --nproc_per_node=$NUM_GPUS \
        train.py $TRAIN_ARGS
fi

echo "=========================================="
echo "Training completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="

# Wandb 대시보드에서 결과 확인
# https://wandb.ai/your-username/dinov3-pose-estimation
