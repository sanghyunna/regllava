#!/bin/bash
# 스크립트 실행 예: 
# bash scripts/v1_5/pretrain_projector_regclip.sh

# ★★★ 사용할 GPU ID 설정 (예: 1번, 2번, 3번 GPU만 사용) ★★★
# 또는 export CUDA_VISIBLE_DEVICES=0,1 # 만약 0번, 1번 GPU를 사용하고 싶다면

# LLaVA 프로젝트 루트 디렉토리 (스크립트 위치 기준으로 상위 2단계)
LLAVA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
echo "LLaVA Project Root: ${LLAVA_ROOT}"

# --- 자동 GPU 수 감지 (CUDA_VISIBLE_DEVICES 반영) ---
PYTHON_EXEC_FOR_GPU_COUNT="python"
# 이제 이 Python 호출은 위에서 설정한 CUDA_VISIBLE_DEVICES의 영향을 받습니다.
NUM_GPUS_TO_USE=$($PYTHON_EXEC_FOR_GPU_COUNT -c "import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)")

if [ "$NUM_GPUS_TO_USE" -eq 0 ]; then
    echo "오류: CUDA_VISIBLE_DEVICES에 설정된 GPU를 찾을 수 없거나 사용 가능한 CUDA GPU가 없습니다."
    echo "CUDA_VISIBLE_DEVICES 설정 (${CUDA_VISIBLE_DEVICES}) 및 CUDA 환경을 확인하세요."
    exit 1
fi
echo "사용할 GPU ID: ${CUDA_VISIBLE_DEVICES}"
echo "사용할 GPU 수 (CUDA_VISIBLE_DEVICES 기준): ${NUM_GPUS_TO_USE}"
# --- 자동 GPU 수 감지 끝 ---


# --- 사용자 설정 변수 ---
# 1. LLM (Vicuna) 경로 또는 HF ID
MODEL_BASE="lmsys/vicuna-7b-v1.5"

# 2. Reg-Gated CLIP 체크포인트 파일의 절대 경로
CLIP_MODEL_NAME_OR_PATH="zer0int/CLIP-Registers-Gated_MLP-ViT-L-14"
CLIP_CHECKPOINT_FILENAME="ViT-L-14-REG-GATED-balanced-ckpt12.safetensors"
VISION_TOWER_PATH="${LLAVA_ROOT}/models/${CLIP_CHECKPOINT_FILENAME}" 

# 3. 학습 데이터 경로 (LLaVA-Pretrain 558k 데이터셋 사용 예시)
DATA_ROOT_DIR="${LLAVA_ROOT}/LLaVA-Pretrain" 
PRETRAIN_DATA_JSON="${DATA_ROOT_DIR}/blip_laion_cc_sbu_558k.json"
PRETRAIN_IMAGE_FOLDER="${DATA_ROOT_DIR}/images"

# 4. 학습 결과 저장 디렉토리
OUTPUT_DIR="${LLAVA_ROOT}/checkpoints/llava-$(basename ${MODEL_BASE})-regclip-projector-stage1"

# 5. DeepSpeed 설정 파일 경로
DEEPSPEED_CONFIG_PATH="${LLAVA_ROOT}/scripts/zero1_projector_tuning.json"

# 6. 학습 하이퍼파라미터
# NUM_GPUS 변수명 변경 및 자동 감지된 값 사용
# NUM_GPUS=${NUM_GPUS_TO_USE} # 이 변수는 DeepSpeed 명령어에서 직접 사용
PER_DEVICE_TRAIN_BATCH_SIZE=16 
GRADIENT_ACCUMULATION_STEPS=2  
LEARNING_RATE=1e-3 
NUM_TRAIN_EPOCHS=2
WEIGHT_DECAY=0.
WARMUP_RATIO=0.03
LR_SCHEDULER_TYPE="cosine"
LOGGING_STEPS=1
SAVE_STEPS=100 
SAVE_TOTAL_LIMIT=2 
MODEL_MAX_LENGTH=2048
REPORT_TO="wandb" 

# --- 설정 변수 끝 ---


echo "--- Environment Diagnostics ---"
echo "[[ Bash version ]]: $BASH_VERSION"
echo "[[ which ld ]]: $(which ld)"
echo "[[ ld --version ]]: $(ld --version 2>/dev/null || echo 'ld version check failed')"
echo "[[ which gcc ]]: $(which gcc)"
echo "[[ gcc --version ]]: $(gcc --version 2>/dev/null || echo 'gcc version check failed')"
echo "[[ which nvcc ]]: $(which nvcc)"
echo "[[ nvcc --version ]]: $(nvcc --version 2>/dev/null || echo 'nvcc version check failed')"
echo "[[ CUDA_HOME ]]: $CUDA_HOME"
echo "[[ CUDA_VISIBLE_DEVICES ]]: $CUDA_VISIBLE_DEVICES"
echo "[[ LD_LIBRARY_PATH ]]: $LD_LIBRARY_PATH"
echo "[[ Python version ]]: $(python --version 2>&1)"
echo "[[ which python ]]: $(which python)"
echo "[[ PyTorch version ]]: $(python -c 'import torch; print(torch.__version__)')"
echo "[[ PyTorch CUDA available ]]: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "[[ PyTorch CUDA version ]]: $(python -c 'import torch; print(torch.version.cuda)')"
echo "[[ Number of GPUs (torch) ]]: $(python -c 'import torch; print(torch.cuda.device_count())')"
echo "[[ DeepSpeed version ]]: $(python -c 'import deepspeed; print(deepspeed.__version__)' 2>/dev/null || echo 'DeepSpeed not found or import error')"
echo "--- End Environment Diagnostics ---"


# --- 사전 준비 ---
# (이전과 동일한 사전 준비 로직)
if [ ! -f "$VISION_TOWER_PATH" ]; then
    echo "Reg-Gated CLIP 체크포인트 파일(${VISION_TOWER_PATH})을 찾을 수 없습니다. 다운로드를 시도합니다..."
    PYTHON_EXEC_FOR_DOWNLOAD="python" 
    DOWNLOAD_SCRIPT_CONTENT="
from huggingface_hub import snapshot_download
from pathlib import Path
model_dir = Path('${LLAVA_ROOT}/models')
model_dir.mkdir(exist_ok=True)
snapshot_download(
    '${CLIP_MODEL_NAME_OR_PATH}',
    local_dir=model_dir,
    allow_patterns=['${CLIP_CHECKPOINT_FILENAME}'],
)
print(f'다운로드 완료: {model_dir}/${CLIP_CHECKPOINT_FILENAME}')
"
    if command -v $PYTHON_EXEC_FOR_DOWNLOAD &> /dev/null
    then
        echo "Python을 사용하여 Hugging Face Hub에서 CLIP 모델 다운로드 중..."
        $PYTHON_EXEC_FOR_DOWNLOAD -c "${DOWNLOAD_SCRIPT_CONTENT}"
        if [ ! -f "$VISION_TOWER_PATH" ]; then
            echo "오류: CLIP 모델 다운로드에 실패했습니다. VISION_TOWER_PATH를 수동으로 설정해주세요."
            exit 1
        fi
    else
        echo "오류: Python 실행 파일을 찾을 수 없습니다. CLIP 모델을 수동으로 다운로드하고 VISION_TOWER_PATH를 설정해주세요."
        exit 1
    fi
else
    echo "Reg-Gated CLIP 체크포인트 파일 확인: ${VISION_TOWER_PATH}"
fi

if [ ! -f "$PRETRAIN_DATA_JSON" ]; then
  echo "오류: 학습 데이터 JSON 파일을 찾을 수 없습니다: $PRETRAIN_DATA_JSON"
  exit 1
fi
if [ ! -d "$PRETRAIN_IMAGE_FOLDER" ]; then
  echo "오류: 이미지 폴더를 찾을 수 없습니다: $PRETRAIN_IMAGE_FOLDER"
  exit 1
fi

if [ ! -f "$DEEPSPEED_CONFIG_PATH" ]; then
    echo "경고: DeepSpeed 설정 파일(${DEEPSPEED_CONFIG_PATH})을 찾을 수 없습니다."
fi

mkdir -p ${OUTPUT_DIR}
# --- 사전 준비 끝 ---

# --- 학습 실행 ---
# NUM_GPUS_TO_USE 변수를 직접 사용
EFFECTIVE_BATCH_SIZE=$((NUM_GPUS_TO_USE * PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))

echo "LLM 경로: $MODEL_BASE"
echo "비전 타워 체크포인트 경로: $VISION_TOWER_PATH"
echo "학습 데이터 JSON: $PRETRAIN_DATA_JSON"
echo "이미지 폴더: $PRETRAIN_IMAGE_FOLDER"
echo "출력 디렉토리: $OUTPUT_DIR"
echo "실제 사용할 GPU 수 (CUDA_VISIBLE_DEVICES 기준): $NUM_GPUS_TO_USE" # 변경된 변수명 사용
echo "디바이스당 배치 크기: $PER_DEVICE_TRAIN_BATCH_SIZE"
echo "그래디언트 축적 스텝: $GRADIENT_ACCUMULATION_STEPS"
echo "유효 배치 크기: $EFFECTIVE_BATCH_SIZE"
echo "프로젝터 학습률: $LEARNING_RATE"

# deepspeed 직접 실행
# --num_gpus 인자에 자동 감지된 NUM_GPUS_TO_USE 변수 사용
# CUDA_VISIBLE_DEVICES는 이미 스크립트 시작 시 export 되어 자식 프로세스(deepspeed)에 전달됨
deepspeed --master_port $((29500 + RANDOM % 100)) \
    "${LLAVA_ROOT}/llava/train/train_mem.py" \
    --deepspeed "${DEEPSPEED_CONFIG_PATH}" \
    --model_name_or_path "${MODEL_BASE}" \
    --version v1 \
    --data_path "${PRETRAIN_DATA_JSON}" \
    --image_folder "${PRETRAIN_IMAGE_FOLDER}" \
    --vision_tower "${VISION_TOWER_PATH}" \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_vision_select_feature patch \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs ${NUM_TRAIN_EPOCHS} \
    --per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps ${SAVE_STEPS} \
    --save_total_limit ${SAVE_TOTAL_LIMIT} \
    --learning_rate ${LEARNING_RATE} \
    --weight_decay ${WEIGHT_DECAY} \
    --warmup_ratio ${WARMUP_RATIO} \
    --lr_scheduler_type "${LR_SCHEDULER_TYPE}" \
    --logging_steps ${LOGGING_STEPS} \
    --tf32 True \
    --model_max_length ${MODEL_MAX_LENGTH} \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess False \
    --report_to "${REPORT_TO}" \
    --tune_mm_mlp_adapter True

echo "프로젝터 파인튜닝 완료. 결과는 ${OUTPUT_DIR} 에 저장되었습니다."