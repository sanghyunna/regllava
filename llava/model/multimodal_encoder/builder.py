# llava/model/multimodal_encoder/builder.py
import os
from .clip_encoder import CLIPVisionTower

def build_vision_tower(vision_tower_cfg, training_args=None, **kwargs):
    """
    vision_tower_cfg : ModelArguments 인스턴스
    training_args    : transformers.TrainingArguments 인스턴스 (bf16, fp16, device_map 등을 포함)
    kwargs           : delay_load 등 추가 옵션
    """
    # 1) 모델 인자에 fp16/bf16/device_map 주입
    if training_args is not None:
        setattr(vision_tower_cfg, 'fp16', getattr(training_args, 'fp16', False))
        setattr(vision_tower_cfg, 'bf16', getattr(training_args, 'bf16', False))
        setattr(vision_tower_cfg, 'device_map', getattr(training_args, 'device_map', None))

    # 2) 실제 vision_tower 문자열(경로 또는 HF ID) 확보
    vision_tower = getattr(
        vision_tower_cfg,
        'mm_vision_tower',
        getattr(vision_tower_cfg, 'vision_tower', None)
    )
    if vision_tower is None:
        raise ValueError("`vision_tower_cfg` 에 vision_tower 경로가 정의돼 있지 않습니다.")

    is_absolute_path = os.path.exists(vision_tower)

    # 3) Reg-Gated, OpenAI, LAION, ShareGPT4V 모델만 지원
    if (
        is_absolute_path
        or vision_tower.startswith("openai")
        or vision_tower.startswith("laion")
        or "ShareGPT4V" in vision_tower
    ):
        # CLIPVisionTower 시그니처: (vision_tower_path_or_name, args, delay_load=False)
        return CLIPVisionTower(
            vision_tower,        # 경로 또는 HF 모델명
            vision_tower_cfg,    # 업데이트된 ModelArguments (args.fp16, args.bf16 포함)
            **kwargs             # delay_load 등 추가 옵션
        )

    raise ValueError(f"Unknown vision tower: {vision_tower}")
