#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import os
import warnings
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from llava.model import *
from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

import os
import warnings
import shutil
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig,
    LlamaConfig, PretrainedConfig          # ← 추가 import
)
from llava.model import *
from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


def _sanitize_llava_cfg(cfg):
    """
    LLaVA는 causal‑LM 이므로 encoder‑decoder 관련 dict 속성을 정리한다.
    """
    cfg.is_encoder_decoder = False
    # decoder: dict → LlamaConfig, 아니면 제거
    dec = getattr(cfg, "decoder", None)
    if isinstance(dec, dict):
        try:
            cfg.decoder = LlamaConfig(**dec, is_decoder=True)
        except TypeError:
            delattr(cfg, "decoder")
    elif not isinstance(dec, PretrainedConfig) and hasattr(cfg, "decoder"):
        delattr(cfg, "decoder")
    # encoder: dict 제거
    if isinstance(getattr(cfg, "encoder", None), dict):
        delattr(cfg, "encoder")
    return cfg

import os, warnings, shutil, torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig,
    LlamaConfig, PretrainedConfig
)
from llava.model import *
from llava.constants import (
    DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
)

# ──────────────────────────────────────────────────────────────────────────
def _sanitize_llava_cfg(cfg):
    """
    LLaVA는 causal‑LM 이므로 encoder‑decoder 관련 dict 속성을 정리한다.
    • is_encoder_decoder → False
    • decoder / encoder / text_config / vision_config
      ─ dict → LlamaConfig(**dict) (가능할 때만)
      ─ 아니면 완전히 삭제
    """
    cfg.is_encoder_decoder = False
    for key in ("decoder", "encoder", "text_config", "vision_config"):
        if not hasattr(cfg, key):
            continue
        val = getattr(cfg, key)
        if isinstance(val, dict) and key == "decoder":
            # dict → LlamaConfig  변환 시도
            try:
                setattr(cfg, key, LlamaConfig(**val, is_decoder=True))
                continue
            except TypeError:
                pass  # 변환 실패 → 삭제
        # dict 이거나 변환 실패했으면 제거
        if isinstance(val, dict) or isinstance(val, PretrainedConfig):
            delattr(cfg, key)
# ──────────────────────────────────────────────────────────────────────────


def load_pretrained_model(
    model_path,
    model_base,
    model_name,
    load_8bit=False,
    load_4bit=False,
    device_map="auto",
    device="cuda",
    use_flash_attn=False,
    **kwargs
):
    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs["device_map"] = {"": device}

    if load_8bit:
        kwargs["load_in_8bit"] = True
    elif load_4bit:
        kwargs["load_in_4bit"] = True
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        kwargs["torch_dtype"] = torch.float16

    if use_flash_attn:
        kwargs["attn_implementation"] = "flash_attention_2"

    # ────────────────────────── LLaVA 계열 ──────────────────────────
    if "llava" in model_name.lower():
        # -- (1) LoRA + base --
        if "lora" in model_name.lower() and model_base is not None:
            from llava.model.language_model.llava_llama import LlavaConfig
            cfg = LlavaConfig.from_pretrained(model_path)
            _sanitize_llava_cfg(cfg)

            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            print("Loading LLaVA from base model...")
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_base, low_cpu_mem_usage=True, config=cfg, **kwargs
            )
            # --- 이하 원본 로직 그대로 (생략) ---
            # ...

        # -- (2) base 모델 + mm_projector --
        elif model_base is not None:
            print("Loading LLaVA from base model...")
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            cfg = AutoConfig.from_pretrained(model_path)
            _sanitize_llava_cfg(cfg)                    # ← 강화된 함수 호출
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_base, low_cpu_mem_usage=True, config=cfg, **kwargs
            )
            # --- projector 가중치 로드 (원본 그대로) ---
            mm_proj = torch.load(os.path.join(model_path, "mm_projector.bin"), map_location="cpu")
            model.load_state_dict({k: v.to(torch.float16) for k, v in mm_proj.items()}, strict=False)

        # -- (3) 단일 LLaVA 체크포인트 --
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_path, low_cpu_mem_usage=True, **kwargs
            )

    # ───────────────────────── 일반 언어 모델 ─────────────────────────
    else:
        # (원본 builder.py 로직 그대로 – 변경 없음)
        if model_base is not None:
            from peft import PeftModel
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)
            model = PeftModel.from_pretrained(model, model_path).merge_and_unload().to(torch.float16)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    # ─────────────── token 추가·vision tower 초기화 (변경 없음) ───────────────
    image_processor = None
    if "llava" in model_name.lower():
        if getattr(model.config, "mm_use_im_patch_token", True):
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if getattr(model.config, "mm_use_im_start_end", False):
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        vt = model.get_vision_tower()
        if not vt.is_loaded:
            vt.load_model(device_map=device_map)

        # ------------------------------------------------------------------
        # device_map 이 dict 인 경우  ➔  하나의 실제 device 값만 추출해 사용
        #   예) {"": "cuda:0"}  or  {"module": "cuda:1"}
        # dict 그대로 넘기면 .to() 가 TypeError 를 발생시키므로 처리 필요
        # ------------------------------------------------------------------
        if isinstance(device_map, dict):
            tgt_device = next(iter(device_map.values()))
            vt.to(device=tgt_device, dtype=torch.float16)
        elif device_map != "auto":
            vt.to(device=device_map, dtype=torch.float16)
        image_processor = vt.image_processor

    context_len = getattr(model.config, "max_sequence_length", 2048)
    return tokenizer, model, image_processor, context_len
