# ------------------------------------------------------------
# 0. 기본 환경 세팅
# ------------------------------------------------------------
import os

import json, os, sys, math, warnings
from pathlib import Path
import torch, torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np, cv2
from PIL import Image
from io import BytesIO
import requests

from transformers import TextStreamer
from llava.constants import (
    IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
)
from llava.conversation import conv_templates
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images, tokenizer_image_token, get_model_name_from_path
)
from llava.model.builder import load_pretrained_model

# ------------------------------------------------------------
# 1. 사용자 파라미터
# ------------------------------------------------------------
ROOT = Path().cwd()                             # 스크립트 위치
MODEL_DIR   = ROOT / "llava-v1.5-7b-local"      # ← config, projector 위치
BASE_LLM    = "lmsys/vicuna-7b-v1.5"            # Vicuna‑7B
IMAGE_PATH  = ROOT / "data/car.jpg"
PROMPT_TXT  = "Describe where the main object is located in the image."

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
MAX_TOKENS  = 256
TEMPERATURE = 0.2
VISUALIZE   = True          # heat‑map on/off

# ------------------------------------------------------------
# 2. 모델 로드
# ------------------------------------------------------------
disable_torch_init()

model_name = get_model_name_from_path(str(MODEL_DIR))
tokenizer, model, image_processor, _ctx_len = load_pretrained_model(
    str(MODEL_DIR),
    model_base = BASE_LLM,
    model_name = model_name,
    device_map = "auto",        # 여러 GPU 자동 분산
    attn_implementation = "eager",
    torch_dtype = torch.float16
)
main_device = next(model.parameters()).device
print(f"[INFO] model loaded → dtype {model.dtype}, main device {main_device}")

# projector 추가 로딩 여부
cfg_path = MODEL_DIR / "config.json"
with open(cfg_path, "r") as f:
    cfg_json = json.load(f)

proj_path = MODEL_DIR / "mm_projector.bin"
if cfg_json.get("use_mm_proj", True) and proj_path.exists():
    print(f"[INFO] loading projector weights from {proj_path}")
    proj_state = torch.load(proj_path, map_location="cpu")
    # 키 필터링
    proj_state = {k.split("mm_projector.",1)[-1]: v.to(model.dtype) 
                  for k,v in proj_state.items() if "mm_projector" in k}
    model.get_model().mm_projector.load_state_dict(proj_state, strict=False)

# ------------------------------------------------------------
# 3. 이미지 전처리
# ------------------------------------------------------------
def load_image(path_or_url:str)->Image.Image:
    if str(path_or_url).startswith(("http://","https://")):
        buf = requests.get(path_or_url, timeout=10); buf.raise_for_status()
        return Image.open(BytesIO(buf.content)).convert("RGB")
    return Image.open(path_or_url).convert("RGB")

pil_img   = load_image(IMAGE_PATH)
orig_size = pil_img.size
img_tensor= process_images([pil_img], image_processor, model.config)
if not isinstance(img_tensor, list): img_tensor=[img_tensor]
img_tensor = [t.to(main_device, dtype=model.dtype) for t in img_tensor]

# ------------------------------------------------------------
# 4. 프롬프트 작성 & 토크나이즈
# ------------------------------------------------------------
conv_key = next((k for k in conv_templates if k in model_name.lower()),
                "llava_v1")
conv = conv_templates[conv_key].copy()

if model.config.mm_use_im_start_end:
    user_prompt = f"{DEFAULT_IM_START_TOKEN}{DEFAULT_IMAGE_TOKEN}{DEFAULT_IM_END_TOKEN}\n{PROMPT_TXT}"
else:
    user_prompt = f"{DEFAULT_IMAGE_TOKEN}\n{PROMPT_TXT}"
conv.append_message(conv.roles[0], user_prompt)
conv.append_message(conv.roles[1], None)
full_prompt = conv.get_prompt()

input_ids = tokenizer_image_token(
    full_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
).unsqueeze(0).to(main_device)
attention_mask = torch.ones_like(input_ids)

assert (input_ids==IMAGE_TOKEN_INDEX).sum() > 0, "<image> 토큰 누락!"

# ------------------------------------------------------------
# 5‑A.  이미지 자체 어텐션 heat‑map
# ------------------------------------------------------------
if VISUALIZE:
    with torch.no_grad():
        outs = model(
            input_ids = input_ids,
            images    = img_tensor,
            image_sizes=[orig_size],
            output_attentions=True,
            return_dict=True
        )
    # 마지막 레이어 self‑attention 합산
    attn = outs.attentions[-1].sum(dim=1)[0].cpu()     # (seq,seq)
    img_tok = (input_ids[0]==IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0].item()
    patch_len = attn.size(1) - img_tok
    patch_attn = attn[img_tok+1, img_tok:img_tok+patch_len]    # 첫 패치 query
    # 격자로 reshape
    side = int(math.sqrt(patch_len))
    patch_attn = patch_attn.float().reshape(side,side)
    hm = torch.nn.functional.interpolate(
        patch_attn[None,None],
        size=pil_img.size[::-1],
        mode="bilinear",
        align_corners=False
    )[0,0].numpy()
    plt.figure(figsize=(6,6))
    plt.title("Vision‑only self‑attention (Patch→Patch)")
    base_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    hm_norm  = (hm-hm.min())/(hm.ptp()+1e-8)
    heat_bgr = cv2.applyColorMap((hm_norm*255).astype(np.uint8), cv2.COLORMAP_JET)
    mix      = cv2.addWeighted(base_bgr,0.6, heat_bgr,0.4,0)
    plt.imshow(cv2.cvtColor(mix, cv2.COLOR_BGR2RGB)); plt.axis("off")
    plt.show()

# ------------------------------------------------------------
# 5‑B. 텍스트 생성
# ------------------------------------------------------------
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
gen_kwargs = dict(
    inputs            = input_ids,
    attention_mask    = attention_mask,
    images            = img_tensor,
    image_sizes       = [orig_size],
    do_sample         = TEMPERATURE>0,
    temperature       = TEMPERATURE,
    max_new_tokens    = MAX_TOKENS,
    streamer          = streamer,
    pad_token_id      = tokenizer.pad_token_id or tokenizer.eos_token_id,
    output_attentions = True,
    return_dict_in_generate=True
)
print(f"\n[USER] {PROMPT_TXT}\n[ASSISTANT] ", end='', flush=True)
with torch.inference_mode():
    gen_out = model.generate(**gen_kwargs)

# ------------------------------------------------------------
# 5‑C. 텍스트→이미지 cross‑attention heat‑map
# ------------------------------------------------------------
if VISUALIZE:
    # 전체 시퀀스 복원(프롬프트 + 생성)
    full_seq = torch.cat([input_ids, 
                          gen_out.sequences[:, input_ids.size(1):]], dim=1)
    with torch.no_grad():
        outs2 = model(
            input_ids = full_seq,
            images    = img_tensor,
            image_sizes=[orig_size],
            output_attentions=True,
            return_dict=True
        )
    attn2 = outs2.attentions[-1].sum(dim=1)[0].cpu()   # (seq,seq)

    txt_start = img_tok + patch_len
    txt_vec   = attn2[txt_start:, img_tok:img_tok+patch_len].mean(dim=0)

    txt_vec = txt_vec.float().reshape(side,side)
    hm2 = F.interpolate(
        txt_vec[None,None],
        size=pil_img.size[::-1],
        mode="bilinear",
        align_corners=False
    )[0,0].numpy()

    plt.figure(figsize=(6,6))
    plt.title("Text→Image cross‑attention (Prompt+Generation)")
    hm2_norm = (hm2-hm2.min())/(hm2.ptp()+1e-8)
    heat_bgr2= cv2.applyColorMap((hm2_norm*255).astype(np.uint8), cv2.COLORMAP_JET)
    mix2     = cv2.addWeighted(base_bgr,0.6, heat_bgr2,0.4,0)
    plt.imshow(cv2.cvtColor(mix2, cv2.COLOR_BGR2RGB)); plt.axis("off")
    plt.show()
