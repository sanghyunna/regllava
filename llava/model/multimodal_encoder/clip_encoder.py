import os
import sys
from pathlib import Path
import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig

try:
    from safetensors.torch import load_file as load_safetensors
except ImportError:
    raise ImportError("safetensors 라이브러리가 필요합니다. `pip install safetensors`로 설치해주세요.")

# --- 사용자 정의 모델 관련 임포트 관리 ---
_USER_CUSTOM_CLIP_MODULES_LOADED = False
_build_custom_vision_model_func = None
_custom_image_transform_func = None

def _ensure_user_custom_clip_modules_loaded():
    global _USER_CUSTOM_CLIP_MODULES_LOADED, _build_custom_vision_model_func, _custom_image_transform_func
    if _USER_CUSTOM_CLIP_MODULES_LOADED:
        return

    try:
        # 이 파일(clip_encoder.py)은 LLaVA 프로젝트 내의 llava/model/multimodal_encoder/ 에 위치.
        # LLaVA 프로젝트 루트는 이 파일 위치에서 세 단계 상위 디렉토리.
        llava_project_root = Path(__file__).resolve().parents[3]
        inferclipregxgated_module_path = llava_project_root / "INFERclipregXGATED"

        if not inferclipregxgated_module_path.is_dir():
            # 개발/테스트 환경 등에서 LLaVA 프로젝트 루트가 sys.path에 직접 추가된 경우를 대비
            # 또는 INFERclipregXGATED가 PYTHONPATH 등을 통해 접근 가능한 경우
            print(f"정보: {inferclipregxgated_module_path} 에서 INFERclipregXGATED 디렉토리를 찾을 수 없습니다. sys.path에서 직접 임포트를 시도합니다.")
            # 이 경우, 사용자는 INFERclipregXGATED가 Python의 import 경로에 있도록 환경을 설정해야 함.
            from INFERclipregXGATED.model import build_model as bm
            from INFERclipregXGATED.clip import _transform as ct
            print("sys.path에서 INFERclipregXGATED 모듈을 성공적으로 임포트했습니다.")
        else:
            # LLaVA 프로젝트 루트를 sys.path에 추가하여 INFERclipregXGATED 모듈을 임포트
            if str(llava_project_root) not in sys.path:
                sys.path.insert(0, str(llava_project_root))
            from INFERclipregXGATED.model import build_model as bm
            from INFERclipregXGATED.clip import _transform as ct
            print(f"{inferclipregxgated_module_path} 에서 INFERclipregXGATED 모듈을 성공적으로 임포트했습니다.")

        _build_custom_vision_model_func = bm
        _custom_image_transform_func = ct
        _USER_CUSTOM_CLIP_MODULES_LOADED = True

    except ImportError as e:
        print(f"치명적 오류: 사용자 정의 Reg-Gated CLIP 모듈(INFERclipregXGATED)을 임포트할 수 없습니다: {e}")
        print("다음 사항을 확인하세요:")
        print(f"1. LLaVA 프로젝트 루트 디렉토리 내에 'INFERclipregXGATED' 폴더가 올바르게 복사되었는지 확인 (현재 예상 경로: {llava_project_root / 'INFERclipregXGATED'})")
        print("2. 'INFERclipregXGATED' 폴더 내에 '__init__.py', 'model.py', 'clip.py' 파일들이 존재하는지 확인")
        print("3. 필요한 의존성 라이브러리가 모두 설치되었는지 확인")
        raise


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower_path_or_name, args, delay_load=False):
        super().__init__()
        self.is_loaded = False
        self.vision_tower_name = vision_tower_path_or_name
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        self.args = args

        self.is_custom_reg_gated_clip = False
        if isinstance(vision_tower_path_or_name, str) and \
           ("reg-gated" in vision_tower_path_or_name.lower() or "reg_gated" in vision_tower_path_or_name.lower() or \
            vision_tower_path_or_name.endswith("REG-GATED-balanced-ckpt12.safetensors")): # 파일 이름으로도 식별
            if not os.path.isfile(vision_tower_path_or_name):
                raise FileNotFoundError(
                    f"Reg-Gated CLIP 모델로 인식되었으나, 제공된 경로가 실제 파일이 아닙니다: {vision_tower_path_or_name}. "
                    f"--vision_tower 인자에는 Reg-Gated CLIP 체크포인트(.safetensors) 파일의 전체 절대 경로를 제공해야 합니다."
                )
            self.is_custom_reg_gated_clip = True
            self.custom_model_ckpt_path = vision_tower_path_or_name
            _ensure_user_custom_clip_modules_loaded() # 사용자 모듈 로드 보장

        self.vision_tower = None # 실제 모델 파라미터
        self.image_processor = None # 이미지 전처리기

        if not delay_load:
            self.load_model(device_map=getattr(args, 'device_map', None))
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model(device_map=getattr(args, 'device_map', None))
        else:
            self._load_config_only()

    def _load_config_only(self):
        if not self.is_custom_reg_gated_clip:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)
        else:
            class MockRegGatedCLIPConfig:
                hidden_size = 1024
                image_size = 224 # 임시 값, load_model에서 실제 값으로 업데이트
                patch_size = 14  # 임시 값, load_model에서 실제 값으로 업데이트
            self.cfg_only = MockRegGatedCLIPConfig()

    def load_model(self, device_map=None):
        """
        비전 타워 파라미터와 이미지 전처리기를 메모리에 로드합니다.
        - fp16 / bf16 / device_map 등이 args에 없더라도 기본값을 사용해
          AttributeError가 발생하지 않도록 처리했습니다.
        """
        if self.is_loaded:
            print(f'{self.vision_tower_name} is already loaded, `load_model` called again, skipping.')
            return

        # ⬇️ 안전한 dtype 결정
        fp16_flag = getattr(self.args, 'fp16', False)
        bf16_flag = getattr(self.args, 'bf16', False)
        target_dtype = torch.float16 if fp16_flag else (torch.bfloat16 if bf16_flag else torch.float32)
        print(f"CLIPVisionTower 로드 시 사용할 target_dtype: {target_dtype} (fp16: {fp16_flag}, bf16: {bf16_flag})")

        # ----- (1) Reg‑Gated CLIP 로드 -----
        if self.is_custom_reg_gated_clip:
            print(f"로딩 시작: 사용자 정의 Reg-Gated CLIP 모델 ({self.custom_model_ckpt_path})")
            _ensure_user_custom_clip_modules_loaded()

            full_clip_model = _build_custom_vision_model_func(load_safetensors(self.custom_model_ckpt_path))
            loaded_vision_tower = full_clip_model.visual.to(dtype=target_dtype)
            loaded_vision_tower.proj = None          # 출력 차원 1024

            for m in loaded_vision_tower.modules():
                if isinstance(m, nn.LayerNorm):
                    m.float()                        # LayerNorm은 float32 유지

            self.vision_tower = loaded_vision_tower
            input_resolution       = self.vision_tower.input_resolution
            self.actual_patch_size = getattr(self.vision_tower, 'patch_size', 14)

            # 전처리기 구성
            class _CustomRegGatedCLIPImageProcessor:
                def __init__(self, transform_fn, resolution, model_args):
                    self.transform_fn = transform_fn
                    self.crop_size    = {'height': resolution, 'width': resolution}
                    self.size         = {'shortest_edge': resolution}
                    self.image_mean   = getattr(model_args, 'image_mean', [0.48145466, 0.4578275, 0.40821073])
                    self.image_std    = getattr(model_args, 'image_std',  [0.26862954, 0.26130258, 0.27577711])

                def preprocess(self, images, return_tensors='pt'):
                    img_list = images if isinstance(images, list) else [images]
                    tensors  = [self.transform_fn(img.convert('RGB')) for img in img_list]
                    if return_tensors == 'pt':
                        return {'pixel_values': torch.stack(tensors)}
                    raise ValueError(f"Unsupported return_tensors type: {return_tensors}")

            self.image_processor = _CustomRegGatedCLIPImageProcessor(
                _custom_image_transform_func(input_resolution),
                input_resolution,
                self.args
            )

            # cfg_only(모델 메타) 보정
            if hasattr(self, 'cfg_only') and not isinstance(self.cfg_only, CLIPVisionConfig):
                self.cfg_only.image_size   = input_resolution
                self.cfg_only.patch_size   = self.actual_patch_size
                self.cfg_only.hidden_size  = 1024

            print(f"사용자 정의 Reg‑Gated CLIP 모델 로드 완료. 입력 해상도: {input_resolution}, 패치 크기: {self.actual_patch_size}")

        # ----- (2) 표준 HuggingFace CLIP 로드 -----
        else:
            self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
            self.vision_tower    = CLIPVisionModel.from_pretrained(
                self.vision_tower_name,
                torch_dtype=target_dtype,
                device_map=device_map
            )
            self.actual_patch_size = self.vision_tower.config.patch_size
            print(f"표준 CLIP 모델 ({self.vision_tower_name}) 로드 완료.")

            if hasattr(self, 'cfg_only') and not isinstance(self.cfg_only, CLIPVisionConfig):
                self.cfg_only = self.vision_tower.config

        # 파라미터 고정
        self.vision_tower.requires_grad_(False)
        self.is_loaded = True


    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if not self.is_loaded:
            device_map = getattr(
                self.args, 'device_map',
                {"": torch.cuda.current_device() if torch.cuda.is_available() else "cpu"}
            )
            self.load_model(device_map=device_map)

        # dtype / device 동기화
        tgt_dev  = next(self.vision_tower.parameters()).device
        tgt_dtype = next(self.vision_tower.parameters()).dtype
        images = images.to(device=tgt_dev, dtype=tgt_dtype)

        # ───────────────────── Reg‑Gated CLIP 분기 ─────────────────────
        if self.is_custom_reg_gated_clip:
            feats = self.vision_tower(images)        # (B,H,T)  또는 (B,T,H)  또는 (B,H)

            # ---- STEP 1. 2‑D → 3‑D 승격,  hidden dim (1024) 탐색 ----
            if feats.ndim == 2:                      # (B,H) → (B,1,H)
                feats = feats.unsqueeze(1)
            elif feats.ndim != 3:
                raise RuntimeError(f"Vision tower output must be 2‑D/3‑D, got {feats.ndim}‑D")

            B, A, C = feats.shape                    # 현재 (B,*,*)
            H = self.config.hidden_size              # 1024

            # ---- STEP 2. hidden 차원을 마지막 축으로 맞춤 ----
            if C == H:                               # 이미 (B,T,H)  ✔
                pass
            elif A == H:                             # (B,H,T) → (B,T,H)
                feats = feats.transpose(1, 2)
            else:
                raise RuntimeError(
                    f"Cannot locate hidden‑dim {H} in {tuple(feats.shape)}"
                )

            # ---- STEP 3. CLS/패치 토큰 선택 ----
            num_cls_patch = (self.config.image_size // self.config.patch_size) ** 2 + 1
            # 비전 타워가 보유한 옵션을 읽어 옵니다. 기본값은 'patch'
            select_feature = self.select_feature     # 생성자에서 이미 설정됨

            if select_feature == 'patch':        # CLS 제외
                feats = feats[:, 1:num_cls_patch]
            elif select_feature == 'cls_patch':  # CLS 포함
                feats = feats[:, :num_cls_patch]
            else:
                print(f"경고: 알 수 없는 select_feature '{select_feature}', cls_patch로 처리")
                feats = feats[:, :num_cls_patch]

        # ─────────────────── 표준 Hugging‑Face CLIP 분기 ──────────────────
        else:
            outs  = self.vision_tower(images, output_hidden_states=True)
            feats = self.feature_select(outs)            # (B, N, H)

        # 항상 (B, N, 1024) & dtype 동일
        return feats.to(images.dtype)


    @property
    def dummy_feature(self):
        return torch.zeros(1, self.num_patches, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        if self.vision_tower is not None and hasattr(self.vision_tower, 'dtype'):
            return self.vision_tower.dtype
        return torch.float16 # 기본값

    @property
    def device(self):
        if self.vision_tower is not None and hasattr(self.vision_tower, 'device'):
            return self.vision_tower.device
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def config(self):
        # load_model이 호출된 후에는 실제 모델의 config (또는 mock config)를 사용
        if self.is_loaded:
            if self.is_custom_reg_gated_clip:
                # MockConfig 인스턴스 반환 (load_model에서 image_size, patch_size 업데이트됨)
                # hidden_size는 고정
                class MockConfig:
                    hidden_size = 1024
                    image_size = self.vision_tower.input_resolution
                    patch_size = self.actual_patch_size
                return MockConfig()
            return self.vision_tower.config # HF CLIPVisionModel의 config
        else:
            # delay_load=True이고 아직 load_model이 호출되지 않았을 때 사용
            return self.cfg_only

    @property
    def hidden_size(self):
        # mm_projector의 입력 차원. RegCLIP(proj=None)은 1024.
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        config_obj = self.config
        if not hasattr(config_obj, 'image_size') or not hasattr(config_obj, 'patch_size') or \
           config_obj.image_size is None or config_obj.patch_size is None:
            # 모델 로드 전이거나 config에 해당 정보가 완전하지 않을 수 있음.
            # 이 경우 기본값이나 에러를 발생시킬 수 있으나, 보통 LLaVA는 모델 로드 후 이 속성을 사용함.
            # ViT-L/14 기본값으로 임시 반환 (실제 값은 load_model 후 config에 의해 결정)
            print("경고: config에 image_size 또는 patch_size 정보가 불완전합니다. 기본값 사용.")
            return 224 // 14 
        return config_obj.image_size // config_obj.patch_size

    @property
    def num_patches(self):
        # 순수 패치 토큰의 수 (CLS 토큰 제외)
        return self.num_patches_per_side ** 2