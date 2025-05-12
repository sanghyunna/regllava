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
        self.num_registers = 0 # <-- 추가: 초기화

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
                image_size = 224 # 임시 값
                patch_size = 14  # 임시 값
                num_registers = 4 # <-- 추가: 기본값 설정
            self.cfg_only = MockRegGatedCLIPConfig()
            self.num_registers = self.cfg_only.num_registers # <-- 추가: 초기값 설정

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
            # build_model이 반환한 CLIP 모델에서 visual 속성 접근
            loaded_vision_tower = full_clip_model.visual.to(dtype=target_dtype)

            # --- 실제 num_registers 값 저장 ---
            self.num_registers = getattr(loaded_vision_tower, 'num_registers', 4) # 로드된 모델에서 값 가져오기
            print(f"  > Custom Reg-Gated CLIP loaded with num_registers: {self.num_registers}")

            # LayerNorm float32 유지 (선택적)
            # for m in loaded_vision_tower.modules():
            #     if isinstance(m, nn.LayerNorm): m.float()

            self.vision_tower = loaded_vision_tower
            input_resolution = getattr(self.vision_tower, 'input_resolution', 224) # 로드된 모델에서 값 가져오기
            self.actual_patch_size = getattr(self.vision_tower, 'patch_size', 14) # 로드된 모델에서 값 가져오기

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
                print("  > Updating MockConfig with actual values...")
                self.cfg_only.image_size   = input_resolution
                self.cfg_only.patch_size   = self.actual_patch_size
                self.cfg_only.hidden_size  = 1024 # ViT-L 기준 고정값
                self.cfg_only.num_registers = self.num_registers # <-- 추가: 실제 값으로 업데이트
                print(f"  > MockConfig updated: image_size={self.cfg_only.image_size}, patch_size={self.cfg_only.patch_size}, num_registers={self.cfg_only.num_registers}")

            print(f"사용자 정의 Reg‑Gated CLIP 모델 로드 완료.")

        # ----- (2) 표준 HuggingFace CLIP 로드 -----
        else:
            self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
            self.vision_tower    = CLIPVisionModel.from_pretrained(
                self.vision_tower_name,
                torch_dtype=target_dtype,
                device_map=device_map
            )
            self.actual_patch_size = self.vision_tower.config.patch_size
            self.num_registers = 0 # 표준 CLIP은 레지스터 없음
            
            # 파라미터 고정 및 로드 완료 플래그 설정 (기존 코드 유지)
            self.vision_tower.requires_grad_(getattr(self.args, 'unfreeze_mm_vision_tower', False)) # unfreeze 옵션 반영
            self.is_loaded = True

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
        if self.vision_tower is None:
            raise RuntimeError("Vision tower is not loaded. Call load_model() first.")
        try:
            vision_params = list(self.vision_tower.parameters())
            if not vision_params:
                # 비전 타워에 파라미터가 없는 경우 (예: nn.Identity를 래핑한 경우 등)
                # 또는 로드가 완전히 실패한 경우.
                # args에서 device/dtype을 가져오려는 시도는 할 수 있으나, 모델 자체의 특성을 반영 못함.
                # 이 경우는 로드 로직에서 더 강력한 검증이 필요할 수 있음.
                print("Warning: Vision tower has no parameters. Device/dtype determination might be unreliable.")
                # 임시로 args를 사용하거나, 에러 발생시키는 것이 더 안전할 수 있음.
                # 여기서는 이전 로직을 유지하되, 경고를 명확히 함.
                if hasattr(self.args, 'device'): tgt_dev = self.args.device
                else: tgt_dev = 'cuda' if torch.cuda.is_available() else 'cpu'

                if getattr(self.args, 'bf16', False): tgt_dtype = torch.bfloat16
                elif getattr(self.args, 'fp16', False): tgt_dtype = torch.float16
                else: tgt_dtype = torch.float32
            else:
                tgt_dev = vision_params[0].device
                tgt_dtype = vision_params[0].dtype
        except StopIteration: # next(self.vision_tower.parameters()) 실패 시
            print("Warning: Could not determine vision tower device/dtype from parameters. Using model's main device/dtype.")
            if hasattr(self.args, 'device'): tgt_dev = self.args.device
            else: tgt_dev = 'cuda' if torch.cuda.is_available() else 'cpu'
            if getattr(self.args, 'bf16', False): tgt_dtype = torch.bfloat16
            elif getattr(self.args, 'fp16', False): tgt_dtype = torch.float16
            else: tgt_dtype = torch.float32

        images = images.to(device=tgt_dev, dtype=tgt_dtype)
        image_features_selected_layer = None # 초기화

        # ───────────────────── Reg‑Gated CLIP 분기 ─────────────────────
        if self.is_custom_reg_gated_clip:
            # --- 커스텀 VisionTransformer 호출 시 output_hidden_states=True 전달 ---
            print(f"Debug (Custom CLIP forward): Calling vision_tower with output_hidden_states=True. select_layer: {self.select_layer}")
            outputs = self.vision_tower(images, output_hidden_states=True)

            hidden_states_tuple = None
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None and len(outputs.hidden_states) > 0:
                hidden_states_tuple = outputs.hidden_states
                print(f"  > Received hidden_states_tuple with {len(hidden_states_tuple)} layers.")
            else:
                print(f"  > Warning: hidden_states_tuple not found or empty in vision tower output. Defaulting to last_hidden_state.")

            # --- select_layer 적용 ---
            # hidden_states_tuple이 있고, self.select_layer가 유효한 경우에만 사용
            # (HF 모델은 select_layer=0이 임베딩, 1부터 트랜스포머 레이어)
            # 커스텀 모델은 hidden_states[0]이 ln_pre 후, hidden_states[1]부터가 트랜스포머 레이어 1의 출력.
            # LLaVA의 select_layer: 음수는 끝에서부터 (예: -1은 마지막, -2는 마지막에서 두번째)
            # 양수는 0부터 시작.
            if hidden_states_tuple is not None:
                num_hidden_layers = len(hidden_states_tuple)
                target_layer_index = self.select_layer
                # 음수 인덱스를 양수 인덱스로 변환
                if target_layer_index < 0:
                    target_layer_index = num_hidden_layers + target_layer_index # 예: -1 -> len-1, -2 -> len-2

                if 0 <= target_layer_index < num_hidden_layers:
                    image_features_full = hidden_states_tuple[target_layer_index]
                    print(f"  > Selected layer {self.select_layer} (index {target_layer_index}) from hidden_states. Shape: {image_features_full.shape}")
                else:
                    print(f"  > Warning: Invalid select_layer index {target_layer_index} (original: {self.select_layer}) for {num_hidden_layers} hidden layers. Defaulting to last_hidden_state.")
                    image_features_full = outputs.last_hidden_state # 폴백
            else: # hidden_states_tuple이 없는 경우 (커스텀 모델이 지원 안 하거나, output_hidden_states=False로 호출된 경우)
                image_features_full = outputs.last_hidden_state # 폴백
                print(f"  > Using last_hidden_state as image_features_full. Shape: {image_features_full.shape}")

            # --- image_features_full 유효성 검사 (폴백 후에도 None일 수 있으므로) ---
            if image_features_full is None:
                if hasattr(outputs, 'last_hidden_state') and outputs.last_hidden_state is not None:
                    image_features_full = outputs.last_hidden_state
                    print("  > Warning: image_features_full was None, re-defaulted to last_hidden_state.")
                else:
                    print("Error: image_features_full is None even after trying last_hidden_state. Vision tower output is problematic.")
                    return self.dummy_feature.to(images.dtype) # dummy_feature 반환

            # --- 특징 선택 로직 (mm_vision_select_feature 적용) ---
            B, S, H = image_features_full.shape
            cfg = self.config
            if cfg.patch_size is None or cfg.patch_size == 0:
                raise ValueError("Patch size is not configured correctly in CLIPVisionTower.config.")
            num_patches = (cfg.image_size // cfg.patch_size) ** 2

            # 시퀀스 길이 계산 시점: image_features_full은 (CLS + REG + Patch) 전체 시퀀스
            expected_seq_len = 1 + self.num_registers + num_patches
            current_seq_len = image_features_full.shape[1]

            if current_seq_len != expected_seq_len:
                print(f"Error (Custom CLIP forward): Unexpected sequence length from selected layer! Expected {expected_seq_len} (1 CLS + {self.num_registers} REG + {num_patches} Patch for image_size {cfg.image_size}), but got {current_seq_len}. Check VisionTransformer's hidden_states output and config.")
                return self.dummy_feature.to(images.dtype)

            print(f"  > (After layer selection) Selecting feature '{self.select_feature}'. Num registers: {self.num_registers}, Num patches: {num_patches}")

            if self.select_feature == 'patch':
                start_index = 1 + self.num_registers
                end_index = start_index + num_patches
                image_features_selected_layer = image_features_full[:, start_index:end_index]
                print(f"    > Sliced for 'patch': start={start_index}, end={end_index}. Final shape: {image_features_selected_layer.shape}")
            elif self.select_feature == 'cls_patch':
                end_index = 1 + self.num_registers + num_patches
                image_features_selected_layer = image_features_full[:, 0:end_index]
                print(f"    > Sliced for 'cls_patch': start=0, end={end_index}. Final shape: {image_features_selected_layer.shape}")
            else:
                print(f"    > Warning: Unknown select_feature '{self.select_feature}'. Defaulting to 'patch'.")
                start_index = 1 + self.num_registers
                end_index = start_index + num_patches
                image_features_selected_layer = image_features_full[:, start_index:end_index]
                print(f"    > Sliced for 'patch' (default): start={start_index}, end={end_index}. Final shape: {image_features_selected_layer.shape}")

        # ─────────────────── 표준 Hugging‑Face CLIP 분기 ──────────────────
        else: # is_custom_reg_gated_clip == False
            # 표준 CLIP 경로는 이미 output_hidden_states=True로 호출하고, self.feature_select 내부에서 self.select_layer 사용
            print(f"Debug (Standard CLIP forward): Calling vision_tower with output_hidden_states=True.")
            outs = self.vision_tower(images, output_hidden_states=True)
            # self.feature_select는 hidden_states[self.select_layer] 를 사용하고, 그 다음 'patch'/'cls_patch' 슬라이싱
            image_features_selected_layer = self.feature_select(outs)
            print(f"  > Selected features using select_layer={self.select_layer}, select_feature='{self.select_feature}'. Final shape: {image_features_selected_layer.shape}")


        # --- 최종 반환 전 null 체크 및 타입 변환 ---
        if image_features_selected_layer is None:
            print("Error: image_features_selected_layer is None before returning from CLIPVisionTower.forward.")
            return self.dummy_feature.to(images.dtype)

        return image_features_selected_layer.to(images.dtype)

    # --- config 속성 수정 ---
    @property
    def config(self):
        if self.is_loaded:
            if self.is_custom_reg_gated_clip:
                # MockConfig 업데이트 (load_model에서 설정된 실제 값 사용)
                class MockConfig:
                    # --- 값 가져오기 전에 self.vision_tower 존재 확인 ---
                    if self.vision_tower is None:
                        # 로드는 되었으나 vision_tower가 None인 비정상 상태
                        # _load_config_only에서 설정된 임시값 사용
                        hidden_size = self.cfg_only.hidden_size if hasattr(self, 'cfg_only') else 1024
                        image_size = self.cfg_only.image_size if hasattr(self, 'cfg_only') else 224
                        patch_size = self.cfg_only.patch_size if hasattr(self, 'cfg_only') else 14
                    else:
                        # 로드된 실제 모델 값 사용
                        hidden_size = getattr(self.vision_tower, 'width', 1024) # width 속성 가정
                        image_size = getattr(self.vision_tower, 'input_resolution', 224)
                        patch_size = getattr(self.vision_tower, 'patch_size', 14)

                    num_registers = self.num_registers # load_model에서 설정된 값
                return MockConfig()
            else:
                # 표준 CLIP: HF config 반환
                if self.vision_tower is None: # 로드 실패/지연 시
                    return self.cfg_only
                return self.vision_tower.config
        else:
            # delay_load=True이고 아직 load_model이 호출되지 않았을 때 사용
            # _load_config_only에서 설정된 cfg_only 반환
            if not hasattr(self, 'cfg_only'): # 예외 처리
                self._load_config_only() # 시도
            return self.cfg_only

    # --- num_patches 속성 수정 ---
    @property
    def num_patches(self):
        config_obj = self.config
        # --- config 객체 및 속성 유효성 검사 강화 ---
        img_size = getattr(config_obj, 'image_size', None)
        patch_size = getattr(config_obj, 'patch_size', None)

        if img_size is not None and patch_size is not None and isinstance(img_size, int) and isinstance(patch_size, int) and patch_size > 0:
            num_patches_val = (img_size // patch_size) ** 2
            # print(f"Debug: Calculated num_patches = {num_patches_val} (image_size={img_size}, patch_size={patch_size})")
            return num_patches_val
        else:
            print(f"Warning: num_patches calculation failed. image_size={img_size}, patch_size={patch_size}. Returning default.")
            # L/14 기본값
            return (224 // 14) ** 2

    # --- dummy_feature 속성 추가 (에러 처리용) ---
    @property
    def dummy_feature(self):
        # num_patches와 hidden_size 계산 시도
        try:
            num_patches = self.num_patches
            hidden_size = self.hidden_size
        except Exception as e:
            print(f"Warning: Could not get num_patches/hidden_size for dummy_feature: {e}. Using defaults.")
            num_patches = 256 # L/14 default
            hidden_size = 1024 # L/14 default

        # device와 dtype 결정 시도
        try:
            dev = self.device
            dt = self.dtype
        except Exception as e:
            print(f"Warning: Could not get device/dtype for dummy_feature: {e}. Using defaults.")
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            dt = torch.float16 # Default dtype

        # print(f"Debug: Creating dummy_feature with shape (1, {num_patches}, {hidden_size}) on {dev} with {dt}")
        return torch.zeros(1, num_patches, hidden_size, device=dev, dtype=dt)

    # --- hidden_size 속성 추가 (일관성) ---
    @property
    def hidden_size(self):
        return getattr(self.config, 'hidden_size', 1024) # 기본값 ViT-L 기준

    # --- device, dtype 속성에서 None 체크 추가 ---
    @property
    def dtype(self):
        if self.vision_tower is not None and hasattr(self.vision_tower, 'dtype') and self.vision_tower.dtype is not None:
            return self.vision_tower.dtype
        # Fallback logic (기존과 유사)
        if getattr(self.args, 'bf16', False): return torch.bfloat16
        if getattr(self.args, 'fp16', False): return torch.float16
        return torch.float32 # 최종 기본값

    @property
    def device(self):
        if self.vision_tower is not None:
            try:
                # 파라미터 리스트에서 디바이스 가져오기
                return next(self.vision_tower.parameters()).device
            except StopIteration:
                # 파라미터 없는 모듈 (예: Identity)
                pass
            # .device 속성이 직접 있는지 확인
            if hasattr(self.vision_tower, 'device') and self.vision_tower.device is not None:
                return self.vision_tower.device
        # Fallback logic
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