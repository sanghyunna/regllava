import warnings
import torch
import torch.nn as nn
import os

from safetensors.torch import load_file as load_safetensors
from INFERclipregXGATED.model import build_model as build_reg_gated_clip
from INFERclipregXGATED.clip import _transform as reg_gated_transform

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False
        self.args = args # LlavaConfig 저장 (이것이 __init__에 전달된다고 가정)

        self.vision_tower_name = vision_tower

        self._num_registers = 0 # GATED 모델이 아니면 0, 또는 load_model에서 설정될 때까지 임시값
        self._actual_patch_size = 14 # 기본값, load_model에서 실제 값으로 업데이트
        self._input_resolution = 224 # 기본값, load_model에서 실제 값으로 업데이트

        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        self._device = None

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()

        # CUSTOM: "GATED" 로컬 모델이고 delay_load=True 인 경우,
        # CLIPVisionConfig.from_pretrained() 호출을 피하고,
        # 나중에 load_model()에서 실제 모델 로드 후 config를 설정합니다.
        elif "GATED" in self.vision_tower_name.upper():
            # 이 경우, cfg_only는 None으로 두거나, 최소한의 기본값으로 설정할 수 있습니다.
            # self.hidden_size 같은 속성은 load_model()이 호출된 후에야 정확해집니다.
            self.cfg_only = None # 또는 기본 CLIPVisionConfig() 인스턴스

        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print(f'{self.vision_tower_name} is already loaded, skipping.')
            return

        # "GATED" 모델 이름 감지 → Reg-Gated CLIP 로드 분기
        if isinstance(self.vision_tower_name, str) and "GATED" in self.vision_tower_name.upper():
            # 1) checkpoint 읽어 build_model 호출

            vision_tower_path = self.vision_tower_name
            # model_dir_for_vision_tower는 LlavaConfig (self.args)에 저장되어 있다고 가정
            if not os.path.isabs(vision_tower_path) and hasattr(self.args, 'model_dir_for_vision_tower') and self.args.model_dir_for_vision_tower is not None:
                vision_tower_path = os.path.join(self.args.model_dir_for_vision_tower, self.vision_tower_name)
            
            if not os.path.exists(vision_tower_path):
                raise FileNotFoundError(f"Custom vision tower GATED file not found at {vision_tower_path}. Original name: {self.vision_tower_name}, Model dir: {getattr(self.args, 'model_dir_for_vision_tower', 'Not Set')}")
            state = load_safetensors(self.vision_tower_name)


            # 2) visual encoder만 뽑아서 dtype/device 설정
            target_dtype = self.dtype
            # build_reg_gated_clip 호출 시 target_dtype 전달
            full_clip = build_reg_gated_clip(state, model_dtype=target_dtype)

            # device_map 처리: "auto"인 경우 실제 장치로 변환
            resolved_device = device_map
            if device_map == "auto":
                # "auto"는 여기서 직접 사용할 수 없으므로, 주 장치를 선택하거나 CPU를 사용
                resolved_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            elif device_map is None: # device_map이 명시적으로 None이면 기본 CUDA 사용
                resolved_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            self._device = resolved_device # 결정된 device를 인스턴스 변수에 저장!

            self.vision_tower = full_clip.visual.to(device=self._device, dtype=target_dtype)
            self.vision_tower.requires_grad_(False)

            # 3) 등록 토큰 수, 해상도, 패치크기 얻기
            self._num_registers = getattr(self.vision_tower, 'num_registers', 4) # GATED 모델의 num_registers 속성 사용
            self._input_resolution = getattr(self.vision_tower, 'input_resolution', 224)
            self._actual_patch_size = getattr(self.vision_tower, 'patch_size', 14)

            # 모델 로드 후, config 객체를 실제 모델의 config로 설정
            # VisionTransformer에 self.width가 직접 없으므로, conv1.out_channels 등에서 가져옴
            # 또는 class_embedding.shape[-1], positional_embedding.shape[-1] 등도 가능
            vision_tower_width = self.vision_tower.conv1.out_channels
            self.vision_tower.config = self._build_dummy_config_for_gated(
                vision_tower_width,
                self._input_resolution,
                self._actual_patch_size
                )

            # 4) 전처리기 설정
            self.image_processor = reg_gated_transform(self._input_resolution)

            # CUSTOM: Compose 객체에 image_mean, image_std 속성 추가
            # 이 값들은 reg_gated_transform 내부의 Normalize에 사용된 값이어야 함.
            # 일반적으로 CLIP 표준 값을 사용하거나, 모델 학습 시 사용된 특정 값을 사용.
            # 예시로 CLIP 표준 값을 사용. 실제 값으로 변경 필요.
            setattr(self.image_processor, 'image_mean', getattr(self.args, 'custom_image_mean', [0.48145466, 0.4578275, 0.40821073]))
            setattr(self.image_processor, 'image_std', getattr(self.args, 'custom_image_std', [0.26862954, 0.26130258, 0.27577711]))
            setattr(self.image_processor, 'crop_size', {'height': self._input_resolution, 'width': self._input_resolution})
            setattr(self.image_processor, 'size', {'shortest_edge': self._input_resolution})

            self.is_loaded = True
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        # 표준 모델 로드 시에도 self.dtype을 존중하도록 torch_dtype 인자 전달
        self.vision_tower = CLIPVisionModel.from_pretrained(
            self.vision_tower_name,
            device_map=device_map,
            torch_dtype=target_dtype)
        self.vision_tower.requires_grad_(False)

        # HF 모델의 경우, 첫 번째 파라미터의 device를 self._device로 설정
        if len(list(self.vision_tower.parameters())) > 0:
            self._device = next(self.vision_tower.parameters()).device
        elif device_map == "auto": # 파라미터가 없지만 auto인 경우 (매우 드문 케이스)
                self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device_map, str): # "cuda:0" 등 명시적 장치 문자열
                self._device = torch.device(device_map)
        else: # 기타 (None 등)
                self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.is_loaded = True

    def _build_dummy_config_for_gated(self, hidden_size, image_size, patch_size):
        """GATED 모델 로드 후, 호환성을 위해 최소한의 config 객체를 생성합니다."""
        # CLIPVisionConfig의 필수 필드들을 채워줍니다.
        # 실제 CLIPVisionConfig 인스턴스를 생성하고 값을 할당하는 것이 좋습니다.
        config = CLIPVisionConfig(
            hidden_size=hidden_size,
            image_size=image_size,
            patch_size=patch_size,
            # projection_dim 등 다른 필요한 기본값들을 추가할 수 있습니다.
        )
        return config

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
    def forward(self, images: torch.Tensor, output_attentions_for_visualization: bool = False):
        """
        Forward pass for CLIPVisionTower.
        Args:
            images: Input image tensor(s). Expected to be a batch tensor for attention viz.
            output_attentions_for_visualization: If True, return attention weights.
        Returns:
            Image features, or (Image features, Vision tower attentions) if requested.
        """
        if not self.is_loaded:
            self.load_model()
            if not self.is_loaded:
                raise RuntimeError(f"Vision tower {self.vision_tower_name} not loaded.")

        target_device = self.device
        target_dtype = self.dtype

        # Assuming 'images' is a pre-processed batch tensor as is typical from the main script.
        # Handling for list of PIL images is omitted for brevity here, as the main script
        # usually provides a batch tensor.
        if not isinstance(images, torch.Tensor):
            # Fallback or error if `images` is not a tensor.
            # This path should ideally not be taken if `llava_run_inference.py` prepares `img_tensor` correctly.
            raise ValueError("`images` argument in CLIPVisionTower.forward must be a tensor.")

        images_batch = images.to(device=target_device, dtype=target_dtype)
        
        # Call the internal vision model (self.vision_tower)
        # self.vision_tower is an instance of VisionTransformer (custom) or CLIPVisionModel (HF)
        # Both should accept output_attentions and return an object with an .attentions attribute.
        vision_model_output = self.vision_tower(
            images_batch,
            output_hidden_states=True, # Required for self.feature_select
            output_attentions=output_attentions_for_visualization
        )
        
        # Extract features using the existing feature_select method
        final_image_features = self.feature_select(vision_model_output).to(target_dtype)
        
        final_attentions = None
        if output_attentions_for_visualization:
            # vision_model_output is expected to have an 'attentions' attribute
            # which is a tuple of tensors (one for each layer of the vision model)
            if hasattr(vision_model_output, 'attentions'):
                final_attentions = vision_model_output.attentions
            else:
                # This case should not happen if VisionTransformer.forward is correctly modified
                print(f"Warning: Vision model output for {self.vision_tower_name} "
                      "did not have 'attentions' attribute when requested.")

        if output_attentions_for_visualization:
            return final_image_features, final_attentions
        else:
            return final_image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    # @property
    # def dtype(self):
    #     if self.vision_tower is not None and hasattr(self.vision_tower, 'dtype'): # 실제 모델의 dtype 우선
    #         return self.vision_tower.dtype
    #     # self.args (LlavaConfig) 기반 dtype 결정 로직 (이전과 동일)
    #     if hasattr(self.args, 'torch_dtype') and self.args.torch_dtype is not None:
    #         return self.args.torch_dtype
    #     elif getattr(self.args, 'fp16', False):
    #         return torch.float16
    #     elif getattr(self.args, 'bf16', False):
    #         return torch.bfloat16
    #     return torch.float32


    @property
    def dtype(self) -> torch.dtype:
        # 1. 모델이 이미 로드되었고, 실제 vision_tower 모듈에 dtype 속성이 있다면 그것을 사용
        if self.is_loaded and self.vision_tower is not None and hasattr(self.vision_tower, 'dtype'):
            return self.vision_tower.dtype
        
        # 2. self.args (LlavaConfig 인스턴스)가 설정되어 있는지 확인
        if not hasattr(self, 'args') or self.args is None:
            warnings.warn(
                "CLIPVisionTower.args (LlavaConfig) is not set. "
                "Defaulting dtype to torch.bfloat16 based on user preference. "
                "This might be unintended if args were expected to provide dtype info."
            )
            return torch.bfloat16 # 사용자가 bf16을 원하므로, 비상시 기본값으로 bf16

        # 3. LlavaConfig에 저장된 torch_dtype 객체를 직접 사용 (가장 우선)
        #    LlavaConfig.__init__에서 self.torch_dtype에 실제 torch.dtype 객체를 저장했다고 가정
        if hasattr(self.args, 'torch_dtype') and isinstance(self.args.torch_dtype, torch.dtype):
            return self.args.torch_dtype
        
        # 4. LlavaConfig에 torch_dtype이 문자열로 저장된 경우 처리 (예: "bfloat16")
        #    (LlavaConfig.__init__에서 self.torch_dtype에 문자열을 저장했을 경우)
        if hasattr(self.args, 'torch_dtype') and isinstance(self.args.torch_dtype, str):
            dtype_str = self.args.torch_dtype.lower()
            if dtype_str == "bfloat16":
                return torch.bfloat16
            elif dtype_str == "float16":
                return torch.float16
            elif dtype_str == "float32":
                return torch.float32
            else:
                warnings.warn(f"Unsupported torch_dtype string '{self.args.torch_dtype}' in LlavaConfig. Defaulting to bfloat16.")
                return torch.bfloat16 # 알 수 없는 문자열이면 bf16으로

        # 5. LlavaConfig의 bf16 또는 fp16 플래그를 확인 (torch_dtype 속성이 없는 경우의 fallback)
        #    bf16을 우선적으로 확인
        if getattr(self.args, 'bf16', False):
            return torch.bfloat16
        elif getattr(self.args, 'fp16', False):
            return torch.float16
        
        # 6. 모든 조건에 해당하지 않으면, 사용자가 bf16을 원한다고 했으므로 bfloat16 반환
        #    또는 LlamaConfig의 기본 dtype을 따르도록 torch.float32를 반환할 수도 있음
        #    여기서는 사용자 요청에 따라 bf16
        return torch.bfloat16

    @property
    def num_registers(self):
        # load_model에서 설정된 _num_registers 값을 반환
        # 또는 GATED 모델의 경우 여기서 하드코딩된 값을 반환할 수도 있음
        if "GATED" in self.vision_tower_name.upper():
            return self._num_registers # load_model에서 설정된 값을 따름 (또는 여기서 4로 고정해도 됨)
        # 표준 HF CLIP 모델은 num_registers가 없으므로 0 반환
        return 0

    @property
    def device(self):
        if self._device is not None:
            return self._device
        # fallback (load_model이 호출되지 않은 극단적인 경우)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        """
        Calculates the number of patches on one side of the square image.
        This value depends on whether the model is loaded, and if it's a GATED model
        or a standard Hugging Face CLIP model.
        """
        if self.is_loaded: # 모델이 로드된 후
            if "GATED" in self.vision_tower_name.upper():
                # GATED 모델: load_model에서 설정된 내부 변수 사용
                # self._input_resolution과 self._actual_patch_size는 load_model에서 설정됨
                if self._actual_patch_size == 0: # 패치 크기가 0이면 나눗셈 오류 방지
                    return 0 
                return self._input_resolution // self._actual_patch_size
            else: # 표준 HF CLIP 모델: 로드된 모델의 config 사용
                if self.vision_tower is not None and hasattr(self.vision_tower, 'config'):
                    config = self.vision_tower.config
                    if config.patch_size == 0: return 0
                    return config.image_size // config.patch_size
                else: # 로드는 되었으나 config 접근 불가 (예외적 상황)
                    warnings.warn("Vision tower loaded but config not accessible, falling back to LlavaConfig for patch info.")
                    # LlavaConfig의 기본값이나 사용자 설정값으로 fallback
                    image_size = getattr(self.args, 'vision_image_size', 224) # LlavaConfig에 vision_image_size 추가 필요
                    patch_size = getattr(self.args, 'vision_patch_size', 14) # LlavaConfig에 vision_patch_size 추가 필요
                    if patch_size == 0: return 0
                    return image_size // patch_size
        else: # 모델이 로드되기 전 (delay_load=True 인 경우)
            if "GATED" in self.vision_tower_name.upper():
                # GATED 모델 로드 전: LlavaConfig의 기본값 또는 사용자 설정값 사용
                # (이 값들은 GATED 모델의 실제 값과 일치해야 함)
                image_size = getattr(self.args, 'vision_image_size', 224) # LlavaConfig에 vision_image_size 필드 추가 고려
                patch_size = getattr(self.args, 'vision_patch_size', 14) # LlavaConfig에 vision_patch_size 필드 추가 고려
                if patch_size == 0: return 0
                return image_size // patch_size
            elif self.cfg_only is not None: # 표준 HF CLIP 모델 로드 전 (cfg_only 사용)
                if self.cfg_only.patch_size == 0: return 0
                return self.cfg_only.image_size // self.cfg_only.patch_size
            else: # cfg_only도 없는 경우 (예: GATED 모델이 아닌데 경로가 잘못된 경우 등)
                warnings.warn("Cannot determine num_patches_per_side before model loading without cfg_only or GATED model hints.")
                return 0 # 또는 적절한 기본값 (예: 224 // 14 = 16)

    @property
    def num_patches(self):
        """
        Calculates the total number of patches.
        """
        n_patches_side = self.num_patches_per_side
        return n_patches_side * n_patches_side



class CLIPVisionTowerS2(CLIPVisionTower):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__(vision_tower, args, delay_load)

        self.s2_scales = getattr(args, 's2_scales', '336,672,1008')
        self.s2_scales = list(map(int, self.s2_scales.split(',')))
        self.s2_scales.sort()
        self.s2_split_size = self.s2_scales[0]
        self.s2_image_size = self.s2_scales[-1]

        try:
            from s2wrapper import forward as multiscale_forward
        except ImportError:
            raise ImportError('Package s2wrapper not found! Please install by running: \npip install git+https://github.com/bfshi/scaling_on_scales.git')
        self.multiscale_forward = multiscale_forward

        # change resize/crop size in preprocessing to the largest image size in s2_scale
        if not delay_load or getattr(args, 'unfreeze_mm_vision_tower', False):
            self.image_processor.size['shortest_edge'] = self.s2_image_size
            self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        self.image_processor.size['shortest_edge'] = self.s2_image_size
        self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

        self.is_loaded = True

    @torch.no_grad()
    def forward_feature(self, images):
        image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
        image_features = self.feature_select(image_forward_outs).to(images.dtype)
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_feature = self.multiscale_forward(self.forward_feature, image.unsqueeze(0), img_sizes=self.s2_scales, max_split_size=self.s2_split_size)
                image_features.append(image_feature)
        else:
            image_features = self.multiscale_forward(self.forward_feature, images, img_sizes=self.s2_scales, max_split_size=self.s2_split_size)

        return image_features

    @property
    def hidden_size(self):
        return self.config.hidden_size * len(self.s2_scales)