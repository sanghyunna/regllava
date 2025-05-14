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


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM


class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"

    def __init__(
        self,
        # LLaMA 기본 인자들 (LlamaConfig에서 상속)
        # attention_dropout은 LlamaConfig의 기본값(0.0)을 따르거나, 여기서 명시적으로 설정 가능
        # super().__init__에 전달되므로, LlamaConfig가 처리함.

        # LLaVA 특화 인자들 (mm_projector, vision_tower 등)
        mm_vision_tower=None,
        mm_hidden_size=None, # 예시: projector 입력 차원 (vision tower 출력)
        mm_vision_select_layer=-2, # CLIPVisionTower에서 사용
        mm_vision_select_feature='patch', # CLIPVisionTower에서 사용
        mm_projector_type='linear', # projector 타입
        # ... 기타 LLaVA 관련 설정 ...

        # 커스텀 비전 인코더 및 경로 관리를 위한 인자들 (이전 논의에서 추가)
        model_dir_for_llava_parts=None, # LLaVA 모델 (프로젝터 등) 파일이 있는 디렉토리
        vision_encoder_base_dir=None,   # 커스텀 비전 인코더 safetensors 파일의 기본 디렉토리
        custom_image_mean=None,         # 커스텀 전처리기 평균값
        custom_image_std=None,          # 커스텀 전처리기 표준편차
        vision_image_size=224,          # 비전 타워 기본 이미지 크기 (CLIPVisionTower의 num_patches_per_side 계산용)
        vision_patch_size=14,           # 비전 타워 기본 패치 크기 (CLIPVisionTower의 num_patches_per_side 계산용)


        # 데이터 타입 관련 인자 (중요)
        torch_dtype=None, # "float16", "bfloat16", "float32" 또는 torch.dtype 객체
        
        **kwargs,
    ):
        # LLaVA 특화 인자들을 self에 먼저 할당
        self.mm_vision_tower = mm_vision_tower
        self.mm_hidden_size = mm_hidden_size
        self.mm_vision_select_layer = mm_vision_select_layer
        self.mm_vision_select_feature = mm_vision_select_feature
        self.mm_projector_type = mm_projector_type

        self.model_dir_for_llava_parts = model_dir_for_llava_parts
        self.vision_encoder_base_dir = vision_encoder_base_dir
        
        # custom_image_mean/std는 기본값을 여기서 설정하거나, 사용하는 곳에서 getattr으로 처리
        self.custom_image_mean = custom_image_mean if custom_image_mean is not None else [0.48145466, 0.4578275, 0.40821073]
        self.custom_image_std = custom_image_std if custom_image_std is not None else [0.26862954, 0.26130258, 0.27577711]
        self.vision_image_size = vision_image_size
        self.vision_patch_size = vision_patch_size

        # torch_dtype 처리 및 fp16/bf16 플래그 설정
        # LlamaConfig는 torch_dtype 인자를 직접 받으므로, kwargs에 추가/수정하여 전달
        # 또한, self.fp16, self.bf16 플래그도 설정하여 CLIPVisionTower 등에서 사용 가능하도록 함
        self.fp16 = False
        self.bf16 = False
        
        _actual_torch_dtype = None # LlamaConfig에 전달될 실제 torch.dtype 객체

        if torch_dtype is not None:
            if isinstance(torch_dtype, str):
                if torch_dtype.lower() == "float16":
                    self.fp16 = True
                    _actual_torch_dtype = torch.float16
                elif torch_dtype.lower() == "bfloat16":
                    self.bf16 = True
                    _actual_torch_dtype = torch.bfloat16
                elif torch_dtype.lower() == "float32":
                    _actual_torch_dtype = torch.float32
                # 다른 문자열 dtype 처리 (필요하다면)
            elif isinstance(torch_dtype, torch.dtype):
                if torch_dtype == torch.float16: self.fp16 = True
                elif torch_dtype == torch.bfloat16: self.bf16 = True
                _actual_torch_dtype = torch_dtype
        
        if _actual_torch_dtype is not None:
            kwargs['torch_dtype'] = _actual_torch_dtype # LlamaConfig가 받을 수 있도록 kwargs에 설정
        
        # LlamaConfig의 __init__ 호출
        # attention_dropout은 LlamaConfig에서 기본값이 0.0이므로, 명시적으로 전달하지 않아도 됨
        # 만약 다른 값을 원한다면 kwargs에 포함시키거나 직접 전달
        # super().__init__(attention_dropout=attention_dropout, **kwargs) # 원래 코드
        super().__init__(**kwargs) # 수정: LlamaConfig가 attention_dropout을 kwargs로 받도록 함 (또는 직접 명시)

        # super().__init__ 호출 후, torch_dtype이 설정되었는지 다시 확인하고,
        # self.torch_dtype (실제 torch.dtype 객체)을 명시적으로 저장할 수 있음.
        # LlamaConfig는 내부적으로 _torch_dtype을 설정하지만, 직접 접근 가능한 self.torch_dtype을 두는 것이 편할 수 있음.
        if hasattr(self, '_torch_dtype') and self._torch_dtype is not None: # LlamaConfig가 설정한 _torch_dtype
             self.torch_dtype = self._torch_dtype
        elif _actual_torch_dtype is not None:
             self.torch_dtype = _actual_torch_dtype
        else: # 기본값 (LlamaConfig의 기본값 또는 시스템 기본값)
             self.torch_dtype = torch.float32 # 또는 torch.get_default_dtype()

        # 만약 LlamaConfig가 fp16, bf16 플래그를 torch_dtype에 따라 자동으로 설정하지 않는다면, 여기서 명시적으로 설정
        # (LlamaConfig는 보통 torch_dtype에 따라 내부적으로 처리함)
        # self.fp16 = (self.torch_dtype == torch.float16)
        # self.bf16 = (self.torch_dtype == torch.bfloat16)



class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlavaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config: LlavaConfig): # Changed type hint to LlavaConfig
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)

        # self.pretraining_tp and self.vocab_size are likely handled by LlamaForCausalLM's __init__
        # If LlamaConfig (and thus LlavaConfig) has pretraining_tp, it should be fine.
        # Verify if LlamaForCausalLM's __init__ correctly sets these based on the config.
        # If not, uncomment and ensure they are correctly sourced from config.
        # self.pretraining_tp = config.pretraining_tp 
        # self.vocab_size = config.vocab_size

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    # NOTE: HF generate() (4.30↑) → forward(cache_position=…)
    #       받아서 무시하거나 상위로 전달해야 에러가 나지 않는다.
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        # Pop position_ids and attention_mask from kwargs.
        # If they are not provided (i.e., None), the base model's generate/forward will handle them.
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)

        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported when calling `generate` directly with `inputs_embeds`.")

        if images is not None:
            # When images are present, inputs_embeds are prepared by prepare_inputs_labels_for_multimodal.
            # This function also returns potentially modified position_ids and attention_mask.
            (
                prepared_input_ids, # This should be None if inputs_embeds is returned
                prepared_position_ids,
                prepared_attention_mask,
                _, # past_key_values
                inputs_embeds, # This is the primary input to the LLM
                _  # labels
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs, # Original input_ids (text tokens)
                position_ids, # Original position_ids from kwargs
                attention_mask, # Original attention_mask from kwargs
                None, # past_key_values for prefill stage
                None, # labels for prefill stage
                images,
                image_sizes=image_sizes
            )
            # Ensure the prepared position_ids and attention_mask are on the same device as inputs_embeds.
            # inputs_embeds should be on the correct model device after prepare_inputs_labels_for_multimodal.
            final_position_ids = prepared_position_ids.to(inputs_embeds.device) if prepared_position_ids is not None else None
            final_attention_mask = prepared_attention_mask.to(inputs_embeds.device) if prepared_attention_mask is not None else None

        else:
            # No images, standard text generation.
            # inputs are token IDs. We need to embed them.
            if inputs is None:
                raise ValueError("`inputs` (input_ids) must be provided if `images` is None.")

            # Determine the target device from the model's embedding layer.
            target_device = self.get_model().embed_tokens.weight.device
            inputs = inputs.to(target_device) # Ensure input_ids are on the correct device.
            inputs_embeds = self.get_model().embed_tokens(inputs)

            # Use position_ids and attention_mask popped from kwargs, ensuring they are on the correct device.
            final_position_ids = position_ids.to(inputs_embeds.device) if position_ids is not None else None
            final_attention_mask = attention_mask.to(inputs_embeds.device) if attention_mask is not None else None

        # Pass inputs_embeds to the base model's generate method.
        # input_ids should ideally be None if inputs_embeds is used,
        # but Hugging Face generate handles this.
        return super().generate(
            # input_ids=None, # Explicitly set to None if not needed by superclass when inputs_embeds is present
            position_ids=final_position_ids,
            attention_mask=final_attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
