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
from llava.model.language_model.llava_llama import LlavaConfig


def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False,
                          device_map="auto", device="cuda", use_flash_attn=False,
                          # torch_dtype을 **kwargs가 아닌 명시적 인자로 받도록 변경 (또는 kwargs에서 우선적으로 사용)
                          # 여기서는 **kwargs에 이미 torch_dtype이 들어온다고 가정하고,
                          # 초기 설정 부분을 수정합니다.
                          **kwargs): # **kwargs 안에 사용자가 전달한 torch_dtype이 있음
    
    # 사용자가 명시적으로 전달한 torch_dtype을_model_kwargs_torch_dtype 변수에 저장
    # 또는 함수 시그니처에 torch_dtype_arg=None 추가 후,
    # effective_torch_dtype = torch_dtype_arg if torch_dtype_arg is not None else kwargs.get('torch_dtype')
    user_specified_torch_dtype = kwargs.get('torch_dtype', None) # 사용자가 전달한 값 (예: torch.bfloat16)

    # kwargs는 AutoModelForCausalLM.from_pretrained 등에 전달될 딕셔너리
    # device_map 설정은 그대로 둠
    model_kwargs = {"device_map": device_map, **kwargs} # 사용자의 **kwargs를 먼저 펼치고 device_map 추가

    if device != "cuda": # 이 부분은 원래 로직 유지
        model_kwargs['device_map'] = {"": device}

    if load_8bit:
        model_kwargs['load_in_8bit'] = True
        # 8bit 로드 시 torch_dtype은 bnb_4bit_compute_dtype 등과 관련되므로,
        # 사용자가 명시한 torch_dtype과 충돌하지 않도록 주의.
        # 보통 이 경우 model_kwargs['torch_dtype']을 설정하지 않거나 bnb 설정에 따름.
        if 'torch_dtype' in model_kwargs: del model_kwargs['torch_dtype'] # 명시적 torch_dtype 제거
    elif load_4bit:
        model_kwargs['load_in_4bit'] = True
        model_kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if user_specified_torch_dtype == torch.bfloat16 else torch.float16, # bf16이면 bf16 사용
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
        if 'torch_dtype' in model_kwargs: del model_kwargs['torch_dtype'] # 명시적 torch_dtype 제거
    else:
        # 8bit/4bit가 아닐 때, 사용자가 명시한 torch_dtype 사용
        if user_specified_torch_dtype is not None:
            model_kwargs['torch_dtype'] = user_specified_torch_dtype
        else: # 사용자가 명시하지 않으면 기본 float16 (기존 로직)
            model_kwargs['torch_dtype'] = torch.float16
            user_specified_torch_dtype = torch.float16 # LlavaConfig에 전달하기 위해 업데이트

    if use_flash_attn: # 이 부분은 원래 로직 유지
        model_kwargs['attn_implementation'] = 'flash_attention_2'

    if 'llava' in model_name.lower():
        # Load LLaVA model
        if 'lora' in model_name.lower() and model_base is None:
            warnings.warn('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged.')
        if 'lora' in model_name.lower() and model_base is not None:
            from llava.model.language_model.llava_llama import LlavaConfig
            lora_cfg_pretrained = LlavaConfig.from_pretrained(model_path)
            lora_cfg_pretrained.attention_dropout = 0.0
            lora_cfg_pretrained.rope_theta = 10000
            lora_cfg_pretrained.attention_bias = False
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            print('Loading LLaVA from base model...')
            model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
                model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

            print('Loading additional LLaVA weights...')
            if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
                non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
            else:
                # this is probably from HF Hub
                from huggingface_hub import hf_hub_download
                def load_from_hf(repo_id, filename, subfolder=None):
                    cache_file = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        subfolder=subfolder)
                    return torch.load(cache_file, map_location='cpu')
                non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')
            non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
            if any(k.startswith('model.model.') for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
            model.load_state_dict(non_lora_trainables, strict=False)

            from peft import PeftModel
            print('Loading LoRA weights...')
            model = PeftModel.from_pretrained(model, model_path)
            print('Merging LoRA weights...')
            model = model.merge_and_unload()
            print('Model is loaded...')
        elif model_base is not None:
            # this may be mm projector only
            print('Loading LLaVA from base model...')
            if 'mpt' in model_name.lower():
                if not os.path.isfile(os.path.join(model_path, 'configuration_mpt.py')):
                    shutil.copyfile(os.path.join(model_base, 'configuration_mpt.py'), os.path.join(model_path, 'configuration_mpt.py'))
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
                cfg_pretrained = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                model = LlavaMptForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)


                # 2. model_base에서 기본 Llama 모델의 전체 설정을 로드합니다.
                # 여기에는 attention_dropout과 같은 Llama의 모든 기본 필드가 포함됩니다.
                full_base_cfg = AutoConfig.from_pretrained(model_base)

                # 3. model_path (LLaVA 모델 경로)에서 LLaVA 관련 설정을 로드합니다.
                #    이것은 LlavaConfig 객체여야 합니다 (config.json의 model_type="llava_llama" 가정).
                from llava.model.language_model.llava_llama import LlavaConfig
                llava_specific_cfg = LlavaConfig.from_pretrained(model_path)

                # 4. 기본 Llama 설정에 LLaVA 특화 설정을 덮어씁니다.
                #    이렇게 하면 Llama 기본값 위에 LLaVA 변경/추가 사항이 적용됩니다.
                #    LlavaConfig가 LlamaConfig를 상속하므로, to_dict()로 변환 후 업데이트.
                config_dict_to_update = full_base_cfg.to_dict()
                config_dict_to_update.update(llava_specific_cfg.to_dict())
                
                # load_pretrained_model 함수에 전달된 torch_dtype (예: torch.bfloat16)을
                # LlavaConfig 생성 시 명시적으로 전달합니다.
                # LlavaConfig.__init__에서 이 값을 사용하여 내부 플래그(bf16, fp16) 및
                # self.torch_dtype (실제 torch.dtype 객체)을 설정합니다.
                # kwargs에서 torch_dtype을 가져와 사용합니다. load_pretrained_model의 **kwargs에 이미 포함되어 있음.
                # 여기서 config_dict_to_update['torch_dtype'] = kwargs.get('torch_dtype') 와 같이 설정합니다.
                # 만약 **kwargs에 torch_dtype이 없다면 (함수 시그니처에 명시적으로 있지만),
                # 명시적으로 torch_dtype 변수를 사용합니다.
                # 이 함수 시그니처에는 **kwargs 외에 명시적인 torch_dtype 인자가 없으므로,
                # **kwargs를 통해 전달된 torch_dtype 값을 사용해야 합니다.
                # (주의: load_pretrained_model의 원래 kwargs에는 torch_dtype이 이미 있습니다.
                #  BitsAndBytesConfig 관련 로직에서 kwargs['torch_dtype'] = torch.float16 로 설정되기도 함)
                #  가장 확실한 것은 load_pretrained_model 함수의 인자로 받은 torch_dtype을 사용하는 것입니다.
                #  함수 시그니처에 torch_dtype이 없으므로, kwargs에서 가져옵니다.
                config_dict_to_update['torch_dtype'] = kwargs.get('torch_dtype')

                # 5. 최종적으로 병합된 dict로부터 LlavaConfig 객체를 생성합니다.
                final_config_for_model = LlavaConfig(**config_dict_to_update)

                model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=final_config_for_model, **kwargs)


            mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
            mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
            model.load_state_dict(mm_projector_weights, strict=False)
        else:
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = LlavaMptForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
            elif 'mistral' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = LlavaMistralForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    **kwargs
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = LlavaLlamaForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    **kwargs
                )
    else:
        # Load language model
        if model_base is not None:
            # PEFT model
            from peft import PeftModel
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print('Convert to FP16...')
            model.to(torch.float16)
        else:
            use_fast = False
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    image_processor = None

    if 'llava' in model_name.lower():
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model(device_map=device_map)
        if device_map != 'auto': # True if device_map is "cuda", "cuda:0", etc.
            # model_kwargs['torch_dtype'] holds the torch.dtype object (e.g., torch.bfloat16)
            # or a string that was resolved earlier. It's the most reliable source for
            # the intended dtype of the entire model system.
            correct_dtype_for_conversion = model_kwargs.get('torch_dtype')

            # Ensure it's a torch.dtype object if it was passed as a string initially
            # (though in your case, it's already a torch.dtype object from the script)
            if isinstance(correct_dtype_for_conversion, str):
                correct_dtype_for_conversion = getattr(torch, correct_dtype_for_conversion, None)
            
            # Fallback logic for safety, though it shouldn't be hit in your specific scenario
            if correct_dtype_for_conversion is None:
                warnings.warn(
                    "Vision tower dtype could not be determined from model_kwargs, attempting to use model.dtype or defaulting to float16.",
                    UserWarning
                )
                if hasattr(model, 'dtype') and isinstance(model.dtype, torch.dtype):
                    correct_dtype_for_conversion = model.dtype
                else:
                    correct_dtype_for_conversion = torch.float16 # Last resort default

            # Now, `correct_dtype_for_conversion` should be a valid torch.dtype.
            # `device_map` here is a string like "cuda" or "cuda:0".
            # vision_tower.to() will move the nn.Module and its parameters.
            vision_tower.to(device=device_map, dtype=correct_dtype_for_conversion)
        image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len