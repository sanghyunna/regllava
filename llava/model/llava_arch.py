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


from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from llava.mm_utils import get_anyres_image_grid_shape


class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)

            if 'unpad' in getattr(config, 'mm_patch_merge_type', ''):
                self.image_newline = nn.Parameter(
                    torch.empty(config.hidden_size, dtype=self.dtype)
                )

    def get_vision_tower(self):
        # When this method is called on an instance of LlavaLlamaModel (or similar),
        # 'self' is that LlavaLlamaModel instance.
        # LlavaLlamaModel itself holds the 'vision_tower' attribute.
        # Changed self.get_model() to self
        vision_tower = getattr(self, 'vision_tower', None) # 수정된 부분
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, training_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = getattr(model_args, 'mm_vision_select_feature', 'patch')
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args, training_args=training_args)
            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)

            if 'unpad' in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of PIL image (width, height).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding:current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding:current_width - padding]

    return unpadded_tensor


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """
        Vision‑tower 출력 → hidden 차원을 마지막 축으로 통일 → CLS/패치 선택 → mm_projector
        출력 shape: (B, N, proj_dim)
        """
        # ① Vision‑tower 실행
        raw = self.get_vision_tower()(images)            # (B,H,T) or (B,T,H) or (B,H)

        # ② 2‑D → 3‑D 승격
        if raw.ndim == 2:                                # (B,H) → (B,1,H)
            raw = raw.unsqueeze(1)
        elif raw.ndim != 3:
            raise RuntimeError(f"Vision tower output must be 2‑D/3‑D, got {raw.ndim}‑D")

        B, A, C = raw.shape
        H = self.get_vision_tower().hidden_size          # 1024

        # ③ hidden 차원을 마지막 축으로
        if C == H:
            feats = raw                                  # (B,T,H)
        elif A == H:                                     # (B,H,T) → (B,T,H)
            feats = raw.transpose(1, 2)
        else:
            raise RuntimeError(f"Hidden dim {H} not found in shape {raw.shape}")

        # ④ CLS / 패치 선택
        num_cls_patch = (self.get_vision_tower().config.image_size //
                        self.get_vision_tower().config.patch_size) ** 2 + 1
        select_feature = getattr(self.get_vision_tower(), "select_feature",
                                getattr(self.config, "mm_vision_select_feature", "patch"))

        if select_feature == "patch":          # CLS 제외
            feats = feats[:, 1:num_cls_patch]
        elif select_feature == "cls_patch":    # CLS 포함
            feats = feats[:, :num_cls_patch]
        else:
            print(f"경고: 알 수 없는 select_feature '{select_feature}', cls_patch로 처리")
            feats = feats[:, :num_cls_patch]
        # ────────────────────────────────────────────────────────
        # ★ NEW: vision‑tower 출력의 dtype / device 를 projector 기준으로 통일
        #   • fp32 → fp16 충돌, multi‑GPU 충돌을 encode_images 내부에서 차단
        #   • nn.Sequential · nn.Linear 어느 쪽 projector 도 지원
        proj = self.get_model().mm_projector
        if isinstance(proj, nn.Sequential):
            first_linear = next((m for m in proj if isinstance(m, nn.Linear)), None)
            w = first_linear.weight if first_linear is not None else None
        elif isinstance(proj, nn.Linear):
           w = proj.weight
        else:
            w = None

        if w is not None:
            tgt_dtype, tgt_device = w.dtype, w.device
            if feats.dtype != tgt_dtype or feats.device != tgt_device:
                feats = feats.to(device=tgt_device, dtype=tgt_dtype, non_blocking=True)
        # ────────────────────────────────────────────────────────

        # ⑤ projector
        return self.get_model().mm_projector(feats)      # (B, N, proj_dim)


    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, image_sizes=None
    ):
        vision_tower = self.get_vision_tower()
        model_device = next(self.parameters()).device
        if vision_tower is None or images is None or (input_ids is not None and input_ids.shape[1] == 1 and not self.training):
            # Handle cases with no vision tower, no images, or during decoding steps (input_ids.shape[1] == 1)
            # if past_key_values is not None and input_ids is not None and images is not None and input_ids.shape[1] == 1:
                # This is a decoding step with past_key_values, multimodal processing for the prompt was already done.
                # We might still need to pass images if the model architecture expects it (e.g., for some cross-attention mechanism not in LLaMA).
                # However, for standard LLaMA-based LLaVA, image features are part of inputs_embeds.
                # If inputs_embeds are already formed (via past_key_values), we don't recompute them.
                # The base model's prepare_inputs_for_generation handles this.
                # For safety, if inputs_embeds is expected by the caller, it should be None here,
                # indicating that the text part needs embedding.
                # If only input_ids are passed, they get embedded.

            # ADDED: Ensure even pass-through tensors are on the correct device if not None
            if input_ids is not None: input_ids = input_ids.to(model_device)
            if position_ids is not None: position_ids = position_ids.to(model_device)
            if attention_mask is not None: attention_mask = attention_mask.to(model_device)
            if labels is not None: labels = labels.to(model_device)

            # If input_ids.shape[1] == 1, it's likely a decoding step.
            # In this case, inputs_embeds are usually handled by past_key_values logic.
            # We should not re-process image features here.
            # The `model.generate` function passes `inputs_embeds` if they are already computed.
            # This function is called when `inputs_embeds` is None (prefill stage).
            # If it's a decoding step and `inputs_embeds` is expected to be constructed,
            # then `input_ids` (single token) needs to be embedded.
            # The image features are already part of the `past_key_values` implicitly.
            # So, we just return the text `input_ids` to be embedded by the caller or subsequent steps.
            # The `None` for `inputs_embeds` signals that the caller should embed `input_ids`.
            # The original `position_ids` and `attention_mask` for this single token are passed through.
            return input_ids, position_ids, attention_mask, past_key_values, None, labels


        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
            image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
            if mm_patch_merge_type == 'flat':
                image_features = [x.flatten(0, 1) for x in image_features]
            elif mm_patch_merge_type.startswith('spatial'):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    if image_feature.shape[0] > 1: # Multi-patch image (e.g., AnyRes)
                        base_image_feature = image_feature[0]
                        image_feature_patches = image_feature[1:] # Renamed to avoid conflict
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]
                        if image_aspect_ratio == 'anyres':
                            num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, self.get_vision_tower().config.patch_size) # Use patch_size from vision_tower config
                            image_feature_patches = image_feature_patches.view(num_patch_height, num_patch_width, height, width, -1)
                        else:
                            # Assuming 'spatial' without 'anyres' means a fixed grid, e.g. 2x2 of base_image_feature
                            # This part might need specific logic if 'spatial' is used differently.
                            # For now, let's assume it's similar to 'flat' if not 'anyres'.
                            # Or, this path should be better defined.
                            # If we reach here, it means image_feature.shape[0] > 1 but not 'anyres'.
                            # This case needs clarification based on how `mm_patch_merge_type='spatial'` (non-anyres) is intended.
                            # Fallback to flattening or raise error.
                            # For simplicity, let's assume if not 'anyres', 'spatial' implies a different, predefined merging not covered here.
                            # Or, it could mean simple concatenation of multiple features if image_feature.shape[0] > 1.
                            # Given the original code, it seems to expect 'anyres' for complex spatial merging.
                            # So, if not 'anyres', and shape[0] > 1, this indicates an unhandled configuration or data.
                            # Let's ensure we use image_feature_patches.
                            # If this path is hit, it's best to clarify the intended behavior for 'spatial' non-'anyres'.
                            # For now, let's assume it implies a simpler structure or an error in setup.
                            # The original code had `image_feature = image_feature[1:]`, which I renamed to `image_feature_patches`.
                            # If `image_aspect_ratio` is not 'anyres' but `mm_patch_merge_type` is 'spatial'
                            # and `image_feature.shape[0] > 1`, this is an ambiguous case.
                            # The original `else: raise NotImplementedError` was for this path.
                            raise NotImplementedError(f"mm_patch_merge_type '{mm_patch_merge_type}' with image_aspect_ratio '{image_aspect_ratio}' and multiple initial image_features per image is not fully defined here.")

                        if 'unpad' in mm_patch_merge_type: # This part is specific to 'anyres' with 'unpad'
                            image_feature_patches = image_feature_patches.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature_patches = image_feature_patches.flatten(1, 2).flatten(2, 3)
                            image_feature_patches = unpad_image(image_feature_patches, image_sizes[image_idx])
                            image_feature_patches = torch.cat((
                                image_feature_patches,
                                self.model.image_newline[:, None, None].expand(*image_feature_patches.shape[:-1], 1).to(image_feature_patches.device) # Ensure image_newline is on correct device
                            ), dim=-1)
                            image_feature_patches = image_feature_patches.flatten(1, 2).transpose(0, 1)
                        else: # This part is for 'anyres' without 'unpad'
                            image_feature_patches = image_feature_patches.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature_patches = image_feature_patches.flatten(0, 3)
                        image_feature = torch.cat((base_image_feature, image_feature_patches), dim=0)
                    else: # Single patch image (standard processing or base_image_feature only)
                        image_feature = image_feature[0]
                        if 'unpad' in mm_patch_merge_type: # This implies it's a single feature that might need unpadding + newline
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[None].to(image_feature.device) # Ensure image_newline is on correct device
                            ), dim=0)
                    new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else: # Single image tensor (B, C, H, W) -> (B, N, D)
            image_features = self.encode_images(images)
            # If batch size is 1, image_features might be (N, D). Reshape to (1, N, D) for consistency.
            if image_features.ndim == 2:
                image_features = image_features.unsqueeze(0)


        _labels = labels # 원본 참조용
        _position_ids = position_ids # 원본 참조용
        _attention_mask = attention_mask # 원본 참조용
        
        # ADDED: Move original input_ids and labels to model_device at the beginning if they are not None
        if input_ids is not None:
            input_ids = input_ids.to(model_device)
        if labels is not None: # 여기서 labels는 함수 인자로 받은 labels
            labels = labels.to(model_device) 
        
        # Default to True for attention_mask if None, and ensure it's boolean
        if attention_mask is None and input_ids is not None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool, device=model_device) # MODIFIED: device=model_device
        elif attention_mask is not None:
            attention_mask = attention_mask.to(device=model_device, dtype=torch.bool) # MODIFIED: device=model_device and ensure bool
        
        if position_ids is None and input_ids is not None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=model_device) 
        elif position_ids is not None: # ADDED from previous suggestion, ensure it's here
            position_ids = position_ids.to(model_device)
        
        # labels가 None(즉, _labels가 None)이고 input_ids가 제공된 경우, IGNORE_INDEX로 채움
        if _labels is None and input_ids is not None: # MODIFIED: _labels (원본) 기준으로 확인
            labels = torch.full_like(input_ids, IGNORE_INDEX, device=model_device) # MODIFIED: device=model_device

        # Process per batch item if input_ids, labels, attention_mask are batched
        # The original code seemed to assume input_ids, labels were lists after unpadding.
        # Let's ensure consistent handling if they start as batched tensors.
        # If they are already lists (e.g., from a dataloader that doesn't pad), this loop is fine.
        # If they are batched tensors, they need to be iterated.
        
        # The logic `input_ids = [cur_input_ids[cur_attention_mask] ...]` unnpacks based on attention_mask.
        # This is usually done for removing padding.
        # If `input_ids` is already a list of unpadded sequences, this step might not be needed or change much.
        # Let's assume input_ids, labels, attention_mask are (batch_size, seq_len)
        
        unpadded_input_ids = []
        unpadded_labels = []
        if input_ids is not None and attention_mask is not None:
            for i in range(input_ids.shape[0]):
                current_mask = attention_mask[i]
                unpadded_input_ids.append(input_ids[i][current_mask])
                if labels is not None:
                    unpadded_labels.append(labels[i][current_mask])
                else:
                    unpadded_labels.append(None) # Or handle appropriately
        elif input_ids is not None: # attention_mask is None, assume all valid
            for i in range(input_ids.shape[0]):
                unpadded_input_ids.append(input_ids[i])
                if labels is not None:
                    unpadded_labels.append(labels[i])
                else:
                    unpadded_labels.append(None) # Or handle appropriately

        # Replace original input_ids and labels with their unpadded versions (now lists)
        input_ids = unpadded_input_ids
        if labels is not None: # only replace if labels were provided
            labels = unpadded_labels


        new_input_embeds_list = []
        new_labels_list = []
        cur_image_idx = 0
        
        # Determine the target device for new tensors, typically from the model or existing embeddings/features.
        # self.device should be the device of the main LlavaMetaForCausalLM module.
        # image_features are already on their correct device from encode_images.
        # Text embeddings will be created on self.get_model().embed_tokens.weight.device.
        # All should be consistent. Let's use self.device as a common reference if needed for new tensors like IGNORE_INDEX.
        model_device = next(self.parameters()).device # A robust way to get module's device

        for batch_idx, cur_input_ids in enumerate(input_ids): # input_ids is now a list of tensors
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum().item() # .item() for scalar
            
            current_labels = labels[batch_idx] if labels is not None and batch_idx < len(labels) else None

            # ADDED: Define embed_tokens_device here, once per batch or even outside if it's always the same
            embed_tokens_device = self.get_model().embed_tokens.weight.device

            if num_images == 0:
                # No image tokens in this sequence. Embed text tokens.
                # Ensure cur_input_ids is on the embedding layer's device.
                cur_input_ids_on_device = cur_input_ids.to(embed_tokens_device)
                cur_input_embeds_val = self.get_model().embed_tokens(cur_input_ids_on_device)
                
                # The original code had `cur_image_features[0:0]` which is an empty slice.
                # This implies if no <image> token, we don't append any image features from the batch's image_features.
                # This seems correct.
                new_input_embeds_list.append(cur_input_embeds_val)
                if current_labels is not None:
                    new_labels_list.append(current_labels.to(model_device))
                else: # If no labels for this item, create IGNORE_INDEX labels matching embeds length
                    new_labels_list.append(torch.full((cur_input_embeds_val.shape[0],), IGNORE_INDEX,
                                                      device=model_device, dtype=torch.long)) # Use long for labels
                # If image_features is a list (one feature tensor per image in batch),
                # and this text sequence corresponds to one of those images,
                # we should advance cur_image_idx if this text sequence "consumed" an image slot.
                # This depends on how image_features and input_ids are aligned.
                # Assuming one image_feature tensor per item in the input_ids list.
                if isinstance(image_features, list) and cur_image_idx < len(image_features):
                    cur_image_idx += 1 # Consume the image feature for this batch item
                elif not isinstance(image_features, list) and image_features.shape[0] > batch_idx : # Batched image_features
                     pass # cur_image_idx is not used in this way for batched features
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels_noim = []
            
            for i in range(len(image_token_indices) - 1):
                start_idx = image_token_indices[i] + 1
                end_idx = image_token_indices[i+1]
                # cur_input_ids는 unpadded_input_ids의 요소, 즉 model_device에 있어야 함
                text_id_part = cur_input_ids[start_idx:end_idx] 
                cur_input_ids_noim.append(text_id_part.to(embed_tokens_device)) # MODIFIED (from previous suggestion)
                
                if current_labels is not None: # current_labels는 unpadded_labels의 요소, 즉 model_device에 있어야 함
                    label_part = current_labels[start_idx:end_idx]
                    cur_labels_noim.append(label_part.to(model_device)) # ADDED: 명시적으로 model_device로 이동
                else: # Create IGNORE_INDEX labels for text parts if original labels are None
                    num_text_tokens = end_idx - start_idx
                    cur_labels_noim.append(torch.full((num_text_tokens,), IGNORE_INDEX,
                                                      device=model_device, dtype=torch.long))


            # Embed text parts
            if len(cur_input_ids_noim) > 0 and sum(s.numel() for s in cur_input_ids_noim) > 0: # Check if there's anything to concat
                concatenated_input_ids_noim = torch.cat(cur_input_ids_noim) 
                split_sizes_text = [x.shape[0] for x in cur_input_ids_noim]
                text_embeds = self.get_model().embed_tokens(concatenated_input_ids_noim)
                text_embeds_no_im_split = torch.split(text_embeds, split_sizes_text, dim=0)
            else: # No text tokens to embed (e.g. prompt is just <image>)
                text_embeds_no_im_split = [] # No text embeddings

            cur_new_input_embeds_parts = []
            cur_new_labels_parts = []

            for i in range(len(text_embeds_no_im_split)): # Iterate up to number of text segments
                cur_new_input_embeds_parts.append(text_embeds_no_im_split[i])
                cur_new_labels_parts.append(cur_labels_noim[i])
                if i < num_images: # If there is an image feature to insert after this text part
                    if isinstance(image_features, list): # One feature tensor per image initially
                        current_image_feature_to_insert = image_features[cur_image_idx].to(model_device)
                    else: # Batched image_features (B, N, D), take the one for this batch item
                          # This assumes image_features corresponds to the batch_idx if not a list.
                          # If multiple images per batch item, this logic needs care.
                          # The original code increments cur_image_idx, suggesting image_features is a flat list of features for all images.
                        current_image_feature_to_insert = image_features[cur_image_idx].to(model_device)

                    cur_image_idx += 1
                    cur_new_input_embeds_parts.append(current_image_feature_to_insert)
                    cur_new_labels_parts.append(torch.full((current_image_feature_to_insert.shape[0],), IGNORE_INDEX,
                                                           device=model_device, dtype=torch.long))
            
            # If the prompt ends with <image> and there are no trailing text tokens
            # but there are still image tokens to account for (num_images >= len(text_embeds_no_im_split))
            if num_images > 0 and (len(text_embeds_no_im_split) == num_images) and \
               (cur_input_ids[image_token_indices[num_images]+1:].numel() == 0):
                # This means an image was the last part of the prompt corresponding to text_embeds_no_im_split
                # And we need to add its features if not already added.
                # The loop `for i in range(len(text_embeds_no_im_split))` would have inserted `num_images` image features
                # if each text part was followed by an image.
                # This case is for when <image> is at the end.
                pass # The loop structure should handle this. If i < num_images, it appends.

            # If there are no text parts at all (e.g. prompt is only "<image>")
            if not text_embeds_no_im_split and num_images > 0: # ADDED: Handles case like prompt="<image>"
                 for _ in range(num_images):
                    if isinstance(image_features, list):
                        current_image_feature_to_insert = image_features[cur_image_idx].to(model_device)
                    else:
                        current_image_feature_to_insert = image_features[cur_image_idx].to(model_device)
                    cur_image_idx += 1
                    cur_new_input_embeds_parts.append(current_image_feature_to_insert)
                    cur_new_labels_parts.append(torch.full((current_image_feature_to_insert.shape[0],), IGNORE_INDEX, device=model_device, dtype=torch.long))


            if not cur_new_input_embeds_parts: # Handle empty prompt after processing
                # This could happen if input_ids was empty for this batch item.
                # Add a dummy embedding and label if necessary, or skip.
                # For now, let's assume this means an empty sequence, which might error later during padding.
                # It's better to ensure input_ids are not empty for any batch item.
                # If it can be empty, we might need to create a single zero embedding.
                # Let's use a small non-empty embedding to avoid issues with torch.cat on empty list.
                # This case should ideally be prevented by earlier checks or input validation.
                # Fallback: create a single zero vector embedding of appropriate dimension
                dummy_dim = self.get_model().config.hidden_size
                cur_new_input_embeds_parts.append(torch.zeros(1, dummy_dim, device=model_device, dtype=self.dtype))
                cur_new_labels_parts.append(torch.full((1,), IGNORE_INDEX, device=model_device, dtype=torch.long))


            cur_new_input_embeds_final = torch.cat(cur_new_input_embeds_parts)
            cur_new_labels_final = torch.cat(cur_new_labels_parts)

            new_input_embeds_list.append(cur_new_input_embeds_final)
            new_labels_list.append(cur_new_labels_final)

        # Truncate sequences to max length
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds_list = [x[:tokenizer_model_max_length] for x in new_input_embeds_list]
            new_labels_list = [x[:tokenizer_model_max_length] for x in new_labels_list]

        # Combine them by padding
        if not new_input_embeds_list: # If the list is empty after processing all batch items (e.g. batch_size=0)
            # This is an edge case. Return None or empty tensors matching expected output structure.
            # For now, let's assume the batch size is > 0.
            # If new_input_embeds_list is empty, the subsequent max_len and padding will fail.
            # This should be caught by an assertion or check earlier if batch_size can be 0.
            # Let's return early if the list is empty.
            # The function expects to return (None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels)
            # Return Nones that match this structure if no processing happened.
             return None, _position_ids, _attention_mask, past_key_values, None, _labels


        max_len = max(x.shape[0] for x in new_input_embeds_list) if new_input_embeds_list else 0
        batch_size = len(new_input_embeds_list)

        # Determine dtype for new_input_embeds_padded from the first item, or self.dtype
        # Determine dtype for new_labels_padded from the first item, or torch.long
        embed_dtype = new_input_embeds_list[0].dtype if new_input_embeds_list else self.dtype # self.dtype refers to LlavaMetaModel's dtype
        label_dtype = new_labels_list[0].dtype if new_labels_list and new_labels_list[0] is not None else torch.long


        new_input_embeds_padded = torch.zeros((batch_size, max_len, new_input_embeds_list[0].shape[1]), dtype=embed_dtype, device=model_device)
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=label_dtype, device=model_device)
        
        # Use the original attention_mask's dtype and device for the new one if _attention_mask was provided
        # Otherwise, use bool and model_device
        final_attn_mask_dtype = _attention_mask.dtype if _attention_mask is not None else torch.bool
        final_attn_mask_device = _attention_mask.device if _attention_mask is not None else model_device
        attention_mask_padded = torch.zeros((batch_size, max_len), dtype=final_attn_mask_dtype, device=final_attn_mask_device)

        # Use the original position_ids's dtype and device if _position_ids was provided
        # Otherwise, use long and model_device
        final_pos_ids_dtype = _position_ids.dtype if _position_ids is not None else torch.long
        final_pos_ids_device = _position_ids.device if _position_ids is not None else model_device
        position_ids_padded = torch.zeros((batch_size, max_len), dtype=final_pos_ids_dtype, device=final_pos_ids_device)


        for i, (cur_new_embed, cur_new_label_item) in enumerate(zip(new_input_embeds_list, new_labels_list)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded[i, -cur_len:] = cur_new_embed
                if cur_len > 0 and cur_new_label_item is not None:
                    new_labels_padded[i, -cur_len:] = cur_new_label_item
                attention_mask_padded[i, -cur_len:] = True
                position_ids_padded[i, -cur_len:] = torch.arange(0, cur_len, dtype=final_pos_ids_dtype, device=final_pos_ids_device)
            else: # Right padding
                new_input_embeds_padded[i, :cur_len] = cur_new_embed
                if cur_len > 0 and cur_new_label_item is not None:
                    new_labels_padded[i, :cur_len] = cur_new_label_item
                attention_mask_padded[i, :cur_len] = True
                position_ids_padded[i, :cur_len] = torch.arange(0, cur_len, dtype=final_pos_ids_dtype, device=final_pos_ids_device)
        
        # Final processed inputs
        final_new_input_embeds = new_input_embeds_padded
        final_new_labels = new_labels_padded if _labels is not None else None # Return None for labels if original was None
        final_attention_mask = attention_mask_padded if _attention_mask is not None else None # Return None if original was None
        final_position_ids = position_ids_padded if _position_ids is not None else None # Return None if original was None


        return None, final_position_ids, final_attention_mask, past_key_values, final_new_input_embeds, final_new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
