from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import os
import torch
import torch.nn.functional as F
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPooling

class LayerNorm(torch.nn.LayerNorm):
    """
    Subclass torch's LayerNorm to handle fp16/bf16.
    """
    def forward(self, x):
        # Store original dtype
        orig_dtype = x.dtype
        orig_x = x
        
        # Check if parameters are in a different dtype than float32
        if self.weight.dtype != torch.float32:
            # Cast weights and bias to float32 temporarily
            weight = self.weight.to(torch.float32)
            bias = self.bias.to(torch.float32) if self.bias is not None else None
            
            # Apply layer norm manually to avoid dtype issues
            x = x.to(torch.float32)
            mean = x.mean(-1, keepdim=True)
            var = x.var(-1, unbiased=False, keepdim=True)
            normalized = (x - mean) / torch.sqrt(var + self.eps)
            
            if weight is not None and bias is not None:
                output = normalized * weight + bias
            elif weight is not None:
                output = normalized * weight
            else:
                output = normalized
        else:
            # If weights are already float32, use the parent implementation
            x = x.to(torch.float32)
            output = super().forward(x)
        
        # Return with original dtype
        return output.to(orig_dtype)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)



class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    # ResidualAttentionBlock.attention 메서드 수정
    def attention(self, x: torch.Tensor, need_weights: bool = False):
        """
        Applies multi-head attention.
        Args:
            x (torch.Tensor): Input tensor.
            need_weights (bool): If True, returns attention weights. Defaults to False.
        Returns:
            torch.Tensor or Tuple[torch.Tensor, torch.Tensor]:
                If need_weights is False, returns attention output.
                If need_weights is True, returns (attention_output, attention_weights).
        """
        attn_mask_internal = None
        if self.attn_mask is not None:
            n_ctx = x.shape[0] # L
            # Assuming self.attn_mask is (L_max, L_max) or compatible for slicing to (L, L)
            attn_mask_internal = self.attn_mask[..., -n_ctx:, -n_ctx:].to(dtype=x.dtype, device=x.device)
        
        # self.attn is nn.MultiheadAttention.
        # Returns attn_output, attn_output_weights if need_weights=True
        # Else, only attn_output
        return self.attn(x, x, x, need_weights=need_weights, attn_mask=attn_mask_internal)

    # ResidualAttentionBlock.forward 메서드 수정
    def forward(self, x: torch.Tensor, output_attentions: bool = False):

        # # 디버깅용 print (ResidualAttentionBlock.forward 내부, self.attention 호출 직후)
        # temp_attention_output = self.attention(self.ln_1(x), need_weights=output_attentions)
        # print(f"Type of temp_attention_output: {type(temp_attention_output)}")
        # if isinstance(temp_attention_output, tuple):
        #     print(f"  Length of tuple: {len(temp_attention_output)}")
        #     for i, item in enumerate(temp_attention_output):
        #         print(f"    Item {i} type: {type(item)}, shape: {item.shape if isinstance(item, torch.Tensor) else 'N/A'}")
        # elif isinstance(temp_attention_output, torch.Tensor):
        #     print(f"  Shape of tensor: {temp_attention_output.shape}")
        # attention_mechanism_output = temp_attention_output # 원래 변수명으로 다시 할당
        """
        Forward pass for the ResidualAttentionBlock.
        Args:
            x (torch.Tensor): Input tensor.
            output_attentions (bool): If True, returns attention weights along with the output.
        Returns:
            torch.Tensor or Tuple[torch.Tensor, torch.Tensor]:
                If output_attentions is False, returns block output.
                If output_attentions is True, returns (block_output, attention_weights).
        """
        # self.attention 메서드는 need_weights 값에 따라 반환 타입이 달라짐.
        attention_mechanism_output = self.attention(self.ln_1(x), need_weights=output_attentions)
        
        attn_output_tensor = None  # 실제 어텐션 연산 결과 (텐서)
        attn_weights_tensor = None # 어텐션 가중치 (텐서 또는 None)

        if isinstance(attention_mechanism_output, tuple):
            # self.attention이 튜플을 반환했다면, (output, weights) 순서로 가정
            attn_output_tensor = attention_mechanism_output[0]
            if output_attentions: # 실제로 어텐션 가중치가 요청된 경우에만 weights를 할당
                attn_weights_tensor = attention_mechanism_output[1]
            # else: output_attentions가 False인데 튜플이 반환된 비정상적인 경우,
            #       attn_weights_tensor는 None으로 유지.
            #       (이런 경우는 nn.MultiheadAttention 기본 동작에서는 발생하지 않아야 함)
        else:
            # self.attention이 단일 텐서를 반환했다면, 그것이 output임
            attn_output_tensor = attention_mechanism_output
            # attn_weights_tensor는 None으로 유지 (output_attentions가 False인 경우)

        # 이제 attn_output_tensor는 항상 텐서여야 함. (None이 아니라고 가정)
        if attn_output_tensor is None:
            # 이 경우는 self.attention이 예상치 못하게 None을 반환했거나 로직 오류.
            # 디버깅을 위해 오류를 발생시키거나 기본값을 설정할 수 있음.
            # 여기서는 오류 발생 가능성을 명시적으로 알리는 것이 좋음.
            raise ValueError("attn_output_tensor in ResidualAttentionBlock is None after attention call.")
            
        x = x + attn_output_tensor  # 잔차 연결
        x = x + self.mlp(self.ln_2(x)) # MLP
        
        if output_attentions:
            # attn_weights_tensor가 None이 아니어야 함 (output_attentions=True이면)
            # 만약 self.attention이 튜플을 반환했으나 두번째 요소가 없거나 None이면 문제가 될 수 있음
            # (nn.MultiheadAttention은 need_weights=True면 항상 두번째 요소로 weights를 반환)
            return x, attn_weights_tensor
        else:
            return x

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


# This class integrates the intermediate gating mechanism into the vision transformer.
class VitTransformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, gate_start_layer: int, num_registers: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.gate_start_layer = gate_start_layer
        self.num_registers = num_registers

        # Use ModuleList to allow non-inplace updates and per-layer operations.
        self.resblocks = nn.ModuleList([ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])
        
        if layers >= gate_start_layer:
            self.intermediate_fusion_mlps = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(2 * width, width),
                    nn.ReLU(),
                    nn.Linear(width, 1)
                ) for _ in range(layers - gate_start_layer + 1)
            ])
            # Initialize each intermediate gating MLP
            for gate in self.intermediate_fusion_mlps:
                for m in gate.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
        else:
            self.intermediate_fusion_mlps = None

    def forward(self, x: torch.Tensor, output_hidden_states: bool = False, output_attentions: bool = False):
        """
        Forward pass for the VitTransformer.
        Args:
            x (torch.Tensor): Input tensor of shape [seq_len, batch, width].
            output_hidden_states (bool): If True, returns all hidden states.
            output_attentions (bool): If True, returns all attention weights from each block.
        Returns:
            Tuple: Depending on `output_hidden_states` and `output_attentions`, can include:
                   - x: The final output tensor.
                   - all_hidden_states: Tuple of hidden states from each block (if `output_hidden_states` is True).
                   - all_attentions: Tuple of attention weights from each block (if `output_attentions` is True).
        """
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None # Store attentions from each block

        for i, block in enumerate(self.resblocks):
            # Pass output_attentions flag to each ResidualAttentionBlock
            layer_outputs = block(x, output_attentions=output_attentions)
            
            current_block_output = layer_outputs
            if output_attentions:
                # If output_attentions was True, layer_outputs is a tuple (output, weights)
                current_block_output = layer_outputs[0]
                current_block_attentions = layer_outputs[1]
                all_attentions = all_attentions + (current_block_attentions,)
            
            x = current_block_output # Update x with the output of the current block
            
            # Gating logic (remains unchanged)
            if self.intermediate_fusion_mlps is not None and (i + 1) >= self.gate_start_layer:
                cls_token = x[0]                     # [batch, width]
                reg_tokens = x[1:1+self.num_registers] # [num_registers, batch, width]
                reg_summary = reg_tokens.mean(dim=0)   # [batch, width]
                fusion_input = torch.cat([cls_token, reg_summary], dim=-1)  # [batch, 2*width]
                gate_index = (i + 1) - self.gate_start_layer
                gate = torch.sigmoid(self.intermediate_fusion_mlps[gate_index](fusion_input))  # [batch, 1]
                fused = gate * cls_token + (1 - gate) * reg_summary
                x = torch.cat([fused.unsqueeze(0), x[1:]], dim=0)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (x,)

        # Construct the tuple of return values based on the flags
        return_values = [x] # The final output x is always returned first
        if output_hidden_states:
            return_values.append(all_hidden_states)
        if output_attentions:
            return_values.append(all_attentions)
        
        return tuple(return_values)



# For CLIP-REG GATED model
class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int, num_registers: int = 4):
        super().__init__()
        self.input_resolution = input_resolution
        self.patch_size = patch_size # <-- 추가: patch_size 저장
        self.output_dim = output_dim # output_dim은 정의되지만, LLaVA에서는 proj를 거치지 않음
        self.num_registers = num_registers

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))

        num_patches = (input_resolution // patch_size) ** 2
        num_tokens = num_patches + 1 + num_registers
        self.positional_embedding = nn.Parameter(scale * torch.randn(num_tokens, width))

        self.ln_pre = LayerNorm(width)

        self.register_tokens = nn.Parameter(torch.empty(num_registers, width)) # build_model에서 실제 값 로드

        self.transformer = VitTransformer(width, layers, heads, gate_start_layer=14, num_registers=num_registers)

        self.ln_post = LayerNorm(width)
        # self.proj 는 LLaVA 연동 시 사용되지 않음.
        # 가중치 로드 시 strict=False를 사용하므로, state_dict에 있더라도 모델 구조상 없어도 됨.
        # 만약 strict=True 로 로드해야 하거나, 다른 용도로 필요하다면 정의는 유지.
        # 여기서는 LLaVA 사용에 집중하므로 주석 처리하거나 삭제 가능 (단, build_model에서 strict=False 확인 필요)
        # self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        # 최종 fusion_mlp 는 LLaVA 출력에 필요 없으므로 주석 처리 또는 삭제
        # self.fusion_mlp = nn.Sequential(...)


    # VisionTransformer.forward 메서드 수정
    def forward(self, x: torch.Tensor, output_hidden_states: bool = False, output_attentions: bool = False):
        """
        Forward pass for the VisionTransformer.
        Args:
            x (torch.Tensor): Input image tensor.
            output_hidden_states (bool): If True, returns all hidden states.
            output_attentions (bool): If True, returns all attention weights from the transformer.
        Returns:
            BaseModelOutputWithPooling: An object containing last_hidden_state, pooler_output,
                                        hidden_states (if requested), and attentions (if requested).
        """
        x = self.conv1(x)  # [B, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # [B, width, num_patches]
        x = x.permute(0, 2, 1)  # [B, num_patches, width]

        # Prepend class and register tokens
        cls_token = self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device) # Batch-wise expansion
        cls_token = cls_token.to(x.dtype) # Ensure dtype
        register_tokens = self.register_tokens.to(x.dtype).unsqueeze(0).expand(x.shape[0], -1, -1) # [B, num_reg, width]
        
        x = torch.cat([cls_token, register_tokens, x], dim=1)  # [B, 1 + num_reg + num_patches, width]
        x = x + self.positional_embedding.to(x.dtype) # Positional embedding

        x = self.ln_pre(x) # Apply LayerNorm before transformer
        
        initial_hidden_state_for_output = None # For BaseModelOutputWithPooling
        if output_hidden_states:
            initial_hidden_state_for_output = x # Store state after ln_pre

        x_permuted = x.permute(1, 0, 2)  # Transformer expects [seq_len, batch, width]

        # Call the modified self.transformer (VitTransformer instance)
        # It will return a tuple: (final_output, optional_hidden_states, optional_attentions)
        transformer_outputs_tuple = self.transformer(
            x_permuted,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions
        )

        # Unpack the outputs from VitTransformer
        current_idx = 0
        final_transformer_block_output = transformer_outputs_tuple[current_idx] # This is x after all resblocks
        current_idx += 1
        
        intermediate_hidden_states_unpermuted = None
        if output_hidden_states:
            intermediate_hidden_states_unpermuted = transformer_outputs_tuple[current_idx]
            current_idx += 1
            
        vit_all_layer_attentions = None # Tuple of attentions from each VitTransformer layer
        if output_attentions:
            vit_all_layer_attentions = transformer_outputs_tuple[current_idx]
            # current_idx +=1 # No more items expected if this was the last conditional output

        # Permute back to [batch, seq_len, width]
        x_final_permuted_back = final_transformer_block_output.permute(1, 0, 2)
        
        last_hidden_state = self.ln_post(x_final_permuted_back) # Apply final LayerNorm
        pooled_output = last_hidden_state[:, 0] # CLS token is at index 0 for pooled output

        # Construct hidden_states tuple for output if requested
        all_hidden_states_for_output = None
        if output_hidden_states:
            all_hidden_states_for_output = (initial_hidden_state_for_output,) # Start with ln_pre output
            if intermediate_hidden_states_unpermuted is not None:
                for hidden_state_unpermuted in intermediate_hidden_states_unpermuted:
                    # Permute each intermediate hidden state back to (B, L, E)
                    all_hidden_states_for_output = all_hidden_states_for_output + (hidden_state_unpermuted.permute(1, 0, 2),)
            # Optionally, add the very last_hidden_state (after final ln_post) to the tuple
            # all_hidden_states_for_output = all_hidden_states_for_output + (last_hidden_state,)

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=all_hidden_states_for_output, # Tuple or None
            attentions=vit_all_layer_attentions     # Tuple of attentions from VitTransformer or None
        )



class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 num_registers: int, # <-- 추가
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        super().__init__()

        self.context_length = context_length

        vision_heads = vision_width // 64
        self.visual = VisionTransformer( # num_registers 전달
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim, # output_dim 전달
            num_registers=num_registers # <-- 추가
        )
        # ... (텍스트 부분 초기화) ...
        self.transformer = Transformer( # Transformer 초기화 추가 (기존 코드에 있었어야 함)
             width=transformer_width,
             layers=transformer_layers,
             heads=transformer_heads,
             attn_mask=self.build_attention_mask()
        )
        self.vocab_size = vocab_size # 추가 (기존 코드에 있었어야 함)
        self.token_embedding = nn.Embedding(vocab_size, transformer_width) # 추가
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width)) # 추가
        self.ln_final = LayerNorm(transformer_width) # 추가
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim)) # 추가
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)) # 추가

        self.initialize_parameters() # 추가 (기존 코드에 있었어야 함)

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)


        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal

        return mask


    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        n_ctx = text.shape[-1]
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding[:n_ctx].type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module, target_dtype: torch.dtype): # 기본값 제거, 명시적 전달 강제
    """Convert applicable model parameters and buffers to a target_dtype"""
    print(f"  Attempting to convert model weights to {target_dtype}...")
    conversion_count = 0

    def _convert_recursive(module: nn.Module):
        nonlocal conversion_count
        # 먼저 자식 모듈에 대해 재귀적으로 호출
        for child_name, child_module in module.named_children():
            _convert_recursive(child_module)

        # 현재 모듈의 직접적인 파라미터 변환
        for param_name, param in module.named_parameters(recurse=False): # recurse=False로 직접 파라미터만
            if param.dtype != target_dtype:
                try:
                    param.data = param.data.to(dtype=target_dtype)
                    if param.grad is not None and param.grad.dtype != target_dtype:
                        param.grad.data = param.grad.data.to(dtype=target_dtype)
                    # print(f"    Converted parameter '{param_name}' to {target_dtype}")
                    conversion_count += 1
                except Exception as e:
                    print(f"    Error converting parameter '{param_name}': {e}")
        
        # 현재 모듈의 직접적인 버퍼 변환
        for buffer_name, buffer_tensor in module.named_buffers(recurse=False): # recurse=False로 직접 버퍼만
            if buffer_tensor.dtype != target_dtype and buffer_tensor.is_floating_point(): # 부동소수점 버퍼만 변환
                try:
                    # 버퍼는 _buffers 딕셔너리를 통해 직접 재할당
                    module._buffers[buffer_name] = buffer_tensor.to(dtype=target_dtype)
                    # print(f"    Converted buffer '{buffer_name}' to {target_dtype}")
                    conversion_count +=1
                except Exception as e:
                    print(f"    Error converting buffer '{buffer_name}': {e}")

    _convert_recursive(model) # model.apply 대신 재귀 함수 사용으로 로그 상세화 가능
    # model.apply(_convert_module_params_buffers) # 또는 apply 사용 유지
    if conversion_count == 0:
        print(f"  No weights needed conversion to {target_dtype}, or conversion failed silently for some.")
    else:
        print(f"  Successfully converted {conversion_count} parameters/buffers to {target_dtype}.")


def adjust_state_dict(state_dict, regtoken_path="regtokens"):
    new_state_dict = {}

    # Expand Positional Embedding if Needed
    old_pos_embed = state_dict["visual.positional_embedding"]
    old_size = old_pos_embed.shape[0]
    new_size = 261 #old_size + 4  # # 256 [VIS] + 1 [CLS] + 4 [REG] tokens

    if old_size != new_size:
        print(f"[---! INFO !---] Expanding positional embedding from {old_size} → {new_size}")

        # Initialize new [REG] positions using mean of existing embeddings
        mean_embedding = old_pos_embed.mean(dim=0, keepdim=True)
        expanded_embedding = torch.cat([old_pos_embed, mean_embedding.repeat(4, 1)], dim=0)

        new_state_dict["visual.positional_embedding"] = expanded_embedding
    else:
        print("[---! OK !---] Positional embedding size is already correct, skipping expansion.")

    # Inject Register Tokens: Load if available, otherwise initialize empty
    if "visual.register_tokens" not in state_dict:
        print("[---! INFO !---] Register tokens missing, using torch.empty placeholder.")
        reg_tokens = []
        for i in range(1, 5):
            reg_tokens.append(torch.empty(1024, dtype=torch.float32))
            
        new_state_dict["visual.register_tokens"] = torch.stack(reg_tokens)
    else:
        print("[---! OK !---] [REG] tokens already present, skipping injection.")

    return new_state_dict



# Modify the build_model function to adjust the state_dict before loading it
def build_model(state_dict: dict, regtoken_path="regtokens", model_dtype: torch.dtype = torch.float16):
    # ... (기존 model 구조 및 파라미터 계산 로직은 동일) ...
    vision_width = state_dict["visual.conv1.weight"].shape[0]
    layer_indices = set()
    for k in state_dict.keys():
        if k.startswith("visual.transformer.resblocks.") and k.endswith(".attn.in_proj_weight"):
            try:
                layer_indices.add(int(k.split('.')[3]))
            except (IndexError, ValueError):
                continue
    vision_layers = len(layer_indices)
    if vision_layers == 0:
         vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")]) # fallback

    vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
    if "visual.register_tokens" in state_dict:
        num_registers = state_dict["visual.register_tokens"].shape[0]
    else:
        num_registers = 4 # 기본값 또는 다른 추론 로직
    print(f"Determined num_registers: {num_registers}")

    num_tokens = state_dict["visual.positional_embedding"].shape[0]
    num_patches_calc = num_tokens - 1 - num_registers
    if num_patches_calc <= 0:
         raise ValueError(f"Calculated non-positive number of patches ({num_patches_calc})")
    grid_size = round(num_patches_calc ** 0.5)
    if grid_size * grid_size != num_patches_calc:
         grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5) # fallback

    image_resolution = vision_patch_size * grid_size
    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

    model = CLIP(
        embed_dim=embed_dim, image_resolution=image_resolution, vision_layers=vision_layers,
        vision_width=vision_width, vision_patch_size=vision_patch_size, num_registers=num_registers,
        context_length=context_length, vocab_size=vocab_size, transformer_width=transformer_width,
        transformer_heads=transformer_heads, transformer_layers=transformer_layers
    )

    for key_to_del in ["input_resolution", "context_length", "vocab_size"]: # 오타 수정
        if key_to_del in state_dict:
            del state_dict[key_to_del]

    print("Loading state_dict into CLIP model (strict=False)...")
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"  Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"  Unexpected keys: {unexpected_keys}")

    print(f"Data type of model.visual.conv1.weight BEFORE explicit convert_weights: {model.visual.conv1.weight.dtype}")
    # model_dtype 인자가 torch.bfloat16으로 전달될 것임
    convert_weights(model, target_dtype=model_dtype) 
    print(f"Data type of model.visual.conv1.weight AFTER explicit convert_weights: {model.visual.conv1.weight.dtype}")

    return model.eval()
