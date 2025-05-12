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

    def attention(self, x: torch.Tensor):
        attn_mask = None

        if self.attn_mask is not None:
            n_ctx = x.shape[0]
            attn_mask = self.attn_mask[..., -n_ctx:, -n_ctx:].to(dtype=x.dtype, device=x.device)            

        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
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

    def forward(self, x: torch.Tensor, output_hidden_states: bool = False):
        # x shape: [seq_len, batch, width]
        all_hidden_states = () if output_hidden_states else None # 중간 출력 저장용 튜플

        for i, block in enumerate(self.resblocks):
            x = block(x) # 각 블록 통과

            # --- 게이팅 로직 적용 (블록 출력 후) ---
            # 게이팅은 CLS 토큰을 변경하므로, 저장 시점 중요.
            # HF 관례는 보통 레이어(블록) 출력 직후 저장. 게이팅은 레이어 이후의 추가 처리로 간주.
            # 따라서 게이팅 *전*의 x를 저장하는 것이 일반적일 수 있으나,
            # 여기서는 게이팅이 CLS 토큰 자체를 업데이트하므로, 업데이트 *후*의 상태가
            # 다음 레이어의 입력에 더 가까울 수 있음. 여기서는 게이팅 *후*의 x를 저장.
            if self.intermediate_fusion_mlps is not None and (i + 1) >= self.gate_start_layer:
                cls_token = x[0]                     # [batch, width]
                reg_tokens = x[1:1+self.num_registers] # [num_registers, batch, width]
                reg_summary = reg_tokens.mean(dim=0)   # [batch, width]
                fusion_input = torch.cat([cls_token, reg_summary], dim=-1)  # [batch, 2*width]
                gate_index = (i + 1) - self.gate_start_layer  # index into intermediate_fusion_mlps
                gate = torch.sigmoid(self.intermediate_fusion_mlps[gate_index](fusion_input))  # [batch, 1]
                fused = gate * cls_token + (1 - gate) * reg_summary  # fused representation
                # Instead of an in-place update, create a new tensor for x:
                x = torch.cat([fused.unsqueeze(0), x[1:]], dim=0) # 게이팅 적용된 x

            # --- 중간 출력 저장 ---
            if output_hidden_states:
                # 현재 레이어(게이팅 포함)를 통과한 후의 상태 저장
                all_hidden_states = all_hidden_states + (x,) # 튜플에 추가

        if output_hidden_states:
            return x, all_hidden_states # 최종 출력과 중간 출력 튜플 반환
        else:
            return x # 최종 출력만 반환



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


    def forward(self, x: torch.Tensor, output_hidden_states=False):
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1) # [B, num_patches, width]

        cls_token = self.class_embedding.to(x.dtype).expand(x.shape[0], 1, -1) # [B, 1, width]
        register_tokens = self.register_tokens.to(x.dtype).unsqueeze(0).expand(x.shape[0], -1, -1) # [B, num_reg, width]

        x = torch.cat([cls_token, register_tokens, x], dim=1) # [B, SeqLen, width]
        x = x + self.positional_embedding.to(x.dtype)

        # --- ln_pre 적용 전 상태 저장 (선택적, HF는 보통 임베딩 출력 포함) ---
        # hidden_states_before_ln_pre = x if output_hidden_states else None

        x = self.ln_pre(x) # [B, SeqLen, H]

        # --- 초기 입력 상태 저장 (ln_pre 후) ---
        # 이게 hidden_states의 첫번째 요소가 됨 (HF 관례)
        initial_hidden_state = x if output_hidden_states else None

        x = x.permute(1, 0, 2)  # shape: [seq_len, batch, width]

        # --- VitTransformer 호출 시 output_hidden_states 전달 ---
        transformer_outputs = self.transformer(x, output_hidden_states=output_hidden_states)

        # --- 결과 처리 ---
        if output_hidden_states:
            # 최종 블록 출력과 중간 블록 출력들 분리
            final_block_output = transformer_outputs[0] # [SeqLen, B, H]
            intermediate_hidden_states_unpermuted = transformer_outputs[1] # 튜플: 각 요소 [SeqLen, B, H]
        else:
            final_block_output = transformer_outputs # [SeqLen, B, H]
            intermediate_hidden_states_unpermuted = None

        # --- 최종 블록 출력을 원래 차원으로 복원 ---
        x_final_permuted = final_block_output.permute(1, 0, 2)  # shape: [batch, seq_len, width]

        # --- 최종 LayerNorm 적용 ---
        last_hidden_state = self.ln_post(x_final_permuted) # [B, SeqLen, 1024]

        # --- Pooler 출력 계산 ---
        pooled_output = last_hidden_state[:, 0] # CLS token [B, 1024]

        # --- hidden_states 튜플 구성 ---
        all_hidden_states = None
        if output_hidden_states:
            # 1. 초기 입력 상태 (ln_pre 후)
            all_hidden_states = (initial_hidden_state,)

            # 2. 중간 블록 출력들 (차원 복원)
            # intermediate_hidden_states_unpermuted는 None일 수 있으므로 체크
            if intermediate_hidden_states_unpermuted is not None:
                for hidden_state_unpermuted in intermediate_hidden_states_unpermuted:
                    hidden_state_permuted = hidden_state_unpermuted.permute(1, 0, 2) # [B, SeqLen, H]
                    all_hidden_states = all_hidden_states + (hidden_state_permuted,)

            # 3. 최종 출력 (ln_post 후) - HF는 보통 이것도 포함하지만, LLaVA는 last_hidden_state를 따로 쓰므로 생략 가능
            # all_hidden_states = all_hidden_states + (last_hidden_state,) # 필요 시 이 라인 활성화

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state, # 최종 출력 (ln_post 후)
            pooler_output=pooled_output,
            hidden_states=all_hidden_states    # 구성된 중간 출력 튜플 (또는 None)
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


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


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
def build_model(state_dict: dict, regtoken_path="regtokens"):
    vision_width = state_dict["visual.conv1.weight"].shape[0]
    # --- vision_layers 계산 명확화 ---
    # visual.transformer.resblocks.{i}.attn.in_proj_weight 형태의 키 개수 세기
    layer_indices = set()
    for k in state_dict.keys():
        if k.startswith("visual.transformer.resblocks.") and k.endswith(".attn.in_proj_weight"):
            try:
                layer_indices.add(int(k.split('.')[3]))
            except (IndexError, ValueError):
                continue # 형식에 맞지 않는 키는 무시
    vision_layers = len(layer_indices)
    if vision_layers == 0: # 만약 위 방식이 실패하면 원래 방식으로 시도
         vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
         print(f"Warning: Could not reliably determine vision_layers from resblock keys, using fallback method. Layers: {vision_layers}")


    vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
    # --- num_registers 결정 ---
    # state_dict에 'visual.register_tokens' 키가 있는지, 있다면 shape[0]을 사용
    if "visual.register_tokens" in state_dict:
        num_registers = state_dict["visual.register_tokens"].shape[0]
    else:
        # state_dict에 없으면 positional_embedding 크기에서 추론 시도 (덜 안정적)
        # pos_embed_len = 261 -> 1(CLS) + 4(REG) + 256(Patch) 가정
        pos_embed_len = state_dict["visual.positional_embedding"].shape[0]
        num_patches_plus_cls = (round((pos_embed_len - 1)**0.5))**2 + 1 # 예: (16*16)+1 = 257
        estimated_registers = pos_embed_len - num_patches_plus_cls
        if estimated_registers > 0 and estimated_registers < 10: # 비정상적인 값 필터링
             num_registers = estimated_registers
             print(f"Warning: 'visual.register_tokens' not found in state_dict. Estimated num_registers={num_registers} from positional embedding length {pos_embed_len}.")
        else:
             num_registers = 4 # 추론 실패 시 기본값
             print(f"Warning: Could not determine num_registers from state_dict. Using default value: {num_registers}")
    print(f"Determined num_registers: {num_registers}")

    # --- grid_size, image_resolution 계산 ---
    # positional_embedding 크기 (num_tokens) 에서 CLS(1)와 REG(num_registers) 제외하고 루트 계산
    num_tokens = state_dict["visual.positional_embedding"].shape[0]
    num_patches_calc = num_tokens - 1 - num_registers
    if num_patches_calc <= 0:
         raise ValueError(f"Calculated non-positive number of patches ({num_patches_calc}) from pos_embed_len={num_tokens}, num_registers={num_registers}")
    grid_size = round(num_patches_calc ** 0.5)
    if grid_size * grid_size != num_patches_calc:
         # 만약 제곱수가 아니면, 원래 방식(CLS만 제외) 시도
         grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
         print(f"Warning: Positional embedding length {num_tokens} doesn't fit num_registers {num_registers}. Using fallback grid_size calculation.")

    image_resolution = vision_patch_size * grid_size

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

    model = CLIP(
        embed_dim=embed_dim,                # 키워드 인자
        image_resolution=image_resolution,  # 키워드 인자
        vision_layers=vision_layers,        # 키워드 인자
        vision_width=vision_width,          # 키워드 인자
        vision_patch_size=vision_patch_size,# 키워드 인자
        num_registers=num_registers,        # 키워드 인자
        context_length=context_length,      # 키워드 인자
        vocab_size=vocab_size,              # 키워드 인자
        transformer_width=transformer_width,# 키워드 인자
        transformer_heads=transformer_heads,# 키워드 인자
        transformer_layers=transformer_layers # 키워드 인자
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    # adjust_state_dict 는 positional embedding 크기 조정 등에 필요할 수 있으므로 유지
    # 단, register_tokens 주입 로직은 위에서 이미 처리했으므로 중복될 수 있음 (adjust_state_dict 내부 확인 필요)
    # state_dict = adjust_state_dict(state_dict, regtoken_path)

    convert_weights(model)
    # --- load_state_dict 결과 출력 ---
    print("Loading state_dict into CLIP model (strict=False)...")
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"  Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"  Unexpected keys: {unexpected_keys}") # visual.proj 등이 나올 수 있음 (정상)

    return model.eval()
