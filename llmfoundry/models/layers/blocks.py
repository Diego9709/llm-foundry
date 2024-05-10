# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""GPT Blocks used for the GPT Model."""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from llmfoundry.models.layers.attention import ATTN_CLASS_REGISTRY
from llmfoundry.models.layers.ffn import FFN_CLASS_REGISTRY, build_ffn
from llmfoundry.models.layers.norm import NORM_CLASS_REGISTRY
from transformers.models.llama.modeling_llama import LLAMA_ATTENTION_CLASSES, LlamaMLP, LlamaRMSNorm

from llmfoundry.models.llama.configuration_llama import LlamaConfig

try:
    from flash_attn.bert_padding import unpad_input, pad_input  # type: ignore # yapf: disable # isort: skip
except ImportError as e:
    throw = ImportError("Please install flash-attention to use this feature")

attn_config_defaults: Dict = {
    'attn_type': 'multihead_attention',
    'attn_pdrop': 0.0,
    'attn_impl': 'triton',
    'qk_ln': False,
    'clip_qkv': None,
    'softmax_scale': None,
    'prefix_lm': False,
    'attn_uses_sequence_id': False,
    'sliding_window_size': -1,
    'alibi': False,
    'alibi_bias_max': 8,
    'rope': False,
    'rope_theta': 10000,
    'rope_impl': 'dail',
    'rope_dail_config': {
        'type': 'original',
        'pos_idx_in_fp32': True,
        'xpos_scale_base': 512,
    },
    'rope_hf_config': {
        'type': 'no_scaling',
        'factor': 1.0,
    },
}

alaya_attn_config_defaults: Dict = {
    'attn_type': 'multihead_attention',
    'attn_pdrop': 0.0,
    'attn_impl': 'triton',
    'qk_ln': False,
    'clip_qkv': None,
    'softmax_scale': None,
    'prefix_lm': False,
    'attn_uses_sequence_id': False,
    'sliding_window_size': -1,
    'alibi': True,
    'alibi_bias_max': 8,
    'rope': False,
    'rope_theta': 10000,
    'rope_impl': 'dail',
    'rope_dail_config': {
        'type': 'original',
        'pos_idx_in_fp32': True,
        'xpos_scale_base': 512,
    },
    'rope_hf_config': {
        'type': 'no_scaling',
        'factor': 1.0,
    },
}

llama_attn_config_defaults: Dict = {
    "attn_implementation": "flash_attention_2",

}

llama_ffn_config_defaults: Dict = {
}


class AlayaBlock(nn.Module):
    def __init__(self,
                 d_model: int,
                 n_heads: int,
                 expansion_ratio: int,
                 attn_config: Optional[Dict] = None,
                 ffn_config: Optional[Dict] = None,
                 resid_pdrop: float = 0.0,
                 norm_type: str = 'rmsnorm',
                 fc_type: str = 'torch',
                 device: Optional[str] = None,
                 no_bias: bool = False,
                 use_pad_tok_in_ffn: bool = True,
                 **kwargs: Any,
                 ):
        if attn_config is None:
            attn_config = attn_config_defaults

        if ffn_config is None:
            ffn_config = {
                'ffn_type': 'alaya_mtp',
            }

        del kwargs  # unused, just to capture any extra args from the config
        super().__init__()

        norm_class = NORM_CLASS_REGISTRY[norm_type.lower()]
        assert isinstance(attn_config['attn_type'], str)
        attn_class = ATTN_CLASS_REGISTRY[attn_config['attn_type']]

        args_to_exclude_in_attn_class = {
            'attn_type', 'prefix_lm', 'alibi', 'attn_uses_sequence_id',
            'alibi_bias_max', 'rope', 'rope_theta', 'rope_impl',
            'rope_dail_config', 'rope_hf_config'
        }
        attn_config_subset_for_attn_class = {
            k: v
            for k, v in attn_config.items()
            if k not in args_to_exclude_in_attn_class
        }

        self.norm_1 = norm_class(d_model, device=device)
        self.attn = attn_class(
            d_model=d_model,
            n_heads=n_heads,
            fc_type=fc_type,
            device=device,
            **attn_config_subset_for_attn_class,
            bias=not no_bias,
        )
        self.norm_2 = None
        if not getattr(FFN_CLASS_REGISTRY[ffn_config['ffn_type']], '_has_norm',
                       False):
            self.norm_2 = norm_class(d_model, device=device)
        self.ffn = build_ffn(
            d_model=d_model,
            expansion_ratio=expansion_ratio,
            device=device,
            bias=not no_bias,
            **ffn_config,
        )
        self.resid_attn_dropout = nn.Dropout(resid_pdrop)
        self.resid_ffn_dropout = nn.Dropout(resid_pdrop)

        self.use_pad_tok_in_ffn = use_pad_tok_in_ffn

    def forward(
            self,
            x: torch.Tensor,
            past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            attn_bias: Optional[torch.Tensor] = None,
            rotary_emb_w_meta_info: Optional[Dict] = None,
            attention_mask: Optional[torch.ByteTensor] = None,
            is_causal: bool = True,
            output_attentions: bool = False,
            attention_mask_in_length: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        a = self.norm_1(x)
        b, attn_weights, past_key_value = self.attn(
            a,
            past_key_value=past_key_value,
            attn_bias=attn_bias,
            rotary_emb_w_meta_info=rotary_emb_w_meta_info,
            attention_mask=attention_mask,
            is_causal=is_causal,
            needs_weights=output_attentions,
            attention_mask_in_length=attention_mask_in_length,
        )
        x = x + self.resid_attn_dropout(b)
        m = x
        if self.norm_2 is not None:
            m = self.norm_2(x)
        batch_size, seq_len = m.size()[:2]
        indices = None
        if not self.use_pad_tok_in_ffn:
            assert unpad_input is not None
            m, indices, _, _ = unpad_input(m, attention_mask)
        n = self.ffn(m)
        if not self.use_pad_tok_in_ffn:
            assert pad_input is not None
            n = pad_input(n, indices, batch_size, seq_len)
        x = x + self.resid_ffn_dropout(n)
        return x, attn_weights, past_key_value


class MPTBlock(nn.Module):

    def __init__(
            self,
            d_model: int,
            n_heads: int,
            expansion_ratio: int,
            attn_config: Optional[Dict] = None,
            ffn_config: Optional[Dict] = None,
            resid_pdrop: float = 0.0,
            norm_type: str = 'low_precision_layernorm',
            fc_type: str = 'torch',
            device: Optional[str] = None,
            no_bias: bool = False,
            use_pad_tok_in_ffn: bool = True,
            **kwargs: Any,
    ):
        if attn_config is None:
            attn_config = attn_config_defaults

        if ffn_config is None:
            ffn_config = {
                'ffn_type': 'mptmlp',
            }

        del kwargs  # unused, just to capture any extra args from the config
        super().__init__()

        norm_class = NORM_CLASS_REGISTRY[norm_type.lower()]
        assert isinstance(attn_config['attn_type'], str)
        attn_class = ATTN_CLASS_REGISTRY[attn_config['attn_type']]

        # necessary to avoid passing extraneous args into attn_class while allowing the use of **kwargs
        args_to_exclude_in_attn_class = {
            'attn_type', 'prefix_lm', 'alibi', 'attn_uses_sequence_id',
            'alibi_bias_max', 'rope', 'rope_theta', 'rope_impl',
            'rope_dail_config', 'rope_hf_config'
        }
        attn_config_subset_for_attn_class = {
            k: v
            for k, v in attn_config.items()
            if k not in args_to_exclude_in_attn_class
        }

        self.norm_1 = norm_class(d_model, device=device)
        self.attn = attn_class(
            d_model=d_model,
            n_heads=n_heads,
            fc_type=fc_type,
            device=device,
            **attn_config_subset_for_attn_class,
            bias=not no_bias,
        )
        self.norm_2 = None
        if not getattr(FFN_CLASS_REGISTRY[ffn_config['ffn_type']], '_has_norm',
                       False):
            self.norm_2 = norm_class(d_model, device=device)
        self.ffn = build_ffn(
            d_model=d_model,
            expansion_ratio=expansion_ratio,
            device=device,
            bias=not no_bias,
            **ffn_config,
        )
        self.resid_attn_dropout = nn.Dropout(resid_pdrop)
        self.resid_ffn_dropout = nn.Dropout(resid_pdrop)

        self.use_pad_tok_in_ffn = use_pad_tok_in_ffn

    def forward(
            self,
            x: torch.Tensor,
            past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            attn_bias: Optional[torch.Tensor] = None,
            rotary_emb_w_meta_info: Optional[Dict] = None,
            attention_mask: Optional[torch.ByteTensor] = None,
            is_causal: bool = True,
            output_attentions: bool = False,
            attention_mask_in_length: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        a = self.norm_1(x)
        b, attn_weights, past_key_value = self.attn(
            a,
            past_key_value=past_key_value,
            attn_bias=attn_bias,
            rotary_emb_w_meta_info=rotary_emb_w_meta_info,
            attention_mask=attention_mask,
            is_causal=is_causal,
            needs_weights=output_attentions,
            attention_mask_in_length=attention_mask_in_length,
        )
        x = x + self.resid_attn_dropout(b)
        m = x
        if self.norm_2 is not None:
            m = self.norm_2(x)
        batch_size, seq_len = m.size()[:2]
        indices = None
        if not self.use_pad_tok_in_ffn:
            assert unpad_input is not None
            m, indices, _, _ = unpad_input(m, attention_mask)
        n = self.ffn(m)
        if not self.use_pad_tok_in_ffn:
            assert pad_input is not None
            n = pad_input(n, indices, batch_size, seq_len)
        x = x + self.resid_ffn_dropout(n)
        return x, attn_weights, past_key_value


class LlamaBlock(nn.Module):
    def __init__(self,
                 config: LlamaConfig,
                 layer_idx: int,
                 norm_type: str = 'rmsnorm',

                 ) -> None:
        super().__init__()

        self.hidden_size = config.hidden_size

        self.self_attn = LLAMA_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)

        self.input_layernorm = NORM_CLASS_REGISTRY[norm_type.lower()](config.hidden_size)
        self.post_attention_layernorm = NORM_CLASS_REGISTRY[norm_type.lower()](config.hidden_size)

    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_value: Optional[Tuple[torch.Tensor]] = None,
                output_attentions: Optional[bool] = False,
                use_cache: Optional[bool] = False,
                **kwargs
                ) -> tuple[Any]:

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs
