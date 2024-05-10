import warnings
from typing import (Any, Dict, List, Mapping, MutableMapping, Optional, Tuple,
                    Union)

import math
import torch
import torch.nn.functional as F
from composer.utils import dist
from torch import nn
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from llmfoundry import attn_bias_shape, build_attn_bias
from llmfoundry.models.alaya.configuration_alaya import AlayaConfig
from llmfoundry.models.layers import NORM_CLASS_REGISTRY, SharedEmbedding
from llmfoundry.models.layers.blocks import AlayaBlock
from llmfoundry.models.mpt.modeling_mpt import gen_rotary_embedding, log, apply_sequence_id, \
    gen_attention_mask_in_length
from llmfoundry.models.utils import MODEL_INIT_REGISTRY
from composer.models import HuggingFaceModel
from composer.metrics import (InContextLearningCodeEvalAccuracy,
                              InContextLearningLMAccuracy,
                              InContextLearningLMExpectedCalibrationError,
                              InContextLearningMCExpectedCalibrationError,
                              InContextLearningMultipleChoiceAccuracy,
                              InContextLearningQAAccuracy)
from composer.metrics.nlp import LanguageCrossEntropy, LanguagePerplexity
from omegaconf import DictConfig
from omegaconf import OmegaConf

from llmfoundry.models.utils.hf_prefixlm_converter import add_bidirectional_mask_if_missing


class AlayaPreTrainedModel(PreTrainedModel):
    config_class = AlayaConfig
    base_model_prefix = 'model'
    _no_split_modules = ['AlayaBlock']


class AlayaModel(AlayaPreTrainedModel):
    def __init__(self, config: AlayaConfig):
        config._validate_config()
        super().__init__(config)

        self.attn_impl = config.attn_config['attn_impl']
        self.prefix_lm = config.attn_config['prefix_lm']
        self.attn_uses_sequence_id = config.attn_config['attn_uses_sequence_id']
        self.alibi = config.attn_config['alibi']
        self.alibi_bias_max = config.attn_config['alibi_bias_max']

        self.learned_pos_emb = config.learned_pos_emb

        if config.init_device == 'mixed':
            if dist.get_local_rank() == 0:
                config.init_device = 'cpu'
            else:
                config.init_device = 'meta'

        if config.norm_type.lower() not in NORM_CLASS_REGISTRY.keys():
            norm_options = '|'.join(NORM_CLASS_REGISTRY.keys())
            raise NotImplementedError(
                f"Received norm_type={config.norm_type}, but only {norm_options} are supported."
            )

        norm_class = NORM_CLASS_REGISTRY[config.norm_type.lower()]

        self.embedding_fraction = config.embedding_fraction

        self.wte = SharedEmbedding(config.vocab_size, config.d_model, device=config.init_device)

        if self.learned_pos_emb:
            self.wpe = torch.nn.Embedding(config.max_seq_len, config.d_model, device=config.init_device)

        self.emb_drop = nn.Dropout(config.emb_pdrop)

        self.blocks = nn.ModuleList(
            [
                AlayaBlock(
                    device=config.init_device,
                    **config.to_dict()
                )
                for _ in range(config.n_layers)
            ]
        )

        self.norm_f = norm_class(config.d_model, device=config.init_device)

        self.rope = config.attn_config['rope']

        self.rope_impl = None

        if self.rope:
            self.rope_impl = config.attn_config['rope_impl']
            self.rotary_embedding = gen_rotary_embedding(
                rope_head_dim=config.d_model // config.n_heads,
                rope_impl=self.rope_impl,
                rope_theta=config.attn_config['rope_theta'],
                rope_dail_config=config.attn_config['rope_dail_config'],
                rope_hf_config=config.attn_config['rope_hf_config'],
                max_seq_len=self.config.max_seq_len)

        if config.init_device != 'meta':
            log.info(
                f'We recommend using config.init_device="meta" with Composer + FSDP for faster initialization.'
            )
            self.apply(self.param_init_fn)

        self.is_causal = not self.prefix_lm

        self._attn_bias_initialized = False

        self.attn_bias_shape = attn_bias_shape(
            self.attn_impl,
            config.n_heads,
            config.max_seq_len,
            self.alibi,
            prefix_lm=self.prefix_lm,
            causal=self.is_causal,
            use_sequence_id=self.attn_uses_sequence_id,
        )

        if config.no_bias:
            for module in self.modules():
                if hasattr(module, 'bias') and isinstance(
                        module.bias, nn.Parameter):
                    log.info(f'Removing bias ({module.bias}) from {module}.')
                    module.register_parameter('bias', None)

                # For transformer engine
                if hasattr(module, 'use_bias'):
                    log.info(f'Setting use_bias=False for {module}.')
                    module.use_bias = False

        log.debug(self)
        log.debug(f'Using {self.config.init_config["name"]} initialization.')

    @torch.no_grad()
    def _attn_bias(
            self,
            device: torch.device,
            dtype: torch.dtype,
            attention_mask: Optional[torch.ByteTensor] = None,
            prefix_mask: Optional[torch.ByteTensor] = None,
            sequence_id: Optional[torch.LongTensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.ByteTensor]]:
        if not self._attn_bias_initialized:
            self.attn_bias = torch.zeros(self.attn_bias_shape,
                                         device=device,
                                         dtype=dtype)

            self.attn_bias = build_attn_bias(
                self.attn_impl,
                self.attn_bias,
                self.config.n_heads,
                self.config.max_seq_len,
                causal=self.is_causal,
                alibi=self.alibi,
                alibi_bias_max=self.alibi_bias_max,
            )

            self._attn_bias_initialized = True

        attn_bias = self.attn_bias
        # If using torch or triton, we incorporate the prefix_mask (if appropriate)
        if self.prefix_lm:
            assert isinstance(attn_bias, torch.Tensor)  # pyright
            assert isinstance(prefix_mask, torch.Tensor)  # pyright
            attn_bias = self._apply_prefix_mask(attn_bias, prefix_mask)

        # If using torch or triton, we incorporate sequence_id (if appropriate)
        if self.attn_uses_sequence_id and sequence_id is not None:
            assert isinstance(attn_bias, torch.Tensor)  # pyright
            attn_bias = apply_sequence_id(attn_bias, sequence_id,
                                          self.config.max_seq_len)

        if attention_mask is not None:
            s_k = attention_mask.shape[-1]
            if attn_bias is None:
                attn_bias = torch.zeros((1, 1, 1, s_k),
                                        device=device,
                                        dtype=dtype)
            else:
                _s_k = max(0, attn_bias.size(-1) - s_k)
                attn_bias = attn_bias[:, :, :, _s_k:]
            if prefix_mask is not None and (attention_mask.shape !=
                                            prefix_mask.shape):
                raise ValueError(
                    f'attention_mask shape={attention_mask.shape} ' +
                    f'and prefix_mask shape={prefix_mask.shape} are not equal.')
            min_val = torch.finfo(attn_bias.dtype).min
            attn_bias = attn_bias.masked_fill(
                ~attention_mask.view(-1, 1, 1, s_k), min_val)

        return attn_bias, attention_mask

    def _apply_prefix_mask(self, attn_bias: torch.Tensor, prefix_mask: torch.Tensor) -> torch.Tensor:
        s_k, s_q = attn_bias.shape[-2:]
        if (s_k != self.config.max_seq_len) or (s_q != self.config.max_seq_len):
            raise ValueError(
                'attn_bias does not match the expected shape. ' +
                f'The last two dimensions should both be {self.config.max_length} '
                + f'but are {s_k} and {s_q}.')

        seq_len = prefix_mask.shape[-1]
        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f'prefix_mask sequence length cannot exceed max_seq_len={self.config.max_seq_len}'
            )

        attn_bias = attn_bias[:, :seq_len, :seq_len]

        causal = torch.tril(
            torch.ones((seq_len, seq_len),
                       dtype=torch.bool,
                       device=prefix_mask.device)).view(1, 1, seq_len, seq_len)
        prefix = prefix_mask.view(-1, 1, 1, seq_len)
        cannot_attend = ~torch.logical_or(causal, prefix.bool())

        min_val = torch.finfo(attn_bias.dtype).min
        attn_bias = attn_bias.masked_fill(cannot_attend, min_val)

        return attn_bias

    def param_init_fn(self, module: nn.Module) -> None:
        init_fn_name = self.config.init_config['name']
        MODEL_INIT_REGISTRY[init_fn_name](
            module=module,
            n_layers=self.config.n_layers,
            d_model=self.config.d_model,
            **self.config.init_config,
        )

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[Tuple[torch.FloatTensor]]] = None,
            attention_mask: Optional[torch.ByteTensor] = None,
            prefix_mask: Optional[torch.ByteTensor] = None,
            sequence_id: Optional[torch.LongTensor] = None,
            return_dict: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            use_cache: Optional[bool] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
    ):
        return_dict = (return_dict if return_dict is not None else self.config.return_dict)
        use_cache = (use_cache if use_cache is not None else self.config.use_cache)

        if attention_mask is not None:
            attention_mask = attention_mask.bool()  # type: ignore

        if prefix_mask is not None:
            prefix_mask = prefix_mask.bool()

        if output_attentions:
            if self.attn_impl != 'torch':
                raise NotImplementedError(
                    'output_attentions is not implemented when using attn_impl `flash` or `triton`.'
                )

        if (self.training and attention_mask is not None and
                attention_mask[:, 0].sum() != attention_mask.shape[0]):
            raise NotImplementedError(
                'Alaya does not support training with left padding.')

        if self.training:
            if self.attn_uses_sequence_id and sequence_id is None:
                raise ValueError(
                    'sequence_id is a required argument when Alaya is configured with attn_uses_sequence_id=True '
                    + 'and the model is in train mode.')
            elif (self.attn_uses_sequence_id is False) and (sequence_id
                                                            is not None):
                warnings.warn(
                    'Alaya received non-None input for `sequence_id` but is configured with '
                    'attn_uses_sequence_id=False.'
                    +
                    'This input will be ignored. If you want the model to use `sequence_id`, '
                    'set attn_uses_sequence_id to True.'
                )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('You cannot specify both input_ids and inputs_embeds.')
        elif input_ids is not None:
            S = input_ids.size(1)
            x = self.wte(input_ids)
            input_device = input_ids.device
        elif inputs_embeds is not None:
            S = inputs_embeds.size(1)
            x = inputs_embeds
            input_device = inputs_embeds.device
        else:
            raise ValueError('You must specify either input_ids or inputs_embeds.')

        rotary_emb_w_meta_info = None

        if self.learned_pos_emb or self.rope:

            if past_key_values is not None:
                if len(past_key_values) != self.config.n_layers:
                    raise ValueError(
                        f'past_key_values must provide a past_key_value for each attention '
                        +
                        f'layer in the network ({len(past_key_values)=}; {self.config.n_layers=}).'
                    )

            # For attn_impl: triton and flash the past key tensor spec is (batch, seq, dim).
            # For attn_impl: torch the past key tensor spec is (batch, heads, head_dim, seq).
            # Here we shift position embedding using the `seq` dim of the past key

            if self.attn_impl == 'torch':
                past_position = past_key_values[0][0].size(3)
            else:
                past_position = past_key_values[0][0].size(1)

            if self.learned_pos_emb and (S + past_position >
                                         self.config.max_seq_len):
                raise ValueError(
                    f'Cannot forward input with past sequence length {past_position} and current sequence length '
                    +
                    f'{S + 1}, this model only supports total sequence length <= {self.config.max_seq_len}.'
                )

            if self.learned_pos_emb or (self.rope and self.rope_impl == 'hf'):
                pos = torch.arange(
                    past_position,
                    S + past_position,
                    dtype=torch.long,
                    device=input_device,
                ).unsqueeze(0)
                if attention_mask is not None:
                    # adjust the position indices to account for padding tokens
                    pos = torch.clamp(
                        pos - torch.cumsum((~attention_mask).to(torch.int32),
                                           dim=1)[:, past_position:],
                        min=0,
                    )
                if self.learned_pos_emb:
                    x = x + self.wpe(pos)
                elif self.rope and self.rope_impl == 'hf':
                    rotary_emb_w_meta_info = {
                        'impl': self.rope_impl,
                        'rotary_emb': self.rotary_embedding,
                        'offset_info': pos,
                        'seq_len': S + past_position,
                    }
            elif self.rope and self.rope_impl == 'dail':
                rotary_emb_w_meta_info = {
                    'impl': self.rope_impl,
                    'rotary_emb': self.rotary_embedding,
                    'offset_info': past_position,
                    'seq_len': S + past_position,
                }

        if self.embedding_fraction == 1:
            x = self.emb_drop(x)
        else:
            x_shrunk = (x * self.embedding_fraction) + (
                    x.detach() * (1 - self.embedding_fraction))
            assert isinstance(self.emb_drop, nn.Module)  # pyright
            x = self.emb_drop(x_shrunk)

        attn_bias, attention_mask = self._attn_bias(
            device=x.device,
            dtype=torch.float32,
            attention_mask=attention_mask,
            prefix_mask=prefix_mask,
            sequence_id=sequence_id,
        )
        attention_mask_in_length = gen_attention_mask_in_length(
            sequence_id=sequence_id,
            S=S,
            attn_uses_sequence_id=self.attn_uses_sequence_id,
            attn_impl=self.attn_impl,
            attention_mask=attention_mask)

        presents = () if use_cache else None

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for b_idx, block in enumerate(self.blocks):
            if output_hidden_states:
                assert all_hidden_states is not None  # pyright
                all_hidden_states = all_hidden_states + (x,)
            past_key_value = (past_key_values[b_idx]
                              if past_key_values is not None else None)

            x, attn_weights, present = block(
                x,
                past_key_value=past_key_value,
                attn_bias=attn_bias,
                rotary_emb_w_meta_info=rotary_emb_w_meta_info,
                attention_mask=attention_mask,
                is_causal=self.is_causal,
                output_attentions=bool(output_attentions),
                attention_mask_in_length=attention_mask_in_length,
            )

            if presents is not None:
                presents += (present,)

            if output_attentions:
                assert all_self_attns is not None  # pyright
                all_self_attns = all_self_attns + (attn_weights,)

            x = self.norm_f(x)

            # add hidden states from the last decoder layer
            if output_hidden_states:
                assert all_hidden_states is not None  # pyright
                all_hidden_states = all_hidden_states + (x,)

            return BaseModelOutputWithPast(
                last_hidden_state=x,
                past_key_values=presents,
                hidden_states=all_hidden_states,
                attentions=all_self_attns,
            )


class AlayaForCausalLM(AlayaPreTrainedModel):

    def __init__(self, config: AlayaConfig):
        super().__init__(config)
        log.info(f'Instantiating an AlayaForCausalLM from {__file__}].')

        self.transformer: AlayaModel = AlayaModel(config)
        for child in self.transformer.children():
            if isinstance(child, nn.ModuleList):
                continue
            if isinstance(child, torch.nn.Module):
                child._fsdp_wrap = True

        self.logit_scale = None

        if config.logit_scale:
            logit_scale = config.logit_scale
            if isinstance(logit_scale, str):
                if logit_scale == 'inv_sqrt_d_model':
                    logit_scale = 1 / math.sqrt(config.d_model)
            else:
                raise ValueError(
                    f"logit_scale={logit_scale!r} is not recognized as an option; use numeric value or "
                    f"'inv_sqrt_d_model'.")
            self.logit_scale = logit_scale

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[Tuple[torch.FloatTensor]]] = None,
            attention_mask: Optional[torch.ByteTensor] = None,
            prefix_mask: Optional[torch.ByteTensor] = None,
            sequence_id: Optional[torch.LongTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            return_dict: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            use_cache: Optional[bool] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,

    ):
        outputs = self.transformer(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            prefix_mask=prefix_mask,
            sequence_id=sequence_id,
            return_dict=return_dict,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            inputs_embeds=inputs_embeds,
        )

        hidden_states = outputs.last_hidden_state

        if self.lm_head is not None:
            logits = self.lm_head(hidden_states)
        else:
            hidden_states = hidden_states.to(self.transformer.wte.weight.device)
            logits = self.transformer.wte(hidden_states, True)

        if self.logit_scale is not None:
            if self.logit_scale == 0:
                warnings.warn(
                    f'Multiplying logits by {self.logit_scale=}. This will produce uniform (uninformative) outputs.'
                )
            logits *= self.logit_scale

        loss = None
        if labels is not None:
            _labels = torch.roll(labels, shifts=-1)
            _labels[:, -1] = -100
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                _labels.to(logits.device).view(-1),
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class ComposerMPTCausalLM(HuggingFaceModel):

    def __init__(
            self,
            om_model_config: DictConfig,
            tokenizer: Optional[PreTrainedTokenizerBase] = None,
    ):
        resolved_om_model_config = OmegaConf.to_container(om_model_config,
                                                          resolve=True)
        hf_config = AlayaConfig.from_dict(resolved_om_model_config)
        model = AlayaForCausalLM(hf_config)

        use_train_metrics = om_model_config.get('use_train_metrics', True)
        train_metrics = [LanguageCrossEntropy(),
                         LanguagePerplexity()] if use_train_metrics else []
        eval_metrics = [
            LanguageCrossEntropy(),
            LanguagePerplexity(),
            InContextLearningLMAccuracy(),
            InContextLearningMultipleChoiceAccuracy(),
            InContextLearningQAAccuracy(),
            InContextLearningCodeEvalAccuracy(),
            InContextLearningLMExpectedCalibrationError(),
            InContextLearningMCExpectedCalibrationError(),
        ]

        super().__init__(
            model=model,
            tokenizer=tokenizer,
            use_logits=True,
            metrics=train_metrics,
            eval_metrics=eval_metrics,
            shift_labels=True,
            allow_embedding_resizing=True,
        )

        self.n_active_params = sum(p.numel() for p in self.parameters())

        loss_fn_config = om_model_config.get('loss_fn', 'fused_crossentropy')
        if loss_fn_config == 'fused_crossentropy':
            try:
                # NOTE: The following is the original import statement from flash_attn library, which we have
                # currently replaced with a copy pasted code from the same library's version 1.0.9. The reason is
                # that using the CE loss from FA v2.3.2 results in an illegal memory access error at long sequence
                # lengths (github issue: https://github.com/Dao-AILab/flash-attention/issues/714). from
                # flash_attn.losses.cross_entropy import \ CrossEntropyLoss as FusedCrossEntropyLoss TODO: Once the
                #  problem with using FA v2's CE loss at longer sequence lengths is resolved (github issue:
                #  https://github.com/Dao-AILab/flash-attention/issues/714), revert back to directly importing the CE
                #  loss from FA library.
                from llmfoundry.models.layers.cross_entropy_loss import \
                    CrossEntropyLoss as FusedCrossEntropyLoss

                self.loss_fn = FusedCrossEntropyLoss(ignore_index=-100)
            except:
                raise ValueError(
                    'Fused Cross Entropy is not installed. Either (1) have a CUDA-compatible GPU '
                    +
                    'and `pip install .[gpu]` if installing from source or `pip install '
                    'xentropy-cuda-lib@git+https://github.com/HazyResearch/flash-attention.git@v1.0.3#subdirectory'
                    '=csrc/xentropy`'
                    +
                    'if installing from pypi, or (2) set your config model.loss_fn=torch_crossentropy.'
                )
        elif loss_fn_config == 'torch_crossentropy':
            self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        else:
            raise ValueError(
                f'Specified loss_fn={self.loss_fn} not recognized. `loss_fn` must be one of [`fused_crossentropy`, '
                f'`torch_crossentropy`].'
            )

    def get_targets(self, batch: Mapping) -> torch.Tensor:
        targets = torch.roll(batch['labels'], shifts=-1)
        targets[:, -1] = -100
        if self:
            return targets

    def forward(self, batch: MutableMapping) -> CausalLMOutputWithPast:
        if self.model.transformer.prefix_lm:
            add_bidirectional_mask_if_missing(batch)
        # Note: prefix_mask is only used if model.prefix_lm is True
        return self.model(
            input_ids=batch.get('input_ids', None),
            attention_mask=batch.get('attention_mask', None),
            prefix_mask=batch.get('bidirectional_mask', None),
            sequence_id=batch.get('sequence_id', None),
            inputs_embeds=batch.get('inputs_embeds', None),
        )

    def loss(self, outputs: CausalLMOutputWithPast,
             batch: Mapping) -> torch.Tensor:
        targets = self.get_targets(batch)
        return self.loss_fn(outputs.logits.view(-1, outputs.logits.size(-1)),
                            targets.view(-1))

    def flops_per_batch(self, batch: Mapping) -> int:
        # Note: this computation does not take into account padding, and assumes
        # that the dataset has been constructed without padding. Additionally, we
        # assume the backward pass is approximately 2x the forward pass

        bs, msl = batch['input_ids'].shape[0:2]
        params = self.n_active_params
        if not self.model.transformer.config.tie_word_embeddings:
            # embedding layers are lookup tables, therefore are not counted in the FLOP computation
            params -= self.model.transformer.wte.weight.numel()
        params_flops_per_token = 2 * params
        params_flops_per_seq = params_flops_per_token * msl
        attn_flops_per_seq = (self.model.config.n_layers * 2 * 2 *
                              (self.model.config.d_model * (msl ** 2)))

        return (params_flops_per_seq + attn_flops_per_seq) * 3 * bs
