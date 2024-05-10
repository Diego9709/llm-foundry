import warnings
from typing import Dict, Union, Any, Optional

from transformers import PretrainedConfig

from llmfoundry.models.layers.blocks import alaya_attn_config_defaults

ffn_config_defaults: Dict = {
    'ffn_type': 'alayamlp',
}

init_config_defaults: Dict = {
    'name': 'kaiming_normal_',
    'fan_mode': 'fan_in',
    'init_nonlinearity': 'relu',
    'init_div_is_residual': True,
    'emb_init_std': None,
    'emb_init_uniform_lim': None,
    'init_std': None,
    'init_gain': 0.0,
}


class AlayaConfig(PretrainedConfig):

    def __init__(self,
                 d_model: int = 2048,
                 n_heads: int = 16,
                 n_layers: int = 24,
                 expansion_ratio: Union[int, float] = 4,
                 max_seq_len: int = 2048,
                 vocab_size: int = 50368,
                 resid_pdrop: float = 0.0,
                 emb_pdrop: float = 0.0,
                 learned_pos_emb: bool = True,
                 attn_config=None,
                 ffn_config=None,
                 init_device: str = 'cpu',
                 logit_scale: Optional[Union[float, str]] = None,
                 no_bias: bool = False,
                 embedding_fraction: float = 1.0,
                 norm_type: str = 'low_precision_layernorm',
                 use_cache: bool = False,
                 init_config=None,
                 fc_type: str = 'torch',
                 tie_word_embeddings: bool = True,
                 use_pad_tok_in_ffn: bool = True,
                 **kwargs: Any):

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.expansion_ratio = expansion_ratio
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.resid_pdrop = resid_pdrop
        self.emb_pdrop = emb_pdrop
        self.learned_pos_emb = learned_pos_emb
        self.init_device = init_device
        self.logit_scale = logit_scale
        self.no_bias = no_bias
        self.embedding_fraction = embedding_fraction
        self.norm_type = norm_type
        self.use_cache = use_cache
        self.fc_type = fc_type
        self.use_pad_tok_in_ffn = use_pad_tok_in_ffn

        if init_config is None:
            self.init_config = init_config_defaults
        if ffn_config is None:
            self.ffn_config = ffn_config_defaults
        if attn_config is None:
            self.attn_config = alaya_attn_config_defaults

        if self.attn_config.get('alibi', False) or self.attn_config.get(
                'rope', False):
            self.learned_pos_emb = False
            warnings.warn(
                f'alibi or rope is turned on, setting `learned_pos_emb` to `False.`'
            )

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def _set_config_defaults(self, config: Dict[str, Any], config_defaults: Dict[str, Any]) -> None:
        for key, val in config_defaults.items():
            if key not in config:
                config[key] = val
            elif isinstance(val, dict):
                self._set_config_defaults(config[key], val)

    def _validate_config(self) -> None:
        pass
