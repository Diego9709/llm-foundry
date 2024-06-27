from typing import Dict, Optional

from transformers import PretrainedConfig


ffn_config_defaults: Dict = {
    'ffn_type': 'llamamlp',
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


class LlamaConfig(PretrainedConfig):

    def __init__(
            self,
            vocab_size: int = 32000,
            hidden_size: int = 4096,
            expansion_ratio: int = 4,
            intermediate_size: Optional[int] = None,
            num_hidden_layers: int = 32,
            num_attention_heads: int = 32,
            num_key_value_heads: Optional[int] = None,
            hidden_act: str = "silu",
            normal_type: str = 'rmsnorm',
            max_position_embeddings: int = 2048,
            initializer_range: float = 0.02,
            rms_norm_eps: float = 1e-6,
            use_cache: bool = True,
            pad_token_id: Optional[int] = None,
            bos_token_id: int = 1,
            eos_token_id: int = 2,
            pretraining_tp: int = 1,
            tie_word_embeddings: bool = False,
            rope_theta: float = 10000.0,
            rope_scaling: Optional[float] = None,
            attention_bias: bool = False,
            attention_dropout: float = 0.0,
            init_config: Optional[Dict] = None,
            **kwargs: Dict,
    ):
        """
        Initializes a Llama configuration object.

        Args:
            vocab_size (int, optional): The size of the vocabulary. Defaults to 32000.
            hidden_size (int, optional): The size of the hidden layers. Defaults to 4096.
            expansion_ratio (int, optional): The expansion ratio for the hidden layers. Defaults to 4.
            intermediate_size (int, optional): The size of the intermediate layers. Defaults to None.
            num_hidden_layers (int, optional): The number of hidden layers. Defaults to 32.
            num_attention_heads (int, optional): The number of attention heads. Defaults to 32.
            num_key_value_heads (int, optional): The number of key-value attention heads. Defaults to None.
            hidden_act (str, optional): The activation function for the hidden layers. Defaults to "silu".
            normal_type (str, optional): The type of normalization to use. Defaults to 'rmsnorm'.
            max_position_embeddings (int, optional): The maximum number of position embeddings. Defaults to 2048.
            initializer_range (float, optional): The range for weight initialization. Defaults to 0.02.
            rms_norm_eps (float, optional): The epsilon value for RMS normalization. Defaults to 1e-6.
            use_cache (bool, optional): Whether to use cache during inference. Defaults to True.
            pad_token_id (int, optional): The ID of the padding token. Defaults to None.
            bos_token_id (int, optional): The ID of the beginning-of-sequence token. Defaults to 1.
            eos_token_id (int, optional): The ID of the end-of-sequence token. Defaults to 2.
            pretraining_tp (int, optional): The pretraining type. Defaults to 1.
            tie_word_embeddings (bool, optional): Whether to tie word embeddings. Defaults to False.
            rope_theta (float, optional): The theta value for ROPE regularization. Defaults to 10000.0.
            rope_scaling (float, optional): The scaling factor for ROPE regularization. Defaults to None.
            attention_bias (bool, optional): Whether to use attention bias. Defaults to False.
            attention_dropout (float, optional): The dropout rate for attention layers. Defaults to 0.0.
            init_config (dict, optional): The initial configuration. Defaults to None.
            **kwargs: Additional keyword arguments.

        """
        self.normal_type = normal_type
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.expansion_ratio = expansion_ratio
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.pretraining_tp = pretraining_tp
        
        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self._rope_scaling_validation()
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.init_config = init_config if init_config is not None else init_config_defaults

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

        self._validate_config()

    def _rope_scaling_validation(self):
        """
        Validate the `rope_scaling` configuration.
        """
        if self.rope_scaling is None:
            return

        if not isinstance(self.rope_scaling, dict) or len(self.rope_scaling) != 2:
            raise ValueError(
                "`rope_scaling` must be a dictionary with with two fields, `type` and `factor`, "
                f"got {self.rope_scaling}"
            )
        rope_scaling_type = self.rope_scaling.get("type", None)
        rope_scaling_factor = self.rope_scaling.get("factor", None)
        if rope_scaling_type is None or rope_scaling_type not in ["linear", "dynamic"]:
            raise ValueError(
                f"`rope_scaling`'s type field must be one of ['linear', 'dynamic'], got {rope_scaling_type}"
            )
        if rope_scaling_factor is None or not isinstance(rope_scaling_factor, float) or rope_scaling_factor <= 1.0:
            raise ValueError(f"`rope_scaling`'s factor field must be a float > 1, got {rope_scaling_factor}")

    def _validate_config(self) -> None:
        pass        
        
        
    @property
    def _attn_implementation(self):
        """
        Return the attention implementation type.
        - "eager" for the LlamaAttention implementation.
        - "flash_attention_2" for the LlamaFlashAttention2 implementation.
        - "sdpa" for the LlamaSdpaAttention implementation.
    
        """
        return "flash_attention_2"
