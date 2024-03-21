# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import contextlib
import logging
import math
import warnings
import mlflow
from typing import Any, Dict, Literal, Mapping, Optional, Tuple, Union

from composer.utils import dist, parse_uri
from omegaconf import DictConfig, ListConfig
from omegaconf import OmegaConf as om

from llmfoundry.models.utils import init_empty_weights

log = logging.getLogger(__name__)


def pop_config(cfg: DictConfig,
               key: str,
               must_exist: bool = True,
               default_value: Any = None,
               convert: bool = False) -> Any:
    """Pop a value from the main config file and return it.

    If the key does not exist, return the default_value or raise a RuntimeError
    depending on the must_exist flag. If the convert flag is set to True, then
    we will convert the value to a python object using OmegaConf.to_container.
    """
    value = cfg.pop(key, None)
    if value is not None and convert:
        if not isinstance(value, DictConfig) and not isinstance(
                value, ListConfig):
            raise ValueError(
                f'The key {key} has a value of type {type(value)} that cannot be \
                            converted to a dict or list. Please check your yaml.'
            )
        return om.to_container(value)
    elif value is not None:
        return value
    elif must_exist:
        raise NameError(
            f'The {key} parameter is missing and must exist for execution. Please check your yaml.'
        )
    else:
        return default_value


def calculate_batch_size_info(
    global_batch_size: int, device_microbatch_size: Union[int, Literal['auto']]
) -> Tuple[int, Union[int, Literal['auto']], Union[int, Literal['auto']]]:
    if global_batch_size % dist.get_world_size() != 0:
        raise ValueError(
            f'Global batch size {global_batch_size} is not divisible by {dist.get_world_size()} '
            +
            'as a result, the batch size would be truncated, please adjust `global_batch_size` '
            + f'to be divisible by world size, {dist.get_world_size()}.')
    device_batch_size = global_batch_size // dist.get_world_size()
    if device_microbatch_size == 'auto':
        device_grad_accum = 'auto'
    elif isinstance(device_microbatch_size, int):
        if device_microbatch_size > device_batch_size:
            log.warn(
                f'device_microbatch_size > device_batch_size, ' +
                f'will be reduced from {device_microbatch_size} -> {device_batch_size}.'
            )
            device_microbatch_size = device_batch_size
        device_grad_accum = math.ceil(device_batch_size /
                                      device_microbatch_size)
    else:
        raise ValueError(f'Not sure how to parse {device_microbatch_size=}')

    return device_batch_size, device_microbatch_size, device_grad_accum


# Coming soon: this conversion math will be done inside Composer Trainer
def update_batch_size_info(cfg: DictConfig) -> DictConfig:
    device_train_batch_size, device_train_microbatch_size, device_train_grad_accum = calculate_batch_size_info(
        cfg.global_train_batch_size, cfg.device_train_microbatch_size)
    cfg.n_gpus = dist.get_world_size()
    cfg.device_train_batch_size = device_train_batch_size
    cfg.device_train_microbatch_size = device_train_microbatch_size
    cfg.device_train_grad_accum = device_train_grad_accum
    # Safely set `device_eval_batch_size` if not provided by user
    if 'device_eval_batch_size' not in cfg:
        if cfg.device_train_microbatch_size == 'auto':
            cfg.device_eval_batch_size = 1  # TODO debug auto eval microbatching
        else:
            cfg.device_eval_batch_size = cfg.device_train_microbatch_size
    return cfg


def process_init_device(model_cfg: DictConfig, fsdp_config: Optional[Dict]):
    # Restrict model init_device to 'meta' and 'cpu',
    # using 'cuda' vs. 'cuda:id' is tricky and can lead to common user errors
    # when multiple GPUs are available.
    # Also 'meta' is only valid when using FSDP
    init_context = contextlib.nullcontext()
    if 'init_device' in model_cfg:
        assert model_cfg.init_device in ['meta', 'cpu', 'mixed']
        if fsdp_config is None and model_cfg.init_device == 'meta':
            warnings.warn(
                "Using `cfg.model.init_device='meta'` is only valid when using FSDP! " +\
                "Reverting to `cfg.model.init_device='cpu'`.")
            model_cfg.init_device = 'cpu'
        if model_cfg.init_device == 'meta':
            init_context = init_empty_weights()
        if model_cfg.init_device == 'mixed':
            if fsdp_config is None:
                raise NotImplementedError(
                    'Using init_device `mixed` is only supported with FSDP. ' +
                    'Please add a FSDP config.')
            # Always set `sync_module_states` to True for mixed initialization
            if not fsdp_config.get('sync_module_states', False):
                warnings.warn((
                    'Setting `sync_module_states = True` for FSDP. This is required '
                    'when using mixed initialization.'))
                fsdp_config['sync_module_states'] = True

            # Set defaults for mixed initialization
            fsdp_config.setdefault('use_orig_params', False)
            fsdp_config.setdefault('load_monolith_rank0_only', True)

    # No mixed precision needed for weights when they're already 16 bits
    master_dtype = model_cfg.get('master_weights_dtype')
    small_dtypes = ('bf16', 'fp16', 'float16', 'bfloat16', 'amp_fp16',
                    'amp_bf16')
    if fsdp_config and master_dtype in small_dtypes:
        reduce_dtype = None
        buffer_dtype = None
        mixed_precision = fsdp_config.get('mixed_precision')
        if isinstance(mixed_precision, Mapping):
            reduce_dtype = mixed_precision.get('reduce_dtype')
            buffer_dtype = mixed_precision.get('buffer_dtype')
        fsdp_config['mixed_precision'] = {
            'param_dtype': None,
            'reduce_dtype': reduce_dtype,
            'buffer_dtype': buffer_dtype,
            'keep_low_precision_grads': True,
        }

    return init_context


def log_config(cfg: DictConfig) -> None:
    """Logs the current config and updates the wandb and mlflow configs.

    This function can be called multiple times to update the wandb and MLflow
    config with different variables.
    """
    print('--- Logging Config ---')
    print(om.to_yaml(cfg))
    if 'wandb' in cfg.get('loggers', {}):
        try:
            import wandb
        except ImportError as e:
            raise e
        if wandb.run:
            wandb.config.update(om.to_container(cfg, resolve=True))

    if 'mlflow' in cfg.get('loggers', {}):
        print('--- MLFlow Detected ---')
        try:
            import mlflow
        except ImportError as e:
            raise e
        if mlflow.active_run():
            mlflow.log_params(params=om.to_container(cfg, resolve=True))
            data_stores = log_dataset_uri(cfg)
            for ds in data_stores:
                mlflow.log_input(ds)

def parse_source_dataset(cfg: DictConfig):
    """
    This function parses a run config for dataset information related to training and evaluation stages. 
    It supports extracting paths from different sources including local filesystem, remote locations, Hugging Face datasets, 
    Delta tables, and UC volume paths. The function aggregates unique dataset identifiers and their types from the configuration.

    Args:
        cfg (DictConfig): run configuration

    Returns:
        List[Tuple[str, str]]: A set of tuples where each tuple represents a dataset type ('local', 'hf', 'delta_table', 'uc_volume', 
        remote backend) and the corresponding dataset path or identifier. 
    """
    paths = set()
    data_paths = []

    for data_split in ['train', 'eval']:
        local_path = cfg.get('parameters', {}).get(f'{data_split}_loader', {}).get('dataset', {}).get('local')
        remote_path = cfg.get('parameters', {}).get(f'{data_split}_loader', {}).get('dataset', {}).get('remote')
        hf_path = cfg.get('parameters', {}).get(f'{data_split}_loader', {}).get('dataset', {}).get('hf_name')
        source_dataset_path = cfg.get('parameters', {}).get('metadata', {}).get(f'source_dataset_{data_split}', {})
        delta_table_path = source_dataset_path if source_dataset_path and source_dataset_path.split('.') >= 3 else None
        uc_volume_path = source_dataset_path if source_dataset_path and source_dataset_path.startswith('/Volumes') else None

        if delta_table_path and (delta_table_path not in paths):
            data_paths.append(('delta_table', delta_table_path))
            paths.add(delta_table_path)

        elif uc_volume_path and (uc_volume_path not in paths):
            data_paths.append(('uc_volume', uc_volume_path))
            paths.add(uc_volume_path)

        elif hf_path and (hf_path not in paths):
            data_paths.append(('hf', hf_path))
            paths.add(hf_path)

        elif remote_path and (remote_path not in paths):
            backend, _, _ = parse_uri(remote_path)
            data_paths.append((backend, remote_path))
            paths.add(remote_path)

        elif local_path and (local_path not in paths):
            data_paths.append(('local', local_path))
            paths.add(local_path)

    return data_paths

def log_dataset_uri(cfg: DictConfig) -> mlflow.data.meta_dataset.MetaDataset:
    """
    Extracts dataset information from the provided configuration and translates it into 
    MLFlow-compatible dataset source instances.

    Args:
        cfg (DictConfig): The run configuration object containing dataset definitions.

    Returns:
        List[mlflow.data.meta_dataset.MetaDataset]: A list of MetaDataset instances, 
    """
    # Figure out which data source to use
    data_paths = parse_source_dataset(cfg)
    print(f'--- Data Path Found: {data_paths} ---')

    dataset_source_mapping = {
        's3': mlflow.data.filesystem_dataset_source.FileSystemDatasetSource,
        'oci': mlflow.data.filesystem_dataset_source.FileSystemDatasetSource,
        'https': mlflow.data.http_dataset_source.HTTPDatasetSource,
        'hf': mlflow.data.huggingface_dataset_source.HuggingFaceDatasetSource,
        'delta_table': mlflow.data.delta_dataset_source.DeltaDatasetSource,
        'uc_volume': mlflow.data.filesystem_dataset_source.FileSystemDatasetSource,
        'local': mlflow.data.filesystem_dataset_source.FileSystemDatasetSource,
    }

    data_stores = []
    for dataset_type, path in data_paths:
        source_class = dataset_source_mapping.get(dataset_type)
        
        if source_class:
            if dataset_type == 'delta_table':
                source = source_class(delta_table_name=path)
            else:
                source = source_class(uri=path) if hasattr(source_class, 'uri') else source_class(path=path)
        else:
            log.info(f'{dataset_type} unknown, defaulting to filesystem dataset source')
            source = mlflow.data.filesystem_dataset_source.FileSystemDatasetSource(uri=path)

        data_stores.append(mlflow.data.meta_dataset.MetaDataset(source))
    return data_stores