# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Custom exceptions for the LLMFoundry."""
from collections.abc import Mapping
from typing import Dict, List


# Dataloader exceptions
class MissingHuggingFaceURLSplitError(ValueError):
    """Error thrown when a split is not found in a Hugging Face dataset used by
    the dataloader."""

    def __init__(self) -> None:
        message = 'When using a HuggingFace dataset from a URL, you must set the ' + \
                    '`split` key in the dataset config.'
        super().__init__(message)


class NotEnoughDatasetSamplesError(ValueError):
    """Error thrown when there is not enough data to train a model."""

    def __init__(self, dataset_name: str, split: str,
                 dataloader_batch_size: int, world_size: int,
                 full_dataset_size: int, minimum_dataset_size: int) -> None:
        self.dataset_name = dataset_name
        self.split = split
        self.dataloader_batch_size = dataloader_batch_size
        self.world_size = world_size
        self.full_dataset_size = full_dataset_size
        self.minimum_dataset_size = minimum_dataset_size
        message = (
            f'Your dataset (name={dataset_name}, split={split}) ' +
            f'has {full_dataset_size} samples, but your minimum batch size ' +
            f'is {minimum_dataset_size} because you are running on {world_size} gpus and '
            +
            f'your per device batch size is {dataloader_batch_size}. Please increase the number '
            + f'of samples in your dataset to at least {minimum_dataset_size}.')
        super().__init__(message)


## Tasks exceptions
class UnknownConversationTypeError(KeyError):
    """Error thrown when an unknown conversation type is used in a task."""

    def __init__(self, example: Mapping) -> None:
        message = f'Unknown conversation type {example=}'
        super().__init__(message)


class TooManyKeysInExampleError(ValueError):
    """Error thrown when a data sample has too many keys."""

    def __init__(self, desired_keys: set[str], keys: set[str]) -> None:
        message = f'Data sample has {len(keys)} keys in `allowed_keys`: {desired_keys} Please specify exactly one.'
        super().__init__(message)


class NotEnoughChatDataError(ValueError):
    """Error thrown when there is not enough chat data to train a model."""

    def __init__(self) -> None:
        message = 'Chat example must have at least two messages'
        super().__init__(message)


class InvalidLastChatMessageRoleError(ValueError):
    """Error thrown when the last message role in a chat example is invalid."""

    def __init__(self, last_role: str, expected_roles: set[str]) -> None:
        message = f'Invalid last message role: {last_role}. Expected one of: {expected_roles}'
        super().__init__(message)


class IncorrectMessageKeyQuantityError(ValueError):
    """Error thrown when a conversation message has an incorrect number of
    keys."""

    def __init__(self, num_keys: int) -> None:
        message = f'Expected 2 keys in message, but found {num_keys}'
        super().__init__(message)


class InvalidRoleError(ValueError):
    """Error thrown when a role is invalid."""

    def __init__(self, role: str) -> None:
        message = f'Invalid role: {role}'
        super().__init__(message)


class InvalidContentTypeError(TypeError):
    """Error thrown when the content type is invalid."""

    def __init__(self, content_type: type) -> None:
        message = f'Expected content to be a string, but found {content_type}'
        super().__init__(message)


class InvalidPromptTypeError(TypeError):
    """Error thrown when the prompt type is invalid."""

    def __init__(self, prompt_type: type) -> None:
        message = f'Expected prompt to be a string, but found {prompt_type}'
        super().__init__(message)


class InvalidResponseTypeError(TypeError):
    """Error thrown when the response type is invalid."""

    def __init__(self, response_type: type) -> None:
        message = f'Expected response to be a string, but found {response_type}'
        super().__init__(message)


class MissingLocalPathSplitError(ValueError):
    """Error thrown when a split is not found in a local dataset used by the
    dataloader."""

    def __init__(self, local: str, split: str) -> None:
        message = f'Local directory {local} does not contain split {split}'
        super().__init__(message)


class InvalidPromptResponseKeysError(ValueError):
    """Error thrown when missing expected prompt and response keys."""

    def __init__(self, mapping: Dict[str, str]):
        message = f'Expected {mapping=} to have keys "prompt" and "response".'
        super().__init__(message)


class InvalidFileExtensionError(FileNotFoundError):
    """Error thrown when a file extension is not a safe extension."""

    def __init__(self, dataset_name: str, valid_extensions: List[str]) -> None:
        message = (
            f'safe_load is set to True. No data files with safe extensions {valid_extensions} '
            + f'found for dataset at local path {dataset_name}.')
        super().__init__(message)


class UnableToProcessPromptResponseError(ValueError):
    """Error thrown when a prompt and response cannot be processed."""

    def __init__(self, input: Dict) -> None:
        message = f'Unable to extract prompt/response from {input}'
        super().__init__(message)


## Convert Delta to JSON exceptions
class ClusterDoesNotExistError(ValueError):
    """Error thrown when the cluster does not exist."""

    def __init__(self, cluster_id: str) -> None:
        message = f'Cluster with id {cluster_id} does not exist. Check cluster id and try again!'
        super().__init__(message)


class FailedToCreateSQLConnectionError(RuntimeError):
    """Error thrown when the client fails to create sql connection to Databricks
    workspace."""

    def __init__(self) -> None:
        message = 'Failed to create sql connection to db workspace. To use sql connect, you need to provide http_path and cluster_id!'
        super().__init__(message)


class FailedToConnectToDatabricksError(RuntimeError):
    """Error thrown when the client fails to connect to Databricks."""

    def __init__(self) -> None:
        message = 'Failed to create databricks connection. Check hostname and access token!'
        super().__init__(message)


class JSONOutputFolderNotLocalError(ValueError):
    """Error thrown when the output folder is not local."""

    def __init__(self) -> None:
        message = 'Check the json_output_folder and verify it is a local path!'
        super().__init__(message)


class JSONOutputFolderExistsError(RuntimeError):
    """Error thrown when the output folder already exists."""

    def __init__(self, output_folder: str) -> None:
        message = f'Output folder {output_folder} already exists and is not empty. Please remove it and retry.'
        super().__init__(message)


## Convert Text to MDS exceptions
class InputFolderMissingDataError(ValueError):
    """Error thrown when the input folder is missing data."""

    def __init__(self, input_folder: str) -> None:
        message = f'No text files were found at {input_folder}.'
        super().__init__(message)


class OutputFolderNotEmptyError(FileExistsError):
    """Error thrown when the output folder is not empty."""

    def __init__(self, output_folder: str) -> None:
        message = f'{output_folder} is not empty. Please remove or empty it and retry.'
        super().__init__(message)
