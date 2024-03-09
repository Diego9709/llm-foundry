# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""A collection of common torchmetrics."""

from llmfoundry.eval.metrics.nlp import (
    InContextLearningCodeEvalAccuracy, InContextLearningLMAccuracy,
    InContextLearningLMExpectedCalibrationError,
    InContextLearningMCExpectedCalibrationError, InContextLearningMetric,
    InContextLearningMultipleChoiceAccuracy, InContextLearningGenerationAccuracy, InContextLearningLLMAsAJudge)

__all__ = [
    'InContextLearningLMAccuracy',
    'InContextLearningMultipleChoiceAccuracy',
    'InContextLearningGenerationAccuracy',
    'InContextLearningMCExpectedCalibrationError',
    'InContextLearningLMExpectedCalibrationError',
    'InContextLearningMetric',
    'InContextLearningCodeEvalAccuracy',
    'InContextLearningLLMAsAJudge'
]
