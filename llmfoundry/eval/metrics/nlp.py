# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""A collection of common torchmetrics for NLP tasks."""

import ast
import logging
import os
import re
import string
import warnings
from typing import Any, Dict, List, Optional
from composer.utils import dist, MissingConditionalImportError
import numpy as np
import torch
from copy import deepcopy
import random
from composer.utils.eval_client import (EvalClient, LambdaEvalClient,
                                        LocalEvalClient,
                                        MosaicMLLambdaEvalClient)
from torch import Tensor
from torch.nn import functional as F
from torchmetrics import Metric

log = logging.getLogger(__name__)

__all__ = [
    'InContextLearningMetric',
    'InContextLearningLMAccuracy',
    'InContextLearningMultipleChoiceAccuracy',
    'InContextLearningGenerationAccuracy',
    'InContextLearningCodeEvalAccuracy',
    'InContextLearningLMExpectedCalibrationError',
    'InContextLearningMCExpectedCalibrationError',
    'InContextLearningLLMAsAJudge'
]


class InContextLearningMetric(Metric):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.needs_batch = True

    def update(
        self,
        batch: dict,
        outputs: torch.Tensor,
        labels: torch.Tensor,
    ):
        """Abstract interface for computing an in-context learning metrics.

        The `output_logits` argument is deprecated and will be removed in v0.21 while it's functionality will
        be moved to `outputs`.

        Args:
            batch (dict): Batch must consist minimally of `input_ids` as well as any other structure needed
                to compute the metric.
            output_logits (torch.Tensor): The model outputs evaluated on the batch `input_ids`
            labels (torch.Tensor): The correct outputs.

        Raises:
            NotImplementedError: Abstract method must be implemented by subclasses
        """
        raise NotImplementedError


class InContextLearningGenerationAccuracy(InContextLearningMetric):
    r"""Computes accuracy for In-context learning (ICL) question answering (QA)
    tasks.

    ICL QA tasks consist of some number of example question answering tasks (referred to as the 'context'), followed by a test task where the model must
    match one of the possible answer aliases (referred to as the 'continuation').

    For example, the model may be provided the context below and evaluated on its ability to correctly predict the continuation.

    Context: `Question: Who was president of the United States in 2012?\nAnswer: Barack Obama\nQuestion: Is water wet?\nAnswer: `
    Continuation: [`yes`, `no`]

    Both predictions and answers will be normalized before comparison.

    Adds metric state variables:
        correct (float): The number of instances where the prediction was a prefix for any of the answer aliases.
        total (float): The number of total instances that were predicted.

    Args:
        dist_sync_on_step (bool, optional): Synchronize metric state across processes at
            each forward() before returning the value at the step. Default: ``False``.
    """

    # Make torchmetrics call update only once
    full_state_update = False

    def __init__(self, dist_sync_on_step: bool = False):
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state('correct',
                       default=torch.tensor(0.),
                       dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0.), dist_reduce_fx='sum')

    def normalize_answer(self, answer: str):
        """Lower text and remove punctuation, articles and extra whitespace.

        Copied from https://github.com/mandarjoshi90/triviaqa/blob/master/evaluation/triviaqa_evaluation.py
        """

        def remove_articles(text: str) -> str:
            return re.sub(r'\b(a|an|the)\b', ' ', text)

        def white_space_fix(text: str) -> str:
            return ' '.join(text.split())

        def handle_punc(text: str) -> str:
            exclude = set(string.punctuation +
                          ''.join([u'‘', u'’', u'´', u'`']))
            return ''.join(ch if ch not in exclude else ' ' for ch in text)

        def lower(text: str) -> str:
            return text.lower()

        def replace_underscore(text: str) -> str:
            return text.replace('_', ' ')

        return white_space_fix(
            remove_articles(handle_punc(lower(
                replace_underscore(answer))))).strip()

    def update(
        self,
        batch: Dict[str, Any],
        outputs: List[str],
        labels: List[List[str]],
    ):
        cot_delimiter = batch.get('cot_delimiter', '')
        do_normalization = batch.get('do_normalization', True)
        stopping_criteria = batch.get('stopping_criteria', None)
        for sample_output, sample_labels in zip(outputs, labels):
            final_answer = sample_output

            if stopping_criteria is not None and len(stopping_criteria) > 0:
                final_answer = re.split('|'.join(stopping_criteria),
                                        final_answer)[0]

            if cot_delimiter is not None and len(cot_delimiter) > 0:
                final_answer = final_answer.split(cot_delimiter)[-1]

            if do_normalization:
                cleaned_final_answer = self.normalize_answer(final_answer)
                cleaned_sample_labels = {
                    self.normalize_answer(label) for label in sample_labels
                }
            else:
                cleaned_final_answer = final_answer
                cleaned_sample_labels = set(sample_labels)

            if any(
                    cleaned_final_answer.startswith(label)
                    for label in cleaned_sample_labels):
                self.correct += torch.tensor(1.0)
            self.total += torch.tensor(1.0)

    def compute(self):
        assert isinstance(self.correct, Tensor)
        assert isinstance(self.total, Tensor)
        return self.correct / self.total


class InContextLearningLMAccuracy(InContextLearningMetric):
    r"""Computes accuracy for In-context learning (ICL) language modeling (LM)
    tasks.

    ICL LM tasks consist of some number of example language modeling tasks (referred to as the 'context'), followed by a test task where the model must correctly predict all the tokens
    following tokens in some passage (referred to as the 'continuation').

    For example, the model may be provided the context below and evaluated on its ability to correctly predict the continuation. Note: it doesn't matter
    whether the model correctly predicts the context tokens.

    Context: `The dog is->fuzzy\nthe water is->hot\nthe tree is->`
    Continuation: `green`

    Adds metric state variables:
        correct (float): The number of instances where the prediction masked the target.
        total (float): The number of total instances that were predicted.

    Args:
        dist_sync_on_step (bool, optional): Synchronize metric state across processes at
            each forward() before returning the value at the step. Default: ``False``.
    """

    # Make torchmetrics call update only once
    full_state_update = False

    def __init__(self, dist_sync_on_step: bool = False):
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state('correct',
                       default=torch.tensor(0.),
                       dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0.), dist_reduce_fx='sum')

    def update(self, batch: dict, outputs: torch.Tensor, labels: torch.Tensor):

        for batch_idx, cont_idx in enumerate(batch['continuation_indices']):
            cont_tok_pred = outputs[batch_idx].index_select(dim=0,
                                                            index=cont_idx -
                                                            1).argmax(dim=-1)
            cont_tok_targ = labels[batch_idx].index_select(dim=0,
                                                           index=cont_idx - 1)

            self.correct += (cont_tok_pred == cont_tok_targ).all().int()
            self.total += torch.tensor(1.0)

    def compute(self):
        assert isinstance(self.correct, Tensor)
        assert isinstance(self.total, Tensor)
        return self.correct / self.total


class InContextLearningMultipleChoiceAccuracy(InContextLearningMetric):
    r"""Computes accuracy for In-context learning (ICL) multiple choice (MC)
    tasks.

    ICL MC tasks consists of a series of questions with some number of possible choices (only one of which can be correct).
    At inference time each possible choice is given to the model as a separate input and the one for which the model assigns
    the lowest perplexity to the choice is considered the model's choice. The model is correct if it "chooses" the right answer.

    Context: `The dog is->fuzzy\nthe water is->hot\nthe tree is->`
    Continuation: `green`

    Adds metric state variables:
        correct (float): The number of instances where the prediction masked the target.
        total (float): The number of total instances that were predicted.

    Args:
        dist_sync_on_step (bool, optional): Synchronize metric state across processes at
            each forward() before returning the value at the step. Default: ``False``.
    """

    # Make torchmetrics call update only once
    full_state_update = False

    def __init__(self, dist_sync_on_step: bool = False):
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state('correct',
                       default=torch.tensor(0.0),
                       dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0.0), dist_reduce_fx='sum')

    def update(self, batch: dict, outputs: torch.Tensor, labels: torch.Tensor):

        perplexities = []
        for batch_idx, cont_idx in enumerate(batch['continuation_indices']):
            # continuation indices refer to indices in the original input's token space
            cont_tok_logits = outputs[batch_idx].index_select(dim=0,
                                                              index=cont_idx -
                                                              1)
            # labels have been shifted left by one index, so the cont_idx needs to be shifted as well.
            cont_tok_targ = labels[batch_idx].index_select(dim=0,
                                                           index=cont_idx - 1)
            cross_entropy = F.cross_entropy(cont_tok_logits, cont_tok_targ)
            perplexity = torch.exp(cross_entropy)
            perplexities.append(perplexity)

        for (start, end), gold_idx in zip(batch['choice_groupings'],
                                          batch['gold_indices']):
            subset = perplexities[start:end]
            idx_min = subset.index(min(subset))

            if idx_min == gold_idx:
                self.correct += torch.tensor(1.0)
            self.total += torch.tensor(1.0)

    def compute(self):
        assert isinstance(self.correct, Tensor)
        assert isinstance(self.total, Tensor)
        return self.correct.float() / self.total


class InContextLearningExpectedCalibrationError(InContextLearningMetric):
    """Generic class for Expected Calibration Error (ECE) (cite:
    https://arxiv.org/pdf/1706.04599.pdf).

    Expected calibration error is calculated by dividing predictions into buckets based on the model's confidence (a probability value between 0 and 1).
    We then calculate the accuracy within each bucket and calculate the average gap between confidence and accuracy
    across buckets, weighted by the number of samples in each bucket.

    Each task must implement its own definition of "confidence" to be computed via the `update` method.

    Adds metric state variables:
    bucket_totals (float): The number of instances where the prediction masked the target per bucket.
    bucket_correct (float): The number of total instances that were predicted per bucket.

    Args:
        dist_sync_on_step (bool, optional): Synchronize metric state across processes at
            each forward() before returning the value at the step. Default: ``False``.
        n_buckets (int): Number of distinct buckets to split the confidence distribution into
    """

    def __init__(self, dist_sync_on_step: bool = False, n_buckets: int = 10):
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.n_buckets = n_buckets
        if n_buckets < 1:
            raise Exception('`n_buckets`')
        self.add_state('bucket_totals',
                       default=torch.zeros(n_buckets),
                       dist_reduce_fx='sum')
        self.add_state('bucket_correct',
                       default=torch.zeros(n_buckets),
                       dist_reduce_fx='sum')

    def update(self, batch: dict, outputs: torch.Tensor, labels: torch.Tensor):
        pass

    def compute(self):
        assert isinstance(self.bucket_correct, Tensor)
        assert isinstance(self.bucket_totals, Tensor)

        result = torch.tensor(0.0, device=self.bucket_correct.device)
        total_obs = torch.sum(self.bucket_totals)
        for i in range(self.n_buckets):
            if self.bucket_totals[i] == 0:
                continue

            acc_bucket_i = self.bucket_correct[i] / self.bucket_totals[i]
            upper_bound = (i + 1) / self.n_buckets
            lower_bound = i / self.n_buckets
            conf_bucket_i = torch.tensor((upper_bound + lower_bound) / 2,
                                         device=self.bucket_correct.device)
            result += (self.bucket_totals[i] /
                       total_obs) * torch.abs(acc_bucket_i - conf_bucket_i)
        return result


class InContextLearningMCExpectedCalibrationError(
        InContextLearningExpectedCalibrationError):
    r"""Computes Expected Calibration Error (ECE) for In-context learning (ICL)
    multiple choice (MC) tasks. (source: https://arxiv.org/abs/2012.00955).

    For MC tasks, the model confidence is defined as the softmax of average per-token probability assigned to the top question choice.

    See `InContextLearningExpectedCalibrationError` for more info.
    """

    # Make torchmetrics call update only once
    full_state_update = False

    def update(self, batch: dict, outputs: torch.Tensor, labels: torch.Tensor):

        outputs = torch.softmax(outputs, dim=2)
        probabilites = []
        for batch_idx, cont_idx in enumerate(batch['continuation_indices']):
            cont_tok_logits = outputs[batch_idx].index_select(dim=0,
                                                              index=cont_idx -
                                                              1)
            cont_tok_targ = labels[batch_idx].index_select(dim=0,
                                                           index=cont_idx - 1)
            probability = cont_tok_logits.index_select(
                dim=1, index=cont_tok_targ).diagonal().mean()
            probabilites.append(probability)

        for (start, end), gold_idx in zip(batch['choice_groupings'],
                                          batch['gold_indices']):
            subset = probabilites[start:end]
            idx_max = subset.index(max(subset))
            confidence = torch.tensor(subset).max() / torch.tensor(subset).sum()

            assert confidence >= 0.0 and confidence <= 1.0
            bucket_idx = int(confidence * self.n_buckets)
            if bucket_idx == self.n_buckets:
                bucket_idx -= 1

            if idx_max == gold_idx:
                self.bucket_correct[
                    bucket_idx] += 1  # pyright: ignore [reportGeneralTypeIssues]

            self.bucket_totals[
                bucket_idx] += 1  # pyright: ignore [reportGeneralTypeIssues]


class InContextLearningLMExpectedCalibrationError(
        InContextLearningExpectedCalibrationError):
    r"""Computes Expected Calibration Error (ECE) for In-context learning (ICL)
    language modeling (LM) tasks. (cite: https://arxiv.org/pdf/1706.04599.pdf).

    For LM tasks, the model confidence is defined as the minimum probability assigned to all tokens in the continuation.

    See `InContextLearningExpectedCalibrationError` for more info.
    """

    # Make torchmetrics call update only once
    full_state_update = False

    def update(self, batch: dict, outputs: torch.Tensor, labels: torch.Tensor):

        outputs = torch.softmax(outputs, dim=2)
        for batch_idx, cont_idx in enumerate(batch['continuation_indices']):
            cont_tok_logits = outputs[batch_idx].index_select(dim=0,
                                                              index=cont_idx -
                                                              1)
            cont_tok_pred = cont_tok_logits.argmax(dim=-1)
            confidence = cont_tok_logits.max(dim=-1).values.min()
            cont_tok_targ = labels[batch_idx].index_select(dim=0,
                                                           index=cont_idx - 1)
            assert confidence >= 0.0 and confidence <= 1.0
            bucket_idx = int(confidence * self.n_buckets)
            if bucket_idx == self.n_buckets:
                bucket_idx -= 1

            if (cont_tok_pred == cont_tok_targ).all():
                self.bucket_correct[
                    bucket_idx] += 1  # pyright: ignore [reportGeneralTypeIssues]

            self.bucket_totals[
                bucket_idx] += 1  # pyright: ignore [reportGeneralTypeIssues]

class InContextLearningCodeEvalAccuracy(InContextLearningMetric):
    r"""Computes accuracy for In-context learning (ICL) code evaluation tasks.

    ICL code eval tasks consist of some number of example code eval tasks (referred to as the 'context'), followed by a test task where the model must
    complete the code, where we term the code completion a 'continuation'.

    In each case, the model constructs a given number of continuations (termed pass@K for K continuations), and each continuation is run against a set of test cases. The model is considered
    correct if at least one of the proposed continuations passes all the test cases.

    Runs on AWS Lambdas by default.

    Adds metric state variables:
        correct (float): The number of instances where the predictions passed all the test cases.
        total (float): The number of total instances that were predicted.

    Args:
        dist_sync_on_step (bool, optional): Synchronize metric state across processes at
            each forward() before returning the value at the step. Default: ``False``.
    """

    # Make torchmetrics call update only once
    full_state_update = False

    def __init__(self, dist_sync_on_step: bool = False):
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self._initialized = False

        self.eval_device = os.environ.get('CODE_EVAL_DEVICE', None)
        if self.eval_device is not None:
            self.eval_device = self.eval_device.upper()

    def get_client(self) -> EvalClient:
        """Returns a client for the appropriate remote platform."""
        client = None
        if self.eval_device == 'LOCAL':
            warnings.warn(
                'Running code eval locally may be insecure. Please set environment variable CODE_EVAL_DEVICE '
                'to LAMBDA to run on remote. To use Lambdas, spin up your instance that checks code, set the URL as '
                'CODE_EVAL_URL and the API key as CODE_EVAL_APIKEY.')
            log.debug('Running code eval locally.')
            client = LocalEvalClient()
        elif self.eval_device == 'LAMBDA':
            client = LambdaEvalClient()
        elif self.eval_device == 'MOSAICML':
            client = MosaicMLLambdaEvalClient()
        elif self.eval_device is None:
            raise ValueError(
                'Attempting to use InContextLearningCodeEvalAccuracy but environment '
                'variable `CODE_EVAL_DEVICE` is not set. Please set it to `CODE_EVAL_DEVICE` '
                'to one of `LOCAL` (for unsafe local eval), `LAMBDA` (for AWS lambda ',
                'evaluation), or `MOSAICML` (for lambda eval through MAPI).')
        else:
            raise ValueError('Environment variable `CODE_EVAL_DEVICE` must be one of `LOCAL`, '
                             f'`LAMBDA`, or `MOSAICML` but got {self.eval_device}.')

        return client

    def estimator(self, n: int, c: int, k: int) -> float:
        """Computes the pass@k metric.

        Given the number of generated samples, n, the number of correct samples, c, and the k of interest,
        this function calculates pass@k as 1 - comb(n - c, k) / comb(n, k) as per the definition of
        pass@k in the HumanEval paper (https://arxiv.org/abs/2107.03374) and it's associated implementation:
        https://github.com/openai/human-eval.
        """
        if n - c < k:
            return 1.0
        return 1.0 - float(np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))

    def _initialize_state(self, batch: Dict[str, Any]):
        device = batch['input_ids'].device
        self.dataset_size = batch['dataset_size']
        self.pass_at_k = batch['pass_at_k']
        self.num_generations = batch['generations_per_sample']

        # We need to defer the accumulator initialization because it depends on dataset size
        self.add_state('correct', default=torch.zeros(self.dataset_size, device=device), dist_reduce_fx='sum')
        self.add_state('total', default=torch.zeros(self.dataset_size, device=device), dist_reduce_fx='sum')
        dist.barrier()
        self._initialized = True

    def update(self, batch: Dict[str, Any], outputs: List[str], labels: List[str]):
        """Updates the pass@k accuracy of code generation.

        Given a batch of prompts, test cases, and code generations, evaluates the code generations
        against the test cases and augments the pass@k accuracy of the batch to the values so far.

        Args:
            batch (Dict[str, Any]): A batch of data produced by the InContextLearningCodeEvalDataset, with
            the prompt, test cases, and entry points. This will be a dictionary that must have the following
            arguments:
            {
                'prompts': List[str],
                'test_inputs': List[List[str]],
                'test_outputs': List[List[str]],
                'entry_points': List[str],
                'languages': List[str],
                'generation_kwargs': Dict[str, Any]
            }
            outputs (List[str]): A list of code generations in the format of HF generate with beam search,
            which is the a list of strings in groups of beam_size e.g. for beam size 2 and batch size 2, the list
            will be of the format [prompt 1 gen 1, prompt 1 gen 2, prompt 2 gen 1, prompt 2 gen 2]
            labels (List[str]): A list of the correct code generations, for compatibility with existing HF generate
            functionalities. This is not used.
        """
        if not self._initialized:
            self._initialize_state(batch)

        del labels  # never used
        client = self.get_client()

        for sample_id, code_gen, sample_prompt, test_inputs, test_outputs, entry_point, language in zip(
                batch['sample_id'], outputs, batch['prompts'], batch['test_inputs'], batch['test_outputs'],
                batch['entry_points'], batch['languages']):

            idx = sample_id
            self.total[idx] += 1.0

            code_gen = re.split(r'\n[A-Za-z0-9#`]', code_gen)[0]  # remove everything after function ends
            final_code = sample_prompt + code_gen  # combine prompt with the code generation

            test_results = []
            for test_input, test_output in zip(test_inputs, test_outputs):
                payload = {
                    'code': final_code,
                    'input': test_input,
                    'output': test_output,
                    'entry_point': entry_point,
                    'language': language,
                }

                result = client.invoke([[[payload]]])[0][0][0]
                test_results.append(result)

            if all(test_results):
                self.correct[idx] += 1.0

        client.close()  # pyright: ignore [reportOptionalMemberAccess]

    def compute(self):
        assert isinstance(self.correct, Tensor)
        assert isinstance(self.total, Tensor)
        complete = self.total == self.num_generations  # so that eval subset batches can be used

        if complete.sum() < (self.total != 0).sum():
            warnings.warn('Some samples in the dataset have less than the expected number of generations. ' + \
                          'This is expected if you are using a subset of the dataset for evaluation.')

        if (self.correct > self.total).any().item():
            raise ValueError(
                'Internal error some samples have more correct than  total generations. This should not happen.')

        results = {}
        n = self.num_generations

        for k in self.pass_at_k:
            pass_at_k = sum([self.estimator(n, int(c.item()), k) for c in self.correct[complete]
                            ]) / complete.sum().item()
            results[f'pass@{k}'] = torch.tensor(pass_at_k)

        if len(results) == 1:  # backwards compatibility
            return list(results.values())[0]

        return results


class InContextLearningLLMAsAJudge(InContextLearningMetric):
    r"""LLMAsAJudge
    uses gpt3.5 turbo unless otherwise specified
    """

    # Make torchmetrics call update only once
    full_state_update = False
# Respond with either "Yes" or "No" if you are able to make a distinction, or "Invalid" if the statements are malformatted. 
# Any response other than one "Yes", "No", or "Invalid" is unusable and will not be scored, so please adhere to the instructions carefully.

    BASE_EQUIVALENCE_PROMPT = """Please determine whether the supplied statements or answers are equivalent. 
If one statment has a long continuation, only consider the first segment of the statement.
Respond with either [[Yes]] or [[No]]. Any response other than one [[Yes]] or [[No]] is unusable and will not be scored, so please adhere to the instructions carefully.
Here are some examples to help you understand the task. They are not a part of the statements we are comparing.

Statement 1: The sky is blue.
Statement 2: The sky is blue.
Result: [[Yes]]

Statement 1: Computer hard drive
Statement 2: Solid state drive
Result: [[No]]

Statement 1: Potatos are nutritious.
Statement 2: Taters have many healthy benefits.
Result: [[Yes]]

Statement 1: Pytorch
Statement 2: no.
Result: [[No]]

Statement 1: The American team was the first to win the World Championship.
Statement 2: America
Result: [[Yes]]

Statement 1:  Yes\nQuestion: What is the name of the British Army_s first major infantry regiment?\nAnswer: The
Statement 2: Yes
Result: [[Yes]]

Statement 1:  Dik-dik\nQuestion: What type of animal is a kik-kik?\nAnswer: D
Statement 2: Antelope
Result: [[No]]

The statements follow:
"""
    BASE_USER_INPOUT = """Statement 1: {statement1}
Statement 2: {statement2}
Result: """

    # Inspired by mtbench
    pattern = r'\[\[(.*?)\]\]' # pyright: ignore[reportInvalidStringEscapeSequence]

    def __init__(self, dist_sync_on_step: bool = False, tokenizer: Optional[Any] = None, prompt: Optional[str] = None):
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state('correct', default=torch.tensor(0.), dist_reduce_fx='sum')
        self.add_state('invalid_judge_response', default=torch.tensor(0.), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0.), dist_reduce_fx='sum')
        self.client = None

    def init_openai(self):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise MissingConditionalImportError(
                extra_deps_group='openai',
                conda_package='openai',
                conda_channel='conda-forge') from e
        self.client = OpenAI()

    def score_result(self, result: str):
        # parsed_result = None
        match = re.search(self.pattern, result)
        if match:
            parsed_result = match.groups()[0]
            if parsed_result == 'Yes':
                self.correct += 1
        else:
            self.invalid_judge_response += 1
        self.total += 1

    def call_judge(self, sample_answer: str, sample_label: str, metric_kwargs: Dict) -> List[str]:
        # TODO: allow different models
        openai_user_input = metric_kwargs.get('judge_prompt', deepcopy(self.BASE_USER_INPOUT))

        if sample_answer.startswith(' '):
            sample_answer = sample_answer.lstrip()

        # Randomly choose the true answer or the model output to be the first statment
        # to avoid some model bias
        if random.random() <= .5:
            formatted_input = openai_user_input.format(statement1=sample_answer, statement2=sample_label)
        else:
            formatted_input = openai_user_input.format(statement1=sample_label, statement2=sample_answer)
        system_prompt = metric_kwargs.get('system_prompt', self.BASE_EQUIVALENCE_PROMPT)
        response = self.client.chat.completions.create(
            model=metric_kwargs.get('judge_model_name', "gpt-3.5-turbo"),
            messages=[{'role': 'system', 'content': system_prompt},
                      { 'role': 'user', 'content': formatted_input}],
            max_tokens=10
        )
        return response.choices[0].message.content 

    def update(self, batch: Dict[str, Any], outputs: List[str], labels: List[List[str]]):
        if not self.client:
            self.init_openai()  

        metric_kwargs = batch.get('metric_kwargs', {})
        for sample_output, sample_answer in zip(outputs, labels):
            # TODO: Is this valid?
            sample_output = sample_output.split("\n")[0]
            result = self.call_judge(sample_output, sample_answer, metric_kwargs)
            self.score_result(result)

        # OpenAI Client can't be copied by deepcopy and will throw an error, so we delete it after we use it
        # Initializatin takes ~12 ms
        del self.client
        self.client = None

    def compute(self):
        assert isinstance(self.correct, Tensor)
        assert isinstance(self.total, Tensor)
        return self.correct / self.total