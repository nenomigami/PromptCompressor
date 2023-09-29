from abc import ABC, abstractclassmethod
from typing import List, Dict, Any
from data.instruction_pool import FIXED_TOKENS
import evaluate
import numpy as np

def extract_values(dictionaries: Dict, key):
    result_list = []
    for dict in dictionaries:
        if key in dict:
            result_list.append(dict[key])
    return result_list

class RewardFunction(ABC):
    @abstractclassmethod
    def __call__(
        self,
        infos: List[Dict[str, Any]],
    ) -> float:
        raise NotImplementedError

class RewardFunction(ABC):
    @abstractclassmethod
    def __call__(
        self,
        generated_texts: List[str],
        reference_texts: List[str],
    ) -> float:
        raise NotImplementedError

class MeteorRewardFunction(RewardFunction):
    def __init__(self, coef: float) -> None:
        super().__init__()
        self._metric = evaluate.load('meteor', keep_in_memory=True)
        self._coef = coef

    def __call__(
        self,
        generated_texts: List[str],
        reference_texts: List[str],
    ) -> float:
        assert len(generated_texts) == len(reference_texts)
        scores = []
        for gen, ref in zip(generated_texts, reference_texts):
            score = self._metric.compute(predictions=[gen], references=[ref])['meteor']
            scores.append(score)
        return self._coef * np.array(scores)


class RougeRewardFunction(RewardFunction):
    def __init__(
        self, rouge_type: str, coef: float
    ) -> None:
        super().__init__()
        self._metric = evaluate.load("rouge", keep_in_memory=True)
        self._rouge_type = rouge_type
        self._coef = coef

    def __call__(
        self,
        generated_texts: List[str],
        reference_texts: List[str],
    ) -> float:
        assert len(generated_texts) == len(reference_texts)
        score = self._metric.compute(predictions=generated_texts, references=reference_texts, use_aggregator=False)
        return self._coef * np.array(score[self._rouge_type])


class CombineRewardFunction(RewardFunction):
    def __init__(
        self,
        lamb: float,
        threshold: float,
        rouge_type: str,
    ) -> None:
        super().__init__()
        self._metric = evaluate.load("rouge", keep_in_memory=True)
        self._lambda = lamb
        self._threshold = threshold
        self._rouge_type = rouge_type

    def get_compress_ratio(self, infos: List[Dict[str, Any]], compressed_counts: List[int], fixed_counts: Dict[str, Any]):
        base_counts = extract_values(infos, 'base_token_counts')
        fixed_n_tokens = np.zeros(len(infos), dtype=np.int16)
        for i, info in enumerate(infos):
            for k, v in FIXED_TOKENS.items():
                if v in info["base_prompt"]:
                    fixed_n_tokens[i] += len(fixed_counts[k])
        compress_rate = 1 - (np.array(compressed_counts) - fixed_n_tokens) / (np.array(base_counts) - fixed_n_tokens)
        return compress_rate

    def __call__(
        self,
        infos: List[Dict[str, Any]],
        gen_output: Dict[str, Any],
        fixed_tokens_dict: Dict[str, Any]) -> float:
        similarity = np.array(self._metric.compute(predictions=extract_values(infos, "base_gen_texts"),
                        references=gen_output["gen_texts"], use_aggregator=False)[self._rouge_type])
        compress_rate = self.get_compress_ratio(infos, gen_output["compressed_token_counts"], fixed_tokens_dict)
        rw_info = {'sim': similarity, 'comp': compress_rate}
        rewards = np.where(rw_info['sim'] > self._threshold, rw_info['comp'], self._lambda)
        return rewards, rw_info
