from typing import List, Dict, Tuple, Any
from abc import abstractmethod
from datasets import load_metric
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedModel
from tqdm import tqdm
import copy
import evaluate
import numpy as np
import torch


class BaseMetric:
    @abstractmethod
    def compute(
        self,
        original_generated_texts: List[str],
        compressed_generated_texts: List[str],
        truncated_compressed_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ):
        """
        Returns a dict where key is the metric name and value is again a dict consisting of tuple of individual scores (if any) and corpus level score

        eg. {
            metric_name: (individual_scores, corpus_level_score)
            "metric_1": ([0.5, 0.5, 0.8], 0.1)
        }

        """
        raise NotImplementedError


class MeteorMetric(BaseMetric):
    def __init__(self) -> None:
        super().__init__()
        self._metric = load_metric("meteor")

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ):

        score = self._metric.compute(
            predictions=generated_texts, references=reference_texts
        )["meteor"]

        metric_dict = {"meteor": (None, score)}
        return metric_dict


class Rouge_C(BaseMetric):
    def __init__(self) -> None:
        super().__init__()
        self._metric = load_metric("rouge")

    def compute(
        self,
        original_generated_texts: List[str],
        compressed_generated_texts: List[str],
        truncated_compressed_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ):
        metric_results = self._metric.compute(
            predictions=compressed_generated_texts, references=original_generated_texts, use_stemmer=False
        )
        score_keys = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
        metric_dict = {}
        for rouge_type in score_keys:
            rouge_score = metric_results[rouge_type].mid.fmeasure
            metric_dict[f"rouge_c_{rouge_type}"] = rouge_score
        return metric_dict

class Rouge_R(BaseMetric):
    def __init__(self) -> None:
        super().__init__()
        self._metric = load_metric("rouge")

    def compute(
        self,
        original_generated_texts: List[str],
        compressed_generated_texts: List[str],
        truncated_compressed_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ):
        metric_results = self._metric.compute(
            predictions=truncated_compressed_texts, references=reference_texts, use_stemmer=False
        )
        score_keys = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
        metric_dict = {}
        for rouge_type in score_keys:
            rouge_score = metric_results[rouge_type].mid.fmeasure
            metric_dict[f"rouge_r_{rouge_type}"] = rouge_score
        return metric_dict