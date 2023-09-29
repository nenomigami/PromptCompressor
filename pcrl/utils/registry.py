from typing import Any, Dict, Type, Union

from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from data.instruction_pool import DataPool, AlpacaPlus, GPTeacher
from pcrl.algorithms import MaskablePG, MaskableA2C, MaskablePPO
from pcrl.utils.alg_wrappers import wrap_onpolicy_alg
from pcrl.utils.metric import (
    BaseMetric,
    Rouge_C,
    Rouge_R,
)
from pcrl.envs.act_spaces import BaseActSpace, BatchFixedTokenAction
from pcrl.envs.obs_spaces import BaseObsSpace, FixedTokenObservation, FixedWordObservation
from pcrl.envs.reward import RewardFunction, MeteorRewardFunction, RougeRewardFunction, CombineRewardFunction
from pcrl.model.policy import BatchTokenPolicy


class DataPoolRegistry:
    _registry = {
        "alpaca_plus": AlpacaPlus,
        "gpteacher": GPTeacher,
    }
    @classmethod
    def get(cls, datapool_id: str, kwargs: Dict[str, Any]):
        datapool_cls = cls._registry[datapool_id]
        datapool = datapool_cls.prepare(**kwargs)
        return datapool

    @classmethod
    def add(cls, id: str, datapool_cls: Type[DataPool]):
        DataPoolRegistry._registry[id] = datapool_cls


class ObsRegistry:
    _registry = {
        "FixedTokenObservation": FixedTokenObservation,
        "FixedWordObservation": FixedWordObservation,
    }

    @classmethod
    def get(cls, obs_space_id: str, kwargs: Dict[str, Any]) -> BaseObsSpace:
        obs_space_cls = cls._registry[obs_space_id]
        obs_space = obs_space_cls(**kwargs)
        return obs_space

    @classmethod
    def add(cls, id: str, obs_space_id: Type[BaseObsSpace]):
        ObsRegistry._registry[id] = obs_space_id

class ActRegistry:
    _registry = {
        "BatchFixedTokenAction" : BatchFixedTokenAction
    }

    @classmethod
    def get(cls, act_space_id: str, kwargs: Dict[str, Any]) -> BaseActSpace:
        act_space_cls = cls._registry[act_space_id]
        act_space = act_space_cls(**kwargs)
        return act_space

    @classmethod
    def add(cls, id: str, act_space_cls: Type[BaseActSpace]):
        ActRegistry._registry[id] = act_space_cls


class RewardFunctionRegistry:
    _registry = {
        "meteor": MeteorRewardFunction,
        "rouge": RougeRewardFunction,
        "combine": CombineRewardFunction,
    }

    @classmethod
    def get(cls, reward_fn_id: str, kwargs: Dict[str, Any]) -> RewardFunction:
        reward_cls = cls._registry[reward_fn_id]
        reward_fn = reward_cls(**kwargs)
        return reward_fn

    @classmethod
    def add(cls, id: str, reward_fn_cls: Type[RewardFunction]):
        RewardFunctionRegistry._registry[id] = reward_fn_cls


class MetricRegistry:
    _registry = {
        "rouge_c": Rouge_C,
        "rouge_r": Rouge_R,
    }

    @classmethod
    def get(cls, metric_id: str, kwargs: Dict[str, Any]) -> BaseMetric:
        metric_cls = cls._registry[metric_id]
        metric = metric_cls(**kwargs)
        return metric

    @classmethod
    def add(cls, id: str, metric_cls: Type[BaseMetric]):
        MetricRegistry._registry[id] = metric_cls


class PolicyRegistry:
    _registry = {
        "BatchTokenPolicy": BatchTokenPolicy,
    }

    @classmethod
    def get(cls, policy_id: str) -> Type[BasePolicy]:
        policy_cls = cls._registry[policy_id]
        return policy_cls

    @classmethod
    def add(cls, id: str, policy_cls: Type[BasePolicy]):
        PolicyRegistry._registry[id] = policy_cls


class AlgorithmRegistry:
    _registry = {
        "a2c_mask": MaskableA2C,
        "pg_mask": MaskablePG,
        "ppo_mask": MaskablePPO,
    }

    @classmethod
    def get(
        cls, alg_id: str
    ) -> Union[Type[OnPolicyAlgorithm], Type[OffPolicyAlgorithm]]:
        try:
            alg_cls = cls._registry[alg_id]
        except KeyError:
            raise NotImplementedError
        return alg_cls

    @classmethod
    def add(
        cls, id: str, alg_cls: Union[Type[OnPolicyAlgorithm], Type[OffPolicyAlgorithm]]
    ):
        AlgorithmRegistry._registry[id] = alg_cls


class WrapperRegistry:
    _registry = {
        "a2c_mask": wrap_onpolicy_alg,
        "ppo_mask": wrap_onpolicy_alg,
        "pg_mask": wrap_onpolicy_alg,
    }

    @classmethod
    def get(cls, alg_id: str):
        try:
            wrapper_def = cls._registry[alg_id]
        except KeyError:
            raise NotImplementedError
        return wrapper_def

    @classmethod
    def add(cls, id: str, wrapper_def):
        WrapperRegistry._registry[id] = wrapper_def