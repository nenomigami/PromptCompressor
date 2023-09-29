from typing import Any, Dict, List
from transformers import AutoTokenizer
from data.instruction_pool import Sample
from pcrl.envs import make_vec_env
from pcrl.envs.env import BatchPromptCompEnv
from pcrl.utils.evaluation_utils import evaluate_on_samples
from pcrl.utils.logging_utils import Tracker
from pcrl.utils.metric import BaseMetric
from pcrl.envs.reward import RewardFunction
from pcrl.utils.registry import (DataPoolRegistry, 
                                 ObsRegistry,
                                 ActRegistry,
                                 MetricRegistry,
                                 RewardFunctionRegistry,
                                 PolicyRegistry,
                                 AlgorithmRegistry,
                                 WrapperRegistry)
from pcrl.utils.warm_start import TrainerWarmStartMixin
import gym
import time

def build_tokenizer(tokenizer_config: Dict[str, Any], **kwargs):
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_config["model_name"], **kwargs)
    if tokenizer.pad_token is None and tokenizer_config.get("pad_token_as_eos_token", True):
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = tokenizer_config.get(
        "padding_side", "left")
    tokenizer.truncation_side = tokenizer_config.get(
        "truncation_side", "left")
    return tokenizer


def build_reward_fn(reward_config: Dict[str, Any]):
    reward_fn = RewardFunctionRegistry.get(reward_config["id"],
                                           reward_config.get("args", {}))
    return reward_fn


def build_metrics(metric_configs: List[Dict[str, Any]]):
    metrics = [MetricRegistry.get(metric_config["id"], metric_config.get("args", {}))
               for metric_config in metric_configs]
    return metrics


def build_datapool(gen_config: Dict[str, Any],
                   datapool_config: Dict[str, Any]):
    tokenizer = build_tokenizer(gen_config['tokenizer'], use_fast=False)

    def _get_datapool_by_split(split: str):
        kwargs = datapool_config.get("args", {})
        kwargs["split"] = split
        kwargs["tokenizer"] = tokenizer
        kwargs["gen_config"] = gen_config
        dp_split = DataPoolRegistry.get(datapool_config["id"], kwargs)
        return dp_split

    train_datapool = _get_datapool_by_split("train")
    val_seen_datapool = _get_datapool_by_split("validation_seen")
    val_unseen_datapool = _get_datapool_by_split("validation_unseen")
    val_human_datapool = _get_datapool_by_split("validation_human")

    samples_by_split = {
        "train": [sample for sample, _ in train_datapool],
        "val_seen": [sample for sample, _ in val_seen_datapool],
        "val_unseen": [sample for sample, _ in val_unseen_datapool],
        "val_human": [sample for sample, _ in val_human_datapool],
    }
    return samples_by_split


def build_env(env_config: Dict[str, Any],
              tokenizer: AutoTokenizer,
              samples: List[Sample]=[],
              sample_k: int=None):
    env_kwargs = {
        "tokenizer": tokenizer,
        **env_config.get("args", {})
    }
    env_kwargs["obs_space"] = ObsRegistry.get(env_config.get("obs_space"), env_kwargs)
    env_kwargs["act_space"] = ActRegistry.get(env_config.get("act_space"), env_kwargs)
    env = make_vec_env(BatchPromptCompEnv,
                       n_envs=env_config.get("n_envs", 1),
                       samples=samples,
                       sample_k=sample_k,
                       env_kwargs=env_kwargs)
    return env


def build_alg(alg_config: Dict[str, Any],
              gen_config: Dict[str, Any],
              env: gym.Env,
              reward_fn: RewardFunction,
              metrics: List[BaseMetric],
              tasks: Dict[str, Any],
              tracker: Tracker,
              policy_state: Dict[str, Any]):
    policy_config = alg_config["policy"]
    policy_cls = PolicyRegistry.get(policy_config["id"])
    alg_cls = AlgorithmRegistry.get(alg_config["id"])
    val_tasks = {k:v for k,v in tasks.items() if "val" in k}

    policy_args = policy_config["args"]
    policy_args["state_dict"] = policy_state

    alg_kwargs = {
        "policy": policy_cls,
        "env": env,
        "policy_kwargs": policy_args,
    }
    alg_kwargs = {**alg_kwargs, **alg_config.get("args")}
    wrapper = WrapperRegistry.get(alg_config["id"])
    alg = wrapper(alg_cls, alg_kwargs, gen_config, 
                  reward_fn, metrics, val_tasks, tracker)
    return alg


class OnPolicyTrainer(TrainerWarmStartMixin):
    """
    A generic trainer for training LMs with onpolicy algorithms from SB3
    """

    def __init__(self,
                 gen_config: Dict[str, Any],
                 datapool_config: Dict[str, Any],
                 reward_config: Dict[str, Any],
                 env_config: Dict[str, Any],
                 alg_config: Dict[str, Any],
                 train_eval_config: Dict[str, Any],
                 tracker: Tracker = None,
                 experiment_name: str = ''
                 ):
        self._gen_config = gen_config
        self._datapool_config = datapool_config
        self._reward_config = reward_config
        self._env_config = env_config
        self._alg_config = alg_config
        self._train_eval_config = train_eval_config
        self._tracker = tracker
        self._experiment_name = experiment_name
        self._setup()

    def _setup(self):
        # load trainer state from available previous checkpoint if available
        self.load_trainer_state(self._tracker)

        # build components
        self._reward_fn = build_reward_fn(self._reward_config)
        self._metrics = build_metrics(self._train_eval_config.get("metrics", []))
        self._tasks = build_datapool(self._gen_config, self._datapool_config)
        self._tokenizer = build_tokenizer(self._alg_config["policy"]["args"])
        self._env = build_env(self._env_config, self._tokenizer, 
                              self._tasks["train"], self._alg_config['args']['sample_k'])
        
        self._alg = build_alg(self._alg_config, self._gen_config, self._env,
                              self._reward_fn, self._metrics, self._tasks,
                              self._tracker, self._policy_state_dict)
        # extract train params
        self._max_episode_length = self._env_config["args"]["max_episode_length"]
        self._max_prompt_length = self._env_config["args"]["max_prompt_length"]
        self._train_batch_size = self._alg_config["args"]["batch_size"]
        self._eval_batch_size = self._train_eval_config["eval_batch_size"]
        self._n_iters = int(self._train_eval_config["n_iters"])
        self._sample_k = self._alg_config['args']['sample_k']
        self._n_steps_per_iter = 10#len(self._tasks["train"])/3 #TODO remove the comment symbol

         # gen kwargs for evaluation (if it is different from training gen kwargs)
        self._eval_gen_kwargs = self._train_eval_config.get(
            "generation_kwargs", None)
    def _evaluate_on_datapools(self, epoch: int,
                               splits: List[str] = ["val_seen", "val_unseen", "val_human"]):
        for split in splits:
            evaluate_on_samples(self._alg.gen_model,
                                self._alg.gen_tokenizer,
                                policy=self._alg.policy,
                                env=self._env,
                                samples=self._tasks[split],
                                batch_size=self._eval_batch_size,
                                max_prompt_length=self._max_prompt_length,
                                metrics=self._metrics,
                                epoch=epoch,
                                split_name=split,
                                tracker=self._tracker,
                                gen_kwargs=self._eval_gen_kwargs)
        time.sleep(10)

    def train_and_eval(self):
        # evaluate on val and test set before fine-tuning once
        iter_start = self._trainer_state["current_iter"]
        # self._evaluate_on_datapools(epoch=iter_start)

        # train for given number of iters
        for epoch in range(iter_start, self._n_iters):
            # current state
            self._trainer_state["current_iter"] = epoch

            # inner rollout and learn loop for on-policy algorithm
            self._alg.learn(
                self._n_steps_per_iter,
                eval_freq=self._train_eval_config["eval_freq"], 
            )

            # save the policy checkpoint
            if (epoch + 1) % self._train_eval_config["save_every"] == 0:
                self.save_trainer_state(
                    self._tracker, self._alg.policy, self._trainer_state
                )
                # evaluate on val and test samples
                self._evaluate_on_datapools(epoch=epoch)