from typing import Any, Dict, List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from stable_baselines3.common.callbacks import EventCallback, BaseCallback
from pcrl.utils.evaluation_utils import get_batch, generate, list_in_dict_map
from pcrl.utils.logging_utils import Tracker
from pcrl.utils.metric import BaseMetric
import os
import numpy as np




class MaskableEvalCallback(EventCallback):

    def __init__(self,         
                 gen_model: AutoModelForCausalLM,
                 gen_tokenizer: AutoTokenizer,
                 gen_kwargs: Dict[str, List],         
                 eval_dataset: Dict[str, List],
                 metrics: List[BaseMetric],
                 tracker: Tracker,
                 callback_on_new_best: Optional[BaseCallback] = None,
                 callback_after_eval: Optional[BaseCallback] = None,
                 n_eval_episodes: int = 5,
                 eval_freq: int = 10000,
                 log_path: Optional[str] = None,
                 best_model_save_path: Optional[str] = None,
                 deterministic: bool = True,
                 render: bool = False,
                 verbose: int = 1,
                 warn: bool = True,
                 use_masking: bool = True):
        super().__init__(callback_after_eval, verbose=verbose)

        self.gen_model = gen_model
        self.gen_tokenizer = gen_tokenizer
        self.gen_kwargs = gen_kwargs
        self.eval_dataset = eval_dataset
        self.use_masking = use_masking
        self.metrics = metrics

        self.callback_on_new_best = callback_on_new_best
        if self.callback_on_new_best is not None:
            # Give access to the parent
            self.callback_on_new_best.parent = self

        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        # self.best_mean_reward = -np.inf
        # self.last_mean_reward = -np.inf
        self.deterministic = deterministic
        self.render = render
        self.warn = warn

        self.best_model_save_path = best_model_save_path
        # Logs will be written in ``evaluations.npz``
        if log_path is not None:
            log_path = os.path.join(log_path, "evaluations")
        self.log_path = log_path
        self.evaluations_results = []
        self.evaluations_timesteps = []
        self.evaluations_length = []
        # For computing success rate
        self._is_success_buffer = []
        self.evaluations_successes = []

    def _init_callback(self) -> None:

        # Create folders if needed
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)
        if self.log_path is not None:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

        # Init callback called on new best model
        if self.callback_on_new_best is not None:
            self.callback_on_new_best.init_callback(self.model)

    def _log_success_callback(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        """
        Callback passed to the  ``evaluate_policy`` function
        in order to log the success rate (when applicable),
        for instance when using HER.

        :param locals_:
        :param globals_:
        """
        info = locals_["info"]

        if locals_["done"]:
            maybe_is_success = info.get("is_success")
            if maybe_is_success is not None:
                self._is_success_buffer.append(maybe_is_success)

    def _on_step(self) -> bool:
        #deprecate
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            if self.callback is not None:
                return self._on_event()
            return True
        return True
    
    def update_child_locals(self, locals_: Dict[str, Any]) -> None:
        """
        Update the references to the local variables.

        :param locals_: the local variables during rollout collection
        """
        if self.callback:
            self.callback.update_locals(locals_)
