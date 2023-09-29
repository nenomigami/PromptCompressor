from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

import torch
import torch.nn as nn

from gym.spaces import Space
from gym.spaces.dict import Dict as DictSpace
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import Schedule, TensorDict
from transformers import AutoModelForMaskedLM
from pcrl.utils.warm_start import ActorCriticWarmStartMixin
from pcrl.algorithms.common.distributions import make_masked_proba_distribution

def labels_to_summary(input_batch, label_batch, tokenizer):
    summaries = []
    for input_ids, labels in zip(input_batch, label_batch):
        selected = [int(input_ids[i]) for i in range(len(input_ids))
                           if labels[i] == 1]
        summary = tokenizer.decode(selected, skip_special_tokens=True)
        summaries.append(summary)
    return summaries

class BatchTokenPolicy(BasePolicy, ActorCriticWarmStartMixin):
    def __init__(self, 
                 observation_space: DictSpace,
                 action_space: Space,
                 lr_schedule: Schedule,
                 model_name: str,
                 weight_decay: float = 1e-6,
                 optimizer_class = torch.optim.AdamW,
                 optimizer_kwargs = None,
                 state_dict: Dict[str, Any] = None,
                 **kwargs,
        ):
        super().__init__(observation_space, action_space)
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
        self._action_space = action_space
        self.action_dist = make_masked_proba_distribution(action_space)
        self._build_model_heads(model_name)
        self._setup_optimizer(optimizer_kwargs, weight_decay, optimizer_class)
        self.load_from_dict(state_dict)

    def _build_model_heads(self, model_name: str):
        self._base_model = AutoModelForMaskedLM.from_pretrained(model_name).to('cuda') #TODO before optimizer loading
        self._value_model = nn.Sequential(*[
            nn.Linear(self._base_model.config.hidden_size, 4096, bias=False),
            nn.GELU(),
            nn.Linear(4096, 1, bias=True)]).to('cuda')
        self._policy_model = nn.Sequential(*[
            nn.Linear(self._base_model.config.hidden_size, 4096, bias=False),
            nn.GELU(),
            nn.Linear(4096, 2, bias=True)]).to('cuda')

    def _setup_optimizer(self, optimizer_kwargs: Dict[str, Any],
                         weight_decay: float, optimizer_class: torch.optim):
        params = list(self.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in params if not any(
                nd in n for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [p for n, p in params if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        self.optimizer = optimizer_class(
            optimizer_grouped_parameters, **optimizer_kwargs)

    def forward(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
        action_masks: Optional[np.ndarray] = None,
    ):
        with torch.no_grad():
            output = self._base_model(obs['input_ids'], obs['attention_mask'], output_hidden_states=True)
        seq_length = obs['attention_mask'].sum(axis=-1)
        last_hidden = output["hidden_states"][-1]
        
        values = self._value_model(last_hidden[torch.arange(last_hidden.size(0)), -seq_length])
        logits = self._policy_model(last_hidden)
        
        assert torch.all(torch.isfinite(logits))
        distribution = self.action_dist.proba_distribution(logits)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob


    def evaluate_actions(self, obs: torch.Tensor,
                         actions: torch.Tensor,
                         action_masks: Optional[np.ndarray] = None,
        ):
        with torch.no_grad():
            output = self._base_model(obs['input_ids'].long(), output_hidden_states=True)
        seq_length = obs['attention_mask'].sum(axis=-1).long()
        last_hidden = output["hidden_states"][-1]
        
        values = self._value_model(last_hidden[torch.arange(last_hidden.size(0)), -seq_length])
        logits = self._policy_model(last_hidden)
        assert torch.all(torch.isfinite(logits))
        distribution = self.action_dist.proba_distribution(logits)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        log_prob = distribution.log_prob(actions)
        return values, log_prob, distribution.entropy()


    def _predict(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
        action_masks: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        with torch.no_grad():
            output = self._base_model(obs['input_ids'], obs['attention_mask'], output_hidden_states=True)
        last_hidden = output["hidden_states"][-1]
        
        logits = self._policy_model(last_hidden)
        assert torch.all(torch.isfinite(logits))
        distribution = self.action_dist.proba_distribution(logits)
        if action_masks is not None:
            distribution.apply_masking(action_masks)

        return distribution.get_actions(deterministic=deterministic)

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
        action_masks: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        self.set_training_mode(False)

        obs, _ = self.obs_to_tensor(observation)
        with torch.no_grad():
            actions = self._predict(
                obs, deterministic=deterministic, action_masks=action_masks)
            # Convert to numpy
            actions = actions.cpu().numpy()

        return actions, state


    def predict_values(self, obs: TensorDict):
        with torch.no_grad():
            output = self._base_model(obs['input_ids'], output_hidden_states=True)
        seq_length = obs['attention_mask'].sum(axis=-1)
        last_hidden = output["hidden_states"][-1]
        
        values = self._value_model(last_hidden[torch.arange(last_hidden.size(0)), -seq_length])
        return values