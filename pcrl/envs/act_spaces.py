from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional, Any
from data.instruction_pool import FIXED_TOKENS
import copy
import numpy as np
import gym

class BaseActSpace(ABC):
    @abstractmethod
    def get_act_spec(self, ) -> gym.spaces:
        pass

class BatchFixedTokenAction(BaseActSpace):
    def __init__(
            self,
            max_prompt_length,
            tokenizer,
            **kwargs
        ) -> None:
        self.max_prompt_length = max_prompt_length
        self.tokenizer = tokenizer
        self.special_tokens = tokenizer.all_special_ids
        self.n_command = 1
        self.prefixes = None

    def get_act_spec(self) -> gym.spaces:
        return gym.spaces.MultiDiscrete([self.n_command+1] * self.max_prompt_length)
    
    def process_actions(self, dict_obs: Dict, action: np.array):
        #command = action[0]
        dict_obs = copy.deepcopy(dict_obs)
        assert action.shape == dict_obs['input_ids'].shape

        # action 1 : remove the selected tokens
        input_ids = self.remove_tokens(dict_obs['input_ids'], action)
        unpadded_input_ids = self.remove_pad(input_ids)

        dict_obs["input_ids"] = self.pad_sequence(input_ids)
        dict_obs["attention_mask"] = self.pad_sequence(np.array([1]* len(unpadded_input_ids)), pad = 0)
        return dict_obs

    def pad_sequence(self, inputs: np.array, pad = None):
        pad = self.tokenizer.pad_token_id if pad is None else pad
        padding = np.array([pad] * (self.max_prompt_length-len(inputs)))
        return np.concatenate([padding, inputs])

    def remove_tokens(self, inputs: np.array, action: np.array):
        return inputs[action!=1]

    def remove_pad(self, inputs: np.array):
        return inputs[inputs!=self.tokenizer.pad_token_id]

    def action_mask(self, obs: Dict):
        if self.prefixes is None:
            self.prefixes = [self.tokenizer(prefix)['input_ids'][1:-1] for prefix in ["Instruction:", "Instruction: ", 
                                                                                      "\nInput: ", "\nInput:", "Input: ","Input:",
                                                                                      "\nOutput: \n", "Output:", "Output: "]]
        assert obs['input_ids'].ndim == 1
        _action_mask = np.zeros((self.n_command+1, self.max_prompt_length), dtype=bool)
        
        #[0] do nothing is always True
        _action_mask[0] = True

        #[1] remove the token

        #can't remove special tokens
        _action_mask[1] = ~np.isin(obs['input_ids'], self.special_tokens)
        
        #can't remove prefixes
        for i in range(len(obs['input_ids'])):
            for prefix in self.prefixes:
                if (obs['input_ids'][i:i+len(prefix)].tolist() == prefix):
                    _action_mask[1, i:i+len(prefix)] = False

        _action_mask = _action_mask.T
        return _action_mask

    def decode(self, obs: Dict):
        return self.tokenizer.decode(obs["input_ids"].astype(int), skip_special_tokens=True)