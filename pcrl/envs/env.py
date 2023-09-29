from typing import Dict, Tuple, Optional, List
from gym import Env
from transformers import AutoTokenizer
from data.instruction_pool import Sample
from pcrl.envs.obs_spaces import BaseObsSpace
from pcrl.envs.act_spaces import BaseActSpace
import copy
import numpy as np
import torch

class BatchPromptCompEnv(Env):

    def __init__(
        self,
        rank: int,
        samples: List[Sample],
        tokenizer: AutoTokenizer,
        act_space: BaseActSpace,
        obs_space: BaseObsSpace,
        max_episode_length: int = 1,
        max_prompt_length: Optional[int] = None,
        terminate_on_eos: bool = False,
        context_start_token: Optional[int] = None,
        prompt_truncation_side: str = "left",
        seed = None,
    ):
        if seed is not None:
            np.random.seed(seed)
        self.tokenizer = tokenizer  
        self.max_steps = max_episode_length 
        self._max_text_length = (
            max_prompt_length if max_prompt_length else tokenizer.model_max_length
        )
        self.sample = samples
        self._terminate_on_eos = terminate_on_eos
        self._context_start_token = context_start_token
        self._prompt_truncation_side = prompt_truncation_side
        super().__init__()

        # set the observation and action space here
        self._vocab_size = tokenizer.vocab_size
        self.obs_space = obs_space
        self.act_space = act_space
        self.observation_space = self.obs_space.get_obs_spec()
        self.action_space = self.act_space.get_act_spec()

        # check the tokenizer and add padding tokens
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"  
        self.tokenizer.truncation_side = "left" 

        # init task 
        self.task_idx = 0

        # init tracking variables
        self.current_sample = None
        self.current_obs = None
        self.time_step = None

        #for scst
        self.rank = rank

    def step(self, action: int) -> Tuple[Dict[str, torch.tensor], int, bool, dict]:
        done = False
        self.time_step += 1

        # just update the context tensor and gets the new observation
        self.current_obs = self.act_space.process_actions(self.current_obs, action)

        # decide if the episode is finished or not
        if self.time_step == self.max_steps:
            done = True                                                                           

        # compute reward
        reward = 0  # will be overridden later

        # populate additional info
        compressed_prompt = self.act_space.decode(self.current_obs)

        info = {
            "base_prompt": self.current_sample.prompt_or_input_text,
            "base_gen_texts": self.current_sample.generated_text,
            "base_token_counts": self.current_sample.input_token_counts,
            "compressed_prompt": compressed_prompt,
        }

        return self.current_obs, reward, done, info

    def fake_step(self, action: np.ndarray) -> Tuple[Dict[str, torch.tensor], int, bool, dict]:
        done = False
        action = action[self.rank]
        #self.time_step += 1
        
        #not update observation
        current_obs = copy.deepcopy(self.current_obs)
        current_obs = self.act_space.process_actions(current_obs, action)

        if self.time_step == self.max_steps:
            done = True                       

        reward = 0  # will be overridden later

        # populate additional info
        compressed_prompt = self.act_space.decode(current_obs)

        info = {
            "base_prompt": self.current_sample.prompt_or_input_text,
            "base_gen_texts": self.current_sample.generated_text,
            "base_token_counts": self.current_sample.input_token_counts,
            "compressed_prompt": compressed_prompt,
        }

        return current_obs, reward, done, info


    def reset(self) -> Dict[str, torch.tensor]:
        """
        Resets the environment and starts a new episode
        """
        # gets a new sample if not provided
        idx = self.task_idx
        self.current_sample = self.sample[idx]

        self.current_obs = self.obs_space.observation(self.current_sample)

        self.time_step = 0

        self.task_idx += 1
        self.task_idx %= len(self.sample)

        return self.current_obs
    
    def render(self):
        pass

    def close(self):
        pass

    def action_masks(self) -> np.ndarray:
        """
        Returns the action mask for the current observation
        """
        return self.act_space.action_mask(self.current_obs)
