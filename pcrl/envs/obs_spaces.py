from abc import ABC, abstractmethod
from string import punctuation
from typing import Dict, List, Tuple, Optional, Any
from gym import spaces
import gym
import numpy as np
import nltk

class BaseObsSpace(ABC):
    @abstractmethod
    def get_obs_spec(
            self,
    ) -> gym.spaces.Dict:
        pass

    @abstractmethod
    def observation(self, observation) -> Dict[str, Any]:
        pass


class FixedTokenObservation(BaseObsSpace):
    
    def __init__(
            self,
            tokenizer,
            max_prompt_length,
            **kwargs
        ) -> None:
        self.tokenizer = tokenizer
        self._max_text_length = max_prompt_length
        self._vocab_size = self.tokenizer.vocab_size
    
    def get_obs_spec(self) -> gym.spaces.Dict:
        return gym.spaces.Dict(
            {
                "input_ids": spaces.Box(
                    low=0,
                    high=self._vocab_size,
                    shape=(self._max_text_length,),
                ),
                "attention_mask": spaces.Box(
                    low=0,
                    high=self._vocab_size,
                    shape=(self._max_text_length,),
                ),
            }
        )
    
    def observation(self, sample) -> Dict[str, Any]:
        kwargs = {
            "padding": "max_length",
            "max_length": self._max_text_length,
            "truncation": True
        }
        if isinstance(sample, list):
            prompts = [b.prompt_or_input_text for b in sample]
            output = self.tokenizer(prompts, **kwargs)
        else: 
            output = self.tokenizer(sample.prompt_or_input_text, **kwargs)
        
        observation = {
            "input_ids": np.array(output["input_ids"]),
            "attention_mask": np.array(output["attention_mask"]),
        }
        return observation 
    

class FixedTokenObservation(BaseObsSpace):
    
    def __init__(
            self,
            tokenizer,
            max_prompt_length,
            **kwargs
        ) -> None:
        self.tokenizer = tokenizer
        self._max_text_length = max_prompt_length
        self._vocab_size = self.tokenizer.vocab_size
        
    def get_obs_spec(self) -> gym.spaces.Dict:
        return gym.spaces.Dict(
            {
                "input_ids": spaces.Box(
                    low=0,
                    high=self._vocab_size,
                    shape=(self._max_text_length,),
                ),
                "attention_mask": spaces.Box(
                    low=0,
                    high=self._vocab_size,
                    shape=(self._max_text_length,),
                ),
                # "is_irrel": spaces.Box(
                #     low=0,
                #     high=1,
                #     shape=(self._max_text_length,),
                # )
                # "compress_args": spaces.Box(
                #     low=0,#high는 무한
                #     high=np.inf,
                #     shape=(1,),
                # )
            }
        )

    # def is_irrel_token(self, input_ids: List[int]):
    #     is_irrel = np.zeros(self._max_text_length, dtype=bool)
    #     for idx, token in enumerate(input_ids):
    #         if token in self.tokenizer.all_special_ids:
    #             continue
    #         words = self.tokenizer.decode(token).strip()
    #         if words in self._irrel_token:
    #             is_irrel[idx] = 1        
    #     return is_irrel

    def observation(self, sample) -> Dict[str, Any]:
        kwargs = {
            "padding": "max_length",
            "max_length": self._max_text_length,
            "truncation": True
        }
        if isinstance(sample, list):
            prompts = [b.prompt_or_input_text for b in sample]
            output = self.tokenizer(prompts, **kwargs)
            # is_irrel_array = np.stack([self.is_irrel_token(b) for b in output["input_ids"]])
        else: 
            output = self.tokenizer(sample.prompt_or_input_text, **kwargs)
            # is_irrel_array = self.is_irrel_token(output["input_ids"])

        observation = {
            "input_ids": np.array(output["input_ids"]),
            "attention_mask": np.array(output["attention_mask"]),
            # "is_irrel": is_irrel_array,
            #"compress_args": np.array([sample.stopword_ratio]), TODO 일단은 보류. reward 계산 할 때 info에서 끌어올 수 있으면 얘 없애기.
        }
        return observation 


class FixedWordObservation(BaseObsSpace):
    
    def __init__(self) -> None:
        super().__init__()
    
    def get_obs_spec(self) -> gym.spaces.Dict:
        return super().get_obs_spec()
