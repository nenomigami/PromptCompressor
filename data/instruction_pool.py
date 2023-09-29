from abc import abstractclassmethod
from typing import Any, List, Dict
from dataclasses import dataclass
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from tqdm import tqdm
import math
import nltk
import random
import torch

FIXED_TOKENS = {
    "instruction": "Instruction: ",
    "input": "\nInput: ",
    "output": "\nOutput: \n",
}

def get_fixed_token_counts(tokenizer):
    token_dict = {}
    for k, v in FIXED_TOKENS.items():
        input_ids = tokenizer(v.rstrip(" "))['input_ids']
        input_ids = [id for id in input_ids if id not in tokenizer.all_special_ids]
        token_dict[k] = input_ids
    return token_dict

@dataclass(init=True)
class Sample:
    id: str
    prompt_or_input_text: str
    references: str
    input_token_counts: int
    generated_text: str
    

class DataPool:
    def __init__(self, samples: List[Sample]):
        self._samples = samples

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, ix: int) -> Sample:
        if ix >= len(self):
            raise StopIteration
        sample = self._samples[ix]
        return sample, 1.0

    def sample(self) -> Sample:
        random_sample = random.choice(self._samples)
        return random_sample

    @abstractclassmethod
    def prepare(cls, **args) -> 'DataPool':
        """
        A factory method to instantiate data pool
        """
        raise NotImplementedError

    def split(self, split_ratios: List[float]) -> List['DataPool']:
        start_ix = 0
        pools = []
        for ratio in split_ratios:
            count = int(len(self) * ratio)
            end_ix = start_ix + count
            pools.append(type(self)(self._samples[start_ix: end_ix]))
            start_ix = end_ix
        return pools



class AlpacaPlus(DataPool):
    
    @classmethod
    def prepare(cls, 
                split: str,
                num_tokens_to_predict: int,
                tokenizer: AutoTokenizer,
                gen_config: dict,):
        def preprocess(examples):
            if examples["input"]:
                source = f"{FIXED_TOKENS['instruction']}{examples['instruction']}{FIXED_TOKENS['input']}{examples['input']}{FIXED_TOKENS['output']}"
            else:
                source = f"{FIXED_TOKENS['instruction']}{examples['instruction']}{FIXED_TOKENS['output']}"
            inputs = tokenizer(source, padding="max_length", max_length=128, truncation=True)
            inputs = dict(inputs)
            inputs['text'] = source
            return inputs
        
        nltk.download('stopwords')
        model_cls = AutoModelForSeq2SeqLM if "t5" in gen_config['model_name'] else AutoModelForCausalLM
        gen_model = model_cls.from_pretrained(
            gen_config['model_name'],
            pad_token_id = tokenizer.pad_token_id,
            device_map = 'auto'
        )
        dataset = load_dataset("data/alpaca_plus.py")
        dataset_split = dataset[split].map(preprocess, remove_columns=["instruction", "input",])
        bs=64
        samples = []
        with torch.no_grad():
            for i in tqdm(range(math.ceil(len(dataset_split)/bs))):
                subset = dataset_split[bs*i:bs*(i+1)]
                input_ids = torch.tensor(subset['input_ids'])
                attention_mask = torch.tensor(subset['attention_mask'])
                gen_output = gen_model.generate(
                    inputs=input_ids.to("cuda"),
                    attention_mask=attention_mask.to("cuda"),
                    **gen_config['generation_kwargs'],
                )
                for ix in range(len(subset['output'])):
                    input_text = tokenizer.decode(input_ids[ix], skip_special_tokens=True)
                    input_token_counts = len(tokenizer(input_text)['input_ids'])
                    gen_tokens = gen_output[ix][-num_tokens_to_predict:]
                    gen_texts = tokenizer.decode(gen_tokens, skip_special_tokens=True)
                    sample = Sample(
                        id=f"{split}_{bs*i+ix}",
                        prompt_or_input_text=input_text,
                        references=subset['output'][ix],
                        input_token_counts=input_token_counts,
                        generated_text=gen_texts,
                    )
                    samples.append(sample)

            #TODO Remove this row 
                if i > 10:
                    break
        pool_instance = cls(samples)
        torch.cuda.empty_cache()
        return pool_instance

class GPTeacher(DataPool):
    
    @classmethod
    def prepare(cls, 
                split: str,
                # data_path: str, TODO 없애기
                num_tokens_to_predict: int,
                tokenizer: AutoTokenizer,
                gen_config: dict,):
        def preprocess(examples):
            if examples["input"]:
                #source = f"Instruction: {examples['instruction']}\nInput: {examples['input']}\nOutput: {examples['output']}"
                source = f"{FIXED_TOKENS['instruction']}{examples['instruction']}{FIXED_TOKENS['input']}{examples['input']}{FIXED_TOKENS['output']}"
            else:
                # source = f"Instruction: {examples['instruction']}\nOutput: \n"
                source = f"{FIXED_TOKENS['instruction']}{examples['instruction']}{FIXED_TOKENS['output']}"
            inputs = tokenizer(source, max_length=1024, truncation=True)
            inputs = dict(inputs)
            inputs['text'] = source
            return inputs
        
        nltk.download('stopwords')
        model_cls = AutoModelForSeq2SeqLM if "t5" in gen_config['model_name'] else AutoModelForCausalLM
        gen_model = model_cls.from_pretrained(
            gen_config['model_name'],
            pad_token_id = tokenizer.pad_token_id,
            device_map = 'auto'
        )
        dataset = load_dataset("teknium/GPTeacher-General-Instruct")
        dataset_split = dataset[split].map(preprocess, remove_columns=["instruction", "input",])
        bs=1
        samples = []
        with torch.no_grad():
            for i in tqdm(range(math.ceil(len(dataset_split)/bs))):
                subset = dataset_split[bs*i:bs*(i+1)]
                input_ids = torch.tensor(subset['input_ids'])
                attention_mask = torch.tensor(subset['attention_mask'])
                gen_output = gen_model.generate(
                    inputs=input_ids.to("cuda"),
                    attention_mask=attention_mask.to("cuda"),
                    **gen_config['generation_kwargs'],
                )
                for ix in range(len(subset['response'])):
                    input_text = tokenizer.decode(input_ids[ix], skip_special_tokens=True)
                    input_token_counts = len(tokenizer(input_text)['input_ids'])
                    gen_tokens = gen_output[ix][-num_tokens_to_predict:]
                    gen_texts = tokenizer.decode(gen_tokens, skip_special_tokens=True)
                    sample = Sample(
                        id=f"{split}_{bs*i+ix}",
                        prompt_or_input_text=input_text,
                        references=subset['response'][ix],
                        input_token_counts=input_token_counts,
                        generated_text=gen_texts,
                    )
                    samples.append(sample)

        pool_instance = cls(samples)
        return pool_instance

# class SelfInstruct(DataPool):
    
#     @classmethod
#     def prepare(cls, 
#                 split: str,
#                 # data_path: str, TODO 없애기
#                 num_tokens_to_predict: int,
#                 tokenizer: AutoTokenizer,
#                 gen_config: dict,):
#         def preprocess(examples):
#             if examples["input"]:
#                 #source = f"Instruction: {examples['instruction']}\nInput: {examples['input']}\nOutput: {examples['output']}"
#                 source = f"Instruction: {examples['instruction']}\nInput: {examples['input']}\nOutput: \n"
#             else:
#                 #source = f"Instruction: {examples['instruction']}\nOutput: {examples['output']}"
#                 source = f"Instruction: {examples['instruction']}\nOutput: \n"
#             inputs = tokenizer(source, padding="max_length", max_length=128, truncation=True)
#             return inputs
        
#         nltk.download('stopwords')
#         gen_model = AutoModelForCausalLM.from_pretrained(
#             gen_config['model_name'],
#             pad_token_id = tokenizer.pad_token_id,
#             device_map = 'auto'
#         )
#         #dataset = load_dataset("data/self_instruct")
#         dataset = load_from_disk("data/alpaca_plus_subset2")
#         dataset_split = dataset[split].map(preprocess, remove_columns=["instruction","input", 'split'])
#         bs=64
#         samples = []
#         with torch.no_grad():
#             for i in tqdm(range(math.ceil(len(dataset_split)/bs))):
#                 subset = dataset_split[bs*i:bs*(i+1)]
#                 input_ids = torch.tensor(subset['input_ids'])
#                 attention_mask = torch.tensor(subset['attention_mask'])
#                 gen_output = gen_model.generate(
#                     inputs=input_ids.to("cuda"),
#                     attention_mask=attention_mask.to("cuda"),
#                     **gen_config['generation_kwargs'],
#                 )
#                 for ix in range(len(subset['output'])):
#                     input_text = tokenizer.decode(input_ids[ix], skip_special_tokens=True)
#                     gen_texts = tokenizer.decode(gen_output[ix][-num_tokens_to_predict:], skip_special_tokens=True)
#                     sample = Sample(
#                         id=f"{split}_{bs*i+ix}",
#                         prompt_or_input_text=input_text,
#                         references=subset['output'][ix],
#                         input_token_counts=(input_ids[ix] != tokenizer.pad_token_id).sum().item(),
#                         generated_text=gen_texts,
#                         stopword_ratio=get_stopword_ratio(gen_texts),
#                     )
#                     samples.append(sample)
#         pool_instance = cls(samples)
#         torch.cuda.empty_cache()
#         return pool_instance

def get_stopword_ratio(texts: str): #TODO 이거 word단위가 아니라 gpt token 단위로 계산하기
    stop_words = set(nltk.corpus.stopwords.words('english'))
    words = nltk.tokenize.word_tokenize(texts)
    stopword_ratio = 0
    if len(words) != 0:
        stopword_ratio = sum([1 for word in words if word in stop_words])/len(words)
    return stopword_ratio

def get_stopword_ratio(input_texts: str, input_token_counts: int, tokenizer: AutoTokenizer): #TODO 이거 word단위가 아니라 gpt token 단위로 계산하기
    stop_words = set(nltk.corpus.stopwords.words('english'))
    words = nltk.tokenize.word_tokenize(input_texts)
    stopwords_in_gen_tokens = 0
    for w in words:
        if w in stop_words:
            stopwords_in_gen_tokens += len(tokenizer(w)['input_ids'])
    return stopwords_in_gen_tokens / input_token_counts
