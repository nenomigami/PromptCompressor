from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from tqdm import tqdm
from data.instruction_pool import FIXED_TOKENS, get_fixed_token_counts
import numpy as np
import math
import pandas as pd
import torch
import evaluate
import os 

MODEL_ALIAS = {
    "gpt2-xl": "gpt2-xl-finetuned",
    "flan-t5-xl": "flan-t5-xl-finetuned",
    "llama2": "meta-llama/Llama-2-7b-chat-hf",
    "flan-t5-xxl": "google/flan-t5-xxl",
    "falcon": "tiiuae/falcon-7b-instruct"
}

def get_exp_type(gen_model:str):
    if gen_model in ["gpt2-xl", "flan-t5-xl"]:
        return "instruction"
    elif gen_model in ["llama2", "flan-t5-xxl", "falcon"]:
        return "transfer"
    else:
        raise NotImplementedError

def concat_instruction_input_for_validation(examples):
    if examples["input"]:
        source = FIXED_TOKENS["instruction"] + examples['instruction'] + FIXED_TOKENS["input"] + examples['input'] + FIXED_TOKENS["output"]
    else:
        source = FIXED_TOKENS["instruction"] + examples['instruction'] + FIXED_TOKENS["output"]
    return {"text" : source}

def get_fixed_token_len(texts, fixed_token_counts):
    if isinstance(texts, str):
        texts = [texts] 
    fixed_n_tokens = np.zeros(len(texts), dtype=np.int16)
    for i, text in enumerate(texts):
        for k, v in FIXED_TOKENS.items():
            if v in text or v.rstrip(" ") in text:
                fixed_n_tokens[i] += len(fixed_token_counts[k])
    return fixed_n_tokens

def eval_original(args):
    results_folder = args.results_dir
    eval_type = get_exp_type(args.gen_model)
    directory = f"{results_folder}/{eval_type}/original"
    if not os.path.exists(directory):
        os.makedirs(directory)
    summary = "summary.csv"

    rouge = evaluate.load('rouge')
    # Load the dataset
    dataset = load_dataset("data/alpaca_plus.py")

    gen_model_name = MODEL_ALIAS[args.gen_model]
    bs = args.bs

    # Load the tokenizer
    gen_tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
    if gen_tokenizer.pad_token is None:
        gen_tokenizer.pad_token = gen_tokenizer.eos_token 
    gen_tokenizer.padding_side = "left"
    gen_tokenizer.truncation_side = "left"

    # Load the model
    model_cls = AutoModelForSeq2SeqLM if "t5" in gen_model_name else AutoModelForCausalLM
    gen_model = model_cls.from_pretrained(gen_model_name, device_map = 'auto', cache_dir=".", trust_remote_code=True)

    fixed_token_counts = get_fixed_token_counts(gen_tokenizer)

    for split in ["validation_seen", "validation_unseen", "validation_human"]:
        dataset_split = dataset[split].map(concat_instruction_input_for_validation, batched=False, num_proc=8)
        results = {"prompt": [],
                   "tokens" : [],
                   "gen_texts": [],
                   "token_counts": [],
                   "rouge_L": []}

        for i in tqdm(range(math.ceil(len(dataset_split)/bs))):
            subset = dataset_split[bs*i:bs*(i+1)]
            if bs == 1:
                gen_encodings = gen_tokenizer(subset['text'], return_tensors='pt', max_length=512, truncation=True, return_attention_mask=True)
            else:
                gen_encodings = gen_tokenizer(subset['text'], 
                                return_tensors='pt',
                                max_length=512,
                                padding="max_length",
                                truncation=True,
                                return_attention_mask=True,
                            )
            
            results["prompt"] += subset['text']
            results["tokens"] += [(input_id[input_id!=gen_tokenizer.pad_token_id]).tolist() for input_id in gen_encodings['input_ids']]
            fixed_tokens = torch.tensor(get_fixed_token_len(subset['text'], fixed_token_counts))
            results["token_counts"] += ((gen_encodings['input_ids'] != gen_tokenizer.pad_token_id).sum(axis=-1) - fixed_tokens).tolist()
            
            gen_output = gen_model.generate(
                input_ids=gen_encodings.input_ids.to("cuda"),
                attention_mask=gen_encodings.attention_mask.to("cuda"),
                return_dict_in_generate=True,
                output_scores=True,
                min_new_tokens=1,
                max_new_tokens=128,
                do_sample=False,
            )
            
            # get only the generated text (excluding prompt)
            if "t5" in gen_model_name:
                gen_tokens = gen_output["sequences"]
            else:
                gen_tokens = gen_output["sequences"][:, len(gen_encodings.input_ids[0]):]

            # to texts
            for j, output in enumerate(gen_tokens):
                if eval_type=='transfer':
                    gen_text = gen_tokenizer.decode(output, skip_special_tokens=True)
                else:        
                    reference_len = len(gen_tokenizer(subset['output'][j])['input_ids'])
                    gen_text = gen_tokenizer.decode(output[:reference_len], skip_special_tokens=True)
                results["gen_texts"].append(gen_text)
        
        results["rouge_L"] = rouge.compute(predictions=results["gen_texts"], references=dataset_split['output'], use_aggregator=False)['rougeL']
        df = pd.DataFrame(results)
        summ = pd.DataFrame({
            "id" : args.gen_model,
            "split" : split,
            "model" : "original",
            "rouge_l": df["rouge_L"].mean(),
            "cr": 0,
            "seed": 0
        }, index = [0] 
        )
        if os.path.exists(summary):
            prev_summ = pd.read_csv(summary)
            summ = pd.concat([prev_summ, summ], axis=0).reset_index(drop=True)
            summ.to_csv(summary, index=False)
        else:
            summ.to_csv(summary, index=False)
        df.to_csv(f"{directory}/{args.gen_model}_{split}_wth.csv", index=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_model", type=str, default="llama2")
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--results_dir", type=str, default="results")

    args = parser.parse_args()

    eval_original(args)