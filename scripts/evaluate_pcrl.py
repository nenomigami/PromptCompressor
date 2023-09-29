from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from tqdm import tqdm
from data.instruction_pool import FIXED_TOKENS, get_fixed_token_counts
from data.instruction_pool import FIXED_TOKENS, get_fixed_token_counts
from pcrl.utils.training_utils import build_env, build_tokenizer
from pcrl.envs.act_spaces import BatchFixedTokenAction
from pcrl.envs.obs_spaces import FixedTokenObservation
from pcrl.model.policy import BatchTokenPolicy

import numpy as np
import math
import pandas as pd
import torch
import evaluate
import os 
import nltk
import yaml
nltk.download("stopwords")

MODEL_ALIAS = {
    "gpt2-xl": "gpt2-xl-finetuned_alpaca",
    "flan-t5-xl": "flan-t5-xl-finetuned_alpaca",
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

def eval_pcrl(args):
    results_folder = args.results_dir
    seed = args.seed
    eval_type = get_exp_type(args.gen_model)
    pcrl_model_name = args.pcrl_model
    gen_model_name = MODEL_ALIAS[args.gen_model]

    directory = f"{results_folder}/{eval_type}/pcrl/{pcrl_model_name}/{seed}"
    if not os.path.exists(directory):
        os.makedirs(directory)
    summary = "summary.csv"

    rouge = evaluate.load('rouge')
    # Load the dataset
    dataset = load_dataset("data/alpaca_plus.py")
    bs = args.bs

    model_cls = AutoModelForSeq2SeqLM if "t5" in gen_model_name else AutoModelForCausalLM
    gen_model = model_cls.from_pretrained(gen_model_name, device_map='auto', cache_dir=".", trust_remote_code=True)#, pad_token_id = tokenizer.pad_token_id)
    device = "cuda"

    config_path = f'configs/{pcrl_model_name}.yml'
    with open(config_path, "r") as fp:
        config = yaml.safe_load(fp)

    alg_config = config['alg']
    env_config = config['env']
    env_config['n_envs'] = 1
    env_config['sample_k'] = 1
    env_config['args']['max_prompt_length'] = 512
    policy_config = alg_config["policy"]
    policy_args = policy_config["args"]

    model_tokenizer = build_tokenizer(alg_config['policy']['args'])
    env = build_env(env_config, model_tokenizer, [], env_config['sample_k'])

    gen_tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
    if gen_tokenizer.pad_token is None:
        gen_tokenizer.pad_token = gen_tokenizer.eos_token 
    gen_tokenizer.padding_side = "left"
    gen_tokenizer.truncation_side = "left"

    fixed_token_counts = get_fixed_token_counts(gen_tokenizer)

    obs_space = env.get_attr('obs_space')[0]
    act_space = env.get_attr('act_space')[0]

    checkpoint = 2 if "t5" in pcrl_model_name else 3
    model_pt_path = f"{pcrl_model_name}_{seed}/checkpoints/checkpoint_{checkpoint}"
    
    state_dict = torch.load(model_pt_path, map_location=torch.device("cuda"))
    policy = BatchTokenPolicy(
        env.observation_space,
        env.action_space,
        None,
        policy_args['model_name'],
    ).to(device=device)
    policy.load_from_dict(state_dict=state_dict["policy_state"])

    for split in ["validation_seen", "validation_unseen", "validation_human"]:
        org_df = pd.read_csv(f"{results_folder}/{eval_type}/original/{args.gen_model}_{split}_wth.csv")
        dataset_split = dataset[split].map(concat_instruction_input_for_validation, batched=False, num_proc=8)
        results = {"prompt": [],
                   "tokens" : [],
                   "gen_texts": [],
                   "token_counts": [],
                   "rouge_L": []}

        for i in tqdm(range(math.ceil(len(dataset_split)/bs))):
            if "falcon" in gen_model_name:
                bs = 1
            subset = dataset_split[bs*i:bs*(i+1)]
            b = len(subset['text'])
            model_encodings = model_tokenizer(subset['text'], max_length=512, padding="max_length", truncation=True, return_tensors="pt")
            is_irrel = np.stack([obs_space.is_irrel_token(b) for b in model_encodings.input_ids])
            obs = {
                "input_ids": np.array(model_encodings.input_ids),
                "attention_mask": np.array(model_encodings.attention_mask),
                "is_irrel": is_irrel,
            }

            action_masks = np.stack([act_space.action_mask({k:obs[k][i] for k in obs.keys()}) for i in range(b)])
            action, _ = policy.predict(obs, deterministic=True, action_masks=action_masks)

            new_obs = [act_space.process_actions({k:obs[k][i] for k in obs.keys()}, action[i]) for i in range(b)]
            new_obs = {k:np.array([new_obs[i]['input_ids'] for i in range(b)]).astype(np.int32) for k in obs.keys()}
            
            #org_token_counts = [len(tokens) for tokens in gen_tokenizer(subset['texts'])['input_ids']]
            #results["org_token_counts"] += org_token_counts
            comp_texts = model_tokenizer.batch_decode(new_obs['input_ids'], skip_special_tokens=True)
    

            if bs == 1:
                gen_encodings = gen_tokenizer(comp_texts, return_tensors='pt', max_length=512, truncation=True, return_attention_mask=True)
            else:
                gen_encodings = gen_tokenizer(comp_texts, 
                                return_tensors='pt',
                                max_length=512,
                                padding="max_length",
                                truncation=True,
                                return_attention_mask=True,
                            )
                
            results["prompt"] += comp_texts
            results["tokens"] += [(input_id[input_id!=gen_tokenizer.pad_token_id]).tolist() for input_id in gen_encodings['input_ids']]
            fixed_tokens = torch.tensor(get_fixed_token_len(comp_texts, fixed_token_counts))
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
        #    break

        #results["rouge_L"] = rouge.compute(predictions=results["gen_texts"], references=dataset_split['output'][:bs], use_aggregator=False)['rougeL']
        results["rouge_L"] = rouge.compute(predictions=results["gen_texts"], references=dataset_split['output'], use_aggregator=False)['rougeL']
        df = pd.DataFrame(results)
        summ = pd.DataFrame({
            "id" : args.gen_model,
            "split" : split,
            "model" : pcrl_model_name,
            "rouge_l": df["rouge_L"].mean(),
            "cr": (1 - df['token_counts']/org_df['token_counts']).mean(),
            "seed" : seed,
        }, index = [0] 
        )
        if os.path.exists(summary):
            prev_summ = pd.read_csv(summary)
            summ = pd.concat([prev_summ, summ], axis=0)
            summ.to_csv(summary, index=False)
        else:
            summ.to_csv(summary, index=False)

        #TODO 임시로 10개만
        df.to_csv(f"{directory}/{args.gen_model}_{split}.csv", index=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pcrl_model", type=str, default="gpt2-xl")
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--gen_model", type=str, default="falcon")
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--results_dir", type=str, default="results2")

    args = parser.parse_args()

    eval_pcrl(args)