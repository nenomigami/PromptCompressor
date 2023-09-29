from typing import Any, Dict, List
from gym import Env
from stable_baselines3.common.policies import BasePolicy
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from data.instruction_pool import Sample, get_fixed_token_counts, FIXED_TOKENS
from pcrl.utils.logging_utils import Tracker
from pcrl.utils.metric import BaseMetric
import numpy as np

def get_batch(samples: List[Sample], batch_size: int):
    current_ix = 0
    n_samples = len(samples)
    while current_ix < n_samples:
        current_batch = samples[current_ix : current_ix + batch_size]
        yield current_batch
        current_ix += batch_size

def list_in_dict_map(fn, dict, fn_args=None):
    b = dict[list(dict.keys())[0]].shape[0]
    output = []
    for i in range(b):
        single_obs = {k:dict[k][i] for k in dict.keys()}
        if fn_args is None:
            output.append(fn(single_obs))
        else:
            output.append(fn(single_obs, fn_args[i]))
    return np.stack(output)
        
def get_compress_ratio(batch, compressed_counts, fixed_token_counts):
    compress_rate = np.zeros(len(batch))
    fixed_n_tokens = np.zeros(len(batch), dtype=np.int16)
    for i, sample in enumerate(batch):
        for k, v in FIXED_TOKENS.items():
            if v in sample.prompt_or_input_text:
                fixed_n_tokens[i] += len(fixed_token_counts[k])
        compress_rate[i] = 1 - (compressed_counts[i] - fixed_n_tokens[i]) / (sample.input_token_counts - fixed_n_tokens[i])
        assert compress_rate[i] >= 0 or compress_rate[i] < 1, "compress rate must be between 0 and 1"
    return compress_rate.tolist()

def evaluate_on_samples(
    gen_model: AutoModelForCausalLM,
    gen_tokenizer: AutoTokenizer,
    policy: BasePolicy,
    env: Env,
    samples: List[Sample],
    batch_size: int,
    max_prompt_length: int,
    metrics: List[BaseMetric],
    epoch: int,
    split_name: str,
    tracker: Tracker = None,
    gen_kwargs: Dict[str, Any] = None,
):
    obs_space = env.get_attr("obs_space")[0]
    act_space = env.get_attr("act_space")[0]
    # generate text by batch
    all_original_prompt_texts = []
    all_compressed_prompt_texts = []
    all_original_generated_texts = []
    all_compressed_generated_texts = []
    all_truncated_compressed_texts = []
    all_compressed_ratios = []
    all_ref_texts = []
    
    fixed_token_counts = get_fixed_token_counts(gen_tokenizer)
    #fixed_token_len = len(fixed_token_counts['instruction'] + fixed_token_counts['input'] + fixed_token_counts['output'])
    for batch in tqdm(list(get_batch(samples, batch_size)), desc="Evaluating"):
        #compress prompt
        observations = obs_space.observation(batch)
        action_masks = list_in_dict_map(act_space.action_mask, observations)
        actions, _ = policy.predict(observations, action_masks=action_masks, deterministic=True)

        # original_tokens = [sample.prompt_or_input_text for sample in batch]
        compressed_tokens = list_in_dict_map(act_space.process_actions, observations, actions)
        compressed_texts = [act_space.decode(tokens) for tokens in compressed_tokens]
        #compressed_tokens_counts = get_n_token(compressed_prompt) - get_fixed_token_len(compressed_prompt, fixed_token_counts).item()
        gen_output = generate(
            compressed_texts,
            gen_model,
            gen_tokenizer,
            gen_kwargs,
        )
        ref_tokens = gen_tokenizer(
            [sample.references for sample in batch],
            padding="max_length",
            max_length=512,
        )
        ref_length = np.array([sum(att_mask) for att_mask in ref_tokens['attention_mask']])

        batch_original_prompt_texts = [sample.prompt_or_input_text for sample in batch]
        batch_compressed_prompt_texts = compressed_texts
        batch_original_generated_texts = [sample.generated_text for sample in batch]
        batch_compressed_generated_texts = gen_output['gen_texts']
        batch_truncated_compressed_texts = [gen_tokenizer.decode(
            gen_output['gen_tokens'][i, :length], skip_special_tokens=True)
            for i, length in enumerate(ref_length)] 
        batch_ref_texts = [sample.references for sample in batch]
        batch_compressed_ratios = get_compress_ratio(batch, gen_output['compressed_token_counts'], fixed_token_counts)
        all_original_prompt_texts.extend(batch_original_prompt_texts)
        all_compressed_prompt_texts.extend(batch_compressed_prompt_texts)
        all_original_generated_texts.extend(batch_original_generated_texts)
        all_compressed_generated_texts.extend(batch_compressed_generated_texts)
        all_compressed_ratios.extend(batch_compressed_ratios)
        all_truncated_compressed_texts.extend(batch_truncated_compressed_texts)
        all_ref_texts.extend(batch_ref_texts)
    
    # compute metrics
    sample_scores_by_metric = {}
    if metrics is not None:
        for metric in metrics:
            metric_dict = metric.compute(
                all_original_generated_texts,
                all_compressed_generated_texts,
                all_truncated_compressed_texts,
                all_ref_texts,
                split_name,
            )

            for metric_key, sample_scores in metric_dict.items():
                sample_scores_by_metric[metric_key] = sample_scores
        sample_scores_by_metric["compressed_ratio"] = np.mean(all_compressed_ratios)
    sample_predictions_dict = []
    for ix, (sample, prompt_text, comp_prompt_text, org_generated_text, \
             comp_generated_text, trun_generated_text, compressed_ratio, ref_texts) in enumerate(
        zip(samples, all_original_prompt_texts, all_compressed_prompt_texts, all_original_generated_texts,\
             all_compressed_generated_texts, all_truncated_compressed_texts, all_compressed_ratios, all_ref_texts)
    ):
        if ix >= 28*4:
            break
        sample_prediction = {
            "split_name": split_name,
            "sample_id": sample.id,
            "prompt_text": prompt_text,
            "compressed_prompt": comp_prompt_text,
            "original_generated_text": org_generated_text,
            "compressed_generated_text": comp_generated_text,
            "truncated_compressed_text": trun_generated_text,
            "compressed_ratio": compressed_ratio,
            "ref_text": ref_texts
        }
        sample_predictions_dict.append(sample_prediction)

    if tracker is not None:
        # log the entire predictions
        tracker.log_predictions(epoch, split_name, sample_predictions_dict)
        # log the corpus level scores
        tracker.log_metrics(epoch, split_name, sample_scores_by_metric)

def generate(
    compressed_prompts: List[str],
    gen_model: AutoModelForCausalLM,
    gen_tokenizer: AutoTokenizer,
    gen_kwargs: Dict[str, Any],
    prefix_fn = None,
):   
    if prefix_fn is not None: 
        compressed_prompts = prefix_fn(compressed_prompts)

    encodings = gen_tokenizer(
        compressed_prompts,
        padding="max_length",
        max_length=128,
        return_tensors="pt",
        return_attention_mask=True,
        truncation=True,
    )

    # generate
    gen_output = gen_model.generate(
        inputs=encodings.input_ids.to(gen_model.device),
        attention_mask=encodings.attention_mask.to(gen_model.device),
        **gen_kwargs
    )
    gen_tokens = gen_output[:, -gen_kwargs['max_new_tokens']:]
    gen_texts = gen_tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
    output_dict = {
        "gen_texts": gen_texts,
        "gen_tokens": gen_tokens,
        "compressed_token_counts": (encodings.input_ids != gen_tokenizer.pad_token_id).sum(dim=1).tolist(),
        # "compressed_token": 
    }
    return output_dict
