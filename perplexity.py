import requests
import json
import os
from transformers import LlamaForCausalLM, LlamaTokenizerFast, AutoModelForCausalLM, AutoTokenizer, OPTForCausalLM
import sys
import torch
from tqdm import tqdm
import traceback
import datasets
import numpy as np
import time
from utils import (
    Column_to_check,
    Hf_Name_and_Split,
    prepare_query,
    prepare_dataset,
)

np.random.seed(42)

def self_info(text, model, tokenizer, merge = False):
    def merge_sub_tokens(log_probs, word_ids):
        # merge log probs of sub_tokens
        merged_log_probs = []
        current_word_id = None
        current_word_log_prob = None
        counter = 1

        for log_prob, word_id in zip(log_probs, word_ids):
            if word_id is not None:
                if current_word_id != word_id:
                    if current_word_id is not None:
                        merged_log_probs.extend([current_word_log_prob] * counter)
                    counter = 1
                    current_word_id = word_id
                    current_word_log_prob = log_prob
                else:
                    counter += 1
                    current_word_log_prob = current_word_log_prob + log_prob

        if current_word_id is not None:
            merged_log_probs.extend([current_word_log_prob] * counter)

        return merged_log_probs

    # this function is used to get the self-information of a text
    # the model should be a causal language model, e.g. GPT2LMHeadModel

    # tokenize the text
    text = f"{tokenizer.bos_token}{text}"
    encoding = tokenizer(text, return_tensors="pt", max_length=model.config.max_position_embeddings, truncation=True)
    encoding = encoding.to(model.device)

    # get the logits
    with torch.no_grad():
        logits = model(**encoding).logits
        probs = torch.softmax(logits, dim=-1)
        info = -torch.log(probs)

    input_ids = encoding['input_ids']
    input_ids_expaned = input_ids[:, 1:].unsqueeze(-1)
    info = info[:, :-1].gather(-1, input_ids_expaned).squeeze(-1).squeeze(0).tolist()

    tokens = [tokenizer.decode(token_) for token_ in input_ids.squeeze().tolist()[1:]]
    if merge:
        info = merge_sub_tokens(info, encoding.word_ids()[1:])
    return tokens, info

def select_token_window(text, token_count=400):
    tokens = text.split()
    if len(tokens) <= token_count:
        return text
    ramdom_start = np.random.randint(0, len(tokens) - token_count)
    tokens = tokens[ramdom_start:ramdom_start + token_count]
    return ' '.join(tokens)

def load_model_and_tokenizer(model_name):

    if 'GPTQ' in model_name:
        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

        # only llama-30b use gptq
        model = AutoGPTQForCausalLM.from_quantized(model_name, device = 'cuda:0', use_safetensors = True, disable_exllama=True if '30b' in model_name else False)
        tokenizer = LlamaTokenizerFast.from_pretrained(model_name)
    elif 'llama' in model_name.lower():
        model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map='auto')
        tokenizer = LlamaTokenizerFast.from_pretrained(model_name)
    elif 'opt' in model_name.lower():
        model = OPTForCausalLM.from_pretrained(model_name, device_map='auto')
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    elif 'gpt2' == model_name.lower():
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=True)

    return model, tokenizer

if __name__ == "__main__":

    # if you are doing contamination test, you will need the memorised and fresh baseline, so set this to True
    doing_contamination_test = False
    # default is to use the val or test split, if you want to use the train split, set this to True
    use_train_split = False

    num_token = 300
    num_samples = 300
    output_file = f'reports/perplexity_report.json'
    model_names = ['Qwen/Qwen-7B', 'baichuan-inc/Baichuan2-7B-Base']

    evaluation_datasets = ['mmlu', 'mmlu_train', 'ceval', 'ceval_train']
    datasets_to_check = prepare_dataset(evaluation_datasets, n = num_samples)

    if doing_contamination_test:
        evaluation_base = 'wiki'
        memorised_time = '2022-8'

        baseline_datasets = [f'{evaluation_base}_clean', f'{evaluation_base}_all']
        datasets_to_check.update(prepare_dataset(baseline_datasets, n = 500, config = memorised_time))
    
    datasets_and_texts = {}
    for dataset_name, ds in datasets_to_check.items():

        all_texts = []
        for i, row in tqdm(enumerate(ds), desc=f'Processing {dataset_name}'):
            # query is the verbatized test sample
            query = prepare_query(dataset_name, row)
            if query['query'] is None: continue
            query_chunked = select_token_window(query['query'], token_count=num_token)

            datasets_and_texts.setdefault(dataset_name, []).append(query_chunked)

    results = {}
    for model_name in model_names:
        model, tokenizer = load_model_and_tokenizer(model_name)
        print('=====================')
        print(f'Model: {model_name}')
        results[model_name] = {}

        for dataset_name, texts in datasets_and_texts.items():
            print(f'=====================')
            print(f'Dataset: {dataset_name}')
            infos = []
            for text in tqdm(texts):
                try:
                    tokens, info = self_info(text, model, tokenizer)
                except:
                    traceback.print_exc()
                    time.sleep(10)
                    continue
                infos.append(sum(info)/len(info))
            print(f'Average self-info: {sum(infos)/len(infos)}')
            results[model_name][dataset_name] = sum(infos)/len(infos)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)