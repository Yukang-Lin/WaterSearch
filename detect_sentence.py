from pred_baseline import load_model_and_tokenizer, str2bool
import argparse
import os
import json
import torch
import numpy as np
from scipy.stats import binom, chi2
from tqdm import tqdm
import random
import hashlib
import sys
import os


def load_jsonl(file_path):
    """
    Load a JSONL file and return a list of dictionaries.
    """
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]
    return data

def exact_max_binomial_p_value(green_list, n_tokens, gamma, beam_num):
    """
    Compute p-value for H0: max(k1,k2,k3,k4) ~ max of 4 independent B(n,p).
    """
    observed_max = max(green_list)
    p_value = 1 - (binom.cdf(observed_max - 1, n_tokens, gamma)) ** beam_num
    return p_value

def fisher_test(p_values):
    """
    Combine p-values using Fisher's method.
    """
    combined_stat = -2 * np.sum(np.log(p_values))
    df = 2 * len(p_values)
    combined_p_value = 1 - chi2.cdf(combined_stat, df)
    return combined_p_value

def detect_chi2_sentence(sent_green_nums, chunk_size, gamma, beam_num):
    p_values = []
    for green_list, sent_len in zip(sent_green_nums, chunk_size):
        if sent_len < 6:
            print("Warning: sentence too short to test.")
            continue
        p_value = exact_max_binomial_p_value(green_list, sent_len, gamma, beam_num)
        p_values.append(p_value)
    combined_p_value = fisher_test(p_values)
    return combined_p_value

def build_seed_pool(encoded_prompt, seed_pool, seed=42):
    pre_seed = 1
    for pre_token in encoded_prompt[0][-1:]:
        pre_seed *= (pre_token.item() + 1)
    
    pre_seed = pre_seed & 0xFFFFFFFF
    large_prime = 15485863
    hash_seed = (pre_seed * large_prime + seed) ^ (seed << 1)
    random.seed(hash_seed)
    random.shuffle(seed_pool)
    return seed_pool

def hash_fn(x: int) -> int:
    """solution from https://stackoverflow.com/questions/67219691/python-hash-function-that-returns-32-or-64-bits"""
    x = np.int64(x)
    return int.from_bytes(hashlib.sha256(x).digest()[:4], 'little')

def split_sentences(text):
    import re
    sentence_list = re.findall(r'[^。！？.!?]*[。！？.!?]', text, flags=re.U)
    residual = re.sub(r'[^。！？.!?]*[。！？.!?]', '', text, flags=re.U).strip()
    if residual:
        sentence_list.append(residual)
    return [s.strip() for s in sentence_list if s.strip()]

def get_gnum_per_sentence(mode, tokenizer, prompt, pred, beam_num, chunk_size, gamma, hash_key = 15485863):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    encoded_prompt = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False)
    vocab_size = len(tokenizer.get_vocab())
    seed_pool = list(range(1, 5000))
    sent_green_nums, sent_lens = [], []
    prev_token = encoded_prompt[0][-1].item() 
    sentences = split_sentences(pred)
    prefix = encoded_prompt
    for sentence in sentences:
        encoded_sentence = tokenizer.encode(sentence, return_tensors="pt", truncation=True, add_special_tokens=False)[0]
        seed_pool = build_seed_pool(prefix, seed_pool)
        cur_seeds = seed_pool[:beam_num]
        tmp_prev_token = prev_token
        max_green_token_nums = 0
        sent_green_nums_seed = []
        for seed in cur_seeds:
            green_token_count = 0
            prev_token = tmp_prev_token
            for token in encoded_sentence.tolist():
                if mode == "old":
                    redlist_size = int(vocab_size*(1 - gamma))
                    rng = torch.Generator(device=device)
                    rng.manual_seed(hash_key*prev_token*seed)
                    vocab_permutation = torch.randperm(vocab_size, device=device, generator=rng)
                    redlist_ids = vocab_permutation[:redlist_size]
                elif mode == "gpt":
                    rng = np.random.default_rng(hash_fn(seed))
                    mask = np.array([True] * int(gamma * vocab_size) + [False] * (vocab_size - int(gamma * vocab_size)))
                    rng.shuffle(mask)
                    blacklist_indices = np.where(~mask)[0]
                    redlist_ids = blacklist_indices.tolist()
                elif mode == "v2":
                    greenlist_size = int(vocab_size * gamma)
                    rng = torch.Generator(device=device)
                    rng.manual_seed(hash_key*prev_token*seed)
                    vocab_permutation = torch.randperm(vocab_size, device=device, generator=rng)
                    redlist_ids = vocab_permutation[greenlist_size:]
                else:
                    raise NotImplementedError(f"Mode {mode} is not implemented.")
                
                tok_in_ph_gl = token in redlist_ids
                if not tok_in_ph_gl:
                    green_token_count += 1
                prev_token = token
            if green_token_count > max_green_token_nums:
                max_green_token_nums = green_token_count
            sent_green_nums_seed.append(green_token_count)
        encoded_sentence = encoded_sentence.unsqueeze(0)
        prefix = torch.cat([prefix, encoded_sentence], dim=1)
        sent_green_nums.append(sent_green_nums_seed)
        sent_lens.append(len(encoded_sentence[0]))
    return sent_green_nums, sent_lens



def main(args):
    model2path = json.load(open("config/model2path.json", "r"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # get model name
    model_name = args.input_dir.split("/")[-1].split("_")[0]
    print(model_name)
    # define your model
    tokenizer = load_model_and_tokenizer(model2path[model_name], model_name, device, load_token_only=True)
    # get all files from input_dir
    files = os.listdir(args.input_dir)
    # get all json files
    json_files = [f for f in files if f.endswith(".jsonl")]
    os.makedirs(args.input_dir + "/detect", exist_ok=True)
    if args.task == 'all':
        data_names = ['knowledge_memorization', 'knowledge_understanding', 'longform_qa','finance_qa', 'hotpotqa', 'lcc', 'multi_news', 'qmsum', 'alpacafarm', 'repobench-p', 'processed_c4']
    else:
        data_names = [args.task]
    tp, skip_count = 0, 0
    detection_rate_dict = dict()
    for data_name in data_names:
        print(f"{data_name} has began.........")
        # read jsons
        data_path = os.path.join(args.input_dir, data_name + ".jsonl")
        if os.path.exists(data_path):
            data = load_jsonl(data_path)
        else:
            print(f"Warning: {data_path} does not exist.")
            continue

        p_value_list = []
        print(f"Loaded {len(data)} samples from {data_path}")
        for idx, item in tqdm(enumerate(data), total=len(data)):
            prompt = item['prompt']
            pred = item['pred']
            completions_tokens = item['completions_tokens']
            if len(completions_tokens) >= args.test_min_tokens:
                gnum_list, sent_lens = get_gnum_per_sentence(args.mode, tokenizer, prompt, pred, args.beam_num, args.chunk_size, args.gamma)
                p_value = detect_chi2_sentence(gnum_list, sent_lens, args.gamma, args.beam_num)
            else:
                print(f"Warning: sequence {idx} is too short to test.")
                skip_count += 1
                p_value = None
            p_value_list.append(p_value)

        wm_pred = []
        for p_value in p_value_list:
            if p_value is not None:
                wm_pred.append(True if p_value <= args.threshold else False)
            else:
                wm_pred.append(None)
        save_dict = {
            'p_value_list': p_value_list,
            'wm_pred': wm_pred
        }
        skip_short_wm_pred = [wm for wm in wm_pred if wm is not None]
        wm_pred_average = sum(skip_short_wm_pred) / len(skip_short_wm_pred)
        print(sum(skip_short_wm_pred))
        print(len(skip_short_wm_pred))
        save_dict.update({'wm_pred_average': wm_pred_average})
        detection_rate_dict[data_name] = wm_pred_average

        print(f"wm_pred_average: {wm_pred_average}")
        print(f"num of non-skipped samples: {len(skip_short_wm_pred)}")
        output_path = f"{args.input_dir}/detect/{data_name}_{args.gamma}_{args.delta}.jsonl"
        with open(output_path, 'w') as f:
            json.dump(save_dict, f)

        tp += sum(skip_short_wm_pred)

    if args.task == 'all':
        avg_tp = tp / (200 * 9 + 805 - skip_count)
        print(f"avg_tp: {avg_tp}")
        detection_rate_dict['average'] = avg_tp
    with open(f"{args.input_dir}/detect/{args.gamma}_{args.delta}.jsonl", 'w') as f:
        json.dump(detection_rate_dict, f, indent=4)

parser = argparse.ArgumentParser(description="Process watermark to calculate z-score for every method")
parser.add_argument(
    "--input_dir",
    type=str)
parser.add_argument(
    "--task",
    type=str,
    default='all',
    help="Task to process. If 'all', process all tasks in the input directory."
)
parser.add_argument( # for gpt watermark
        "--wm_key", 
        type=int, 
        default=0)
parser.add_argument(
    "--gamma",
    type=float,
    default=0.25
)
parser.add_argument(
    "--delta",
    type=float,
    default=2.0,
    help="Delta value for watermark detection."
)
parser.add_argument(
    "--beam_num",
    type=int,
    default=4,
    help="Number of beams to use for generation."
)
parser.add_argument(
    "--chunk_size",
    type=int,
    default=20,
    help="Number of tokens to generate in each chunk."
)
parser.add_argument(
    "--threshold",
    type=float,
    default=0.05)

parser.add_argument(
    "--test_min_tokens",
    type=int, 
    default=6)

parser.add_argument( # for v2 watermark
        "--seeding_scheme",
        type=str,
        default="simple_1",
        help="Seeding scheme to use to generate the greenlists at each generation and verification step.",
    )

parser.add_argument( # for v2 watermark
    "--normalizers",
    type=str,
    default="",
    help="Single or comma separated list of the preprocessors/normalizer names to use when performing watermark detection.",
)

parser.add_argument( # for v2 watermark
    "--ignore_repeated_bigrams",
    type=str2bool,
    default=False,
    help="Whether to use the detection method that only counts each unqiue bigram once as either a green or red hit.",
)

parser.add_argument( # for v2 watermark
    "--select_green_tokens",
    type=str2bool,
    default=True,
    help="How to treat the permuation when selecting the greenlist tokens at each step. Legacy is (False) to pick the complement/reds first.",
)


parser.add_argument( 
    "--mode",
    type=str,
    default="old",
    choices=['old', 'new', 'gpt', 'v2', 'no']
)

parser.add_argument( 
    "--dynamic_seed",
    type=str,
    default="markov_1",
    choices=[None, "initial", "markov_1"],
    help="mission-name",
)

args = parser.parse_args()

main(args)
