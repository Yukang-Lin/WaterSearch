import torch.nn.functional as F
from watermark.old_watermark_search import BlacklistLogitsProcessor
from watermark.gptwm_search import GPTWatermarkLogitsWarper
from watermark.watermark_v2 import WatermarkLogitsProcessor
from transformers import LogitsProcessorList, LogitsProcessor
import torch
from typing import List
import time
from collections import Counter
import math
import numpy as np
import random
from transformers import GenerationConfig, DynamicCache, LogitsProcessor

class TextMetrics:
    def __init__(self, tokenizer, device):
        self.tokenizer = tokenizer
        self.device = device
    
    def tokenize(self, text):
        tokens = self.tokenizer.tokenize(text)
        return tokens
    
    def get_ngrams(self, tokens, n):
        if len(tokens) < n:
            return []
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    
    def calculate_bleu(self, reference, candidate, max_n=4, weights=None):
        if weights is None:
            weights = [1.0/max_n] * max_n
        
        ref_tokens = self.tokenize(reference)
        cand_tokens = self.tokenize(candidate)
        
        if len(cand_tokens) == 0:
            return 0.0
        
        precisions = []
        
        for n in range(1, max_n + 1):
            ref_ngrams = Counter(self.get_ngrams(ref_tokens, n))
            cand_ngrams = Counter(self.get_ngrams(cand_tokens, n))
            
            if len(cand_ngrams) == 0:
                precisions.append(0.0)
                continue
            
            matches = 0
            for ngram, count in cand_ngrams.items():
                matches += min(count, ref_ngrams.get(ngram, 0))
            
            precision = matches / sum(cand_ngrams.values())
            precisions.append(precision)
        
        if any(p == 0 for p in precisions):
            return 0.0
        
        log_precisions = [math.log(p) for p in precisions]
        geometric_mean = math.exp(sum(w * lp for w, lp in zip(weights, log_precisions)))
        
        ref_len = len(ref_tokens)
        cand_len = len(cand_tokens)
        
        if cand_len > ref_len:
            bp = 1.0
        else:
            bp = math.exp(1 - ref_len / cand_len) if cand_len > 0 else 0.0
        
        return bp * geometric_mean
    
    def lcs_length(self, seq1, seq2):
        m, n = len(seq1), len(seq2)
        
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    def calculate_rouge_l(self, reference, candidate):
        ref_tokens = self.tokenize(reference)
        cand_tokens = self.tokenize(candidate)
        
        if len(ref_tokens) == 0 or len(cand_tokens) == 0:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        lcs_len = self.lcs_length(ref_tokens, cand_tokens)
        
        precision = lcs_len / len(cand_tokens)
        recall = lcs_len / len(ref_tokens)
        
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def evaluate_batch(self, references, candidates):
        if len(references) != len(candidates):
            raise ValueError("not the same number of references and candidates")
        
        bleu_scores = []
        rouge_scores = {'precision': [], 'recall': [], 'f1': []}
        
        for ref, cand in zip(references, candidates):
            bleu = self.calculate_bleu(ref, cand)
            bleu_scores.append(bleu)
            
            rouge = self.calculate_rouge_l(ref, cand)
            rouge_scores['precision'].append(rouge['precision'])
            rouge_scores['recall'].append(rouge['recall'])
            rouge_scores['f1'].append(rouge['f1'])
        
        return {
            'bleu': {
                'mean': np.mean(bleu_scores),
                'std': np.std(bleu_scores),
                'scores': bleu_scores
            },
            'rouge_l': {
                'precision': {
                    'mean': np.mean(rouge_scores['precision']),
                    'std': np.std(rouge_scores['precision'])
                },
                'recall': {
                    'mean': np.mean(rouge_scores['recall']),
                    'std': np.std(rouge_scores['recall'])
                },
                'f1': {
                    'mean': np.mean(rouge_scores['f1']),
                    'std': np.std(rouge_scores['f1'])
                }
            }
        }


class ParallelLogitsProcessor(LogitsProcessor):
    def __init__(self, processors: List[LogitsProcessor]):
        self.processors = processors

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        batch_size = input_ids.shape[0]
        batch_black_list = [None for _ in range(batch_size)]
        if batch_size != len(self.processors):
            raise ValueError(
                f"Batch size ({batch_size}) must match number of processors ({len(self.processors)})"
            )

        processed_scores = []
        for i in range(batch_size):
            
            processor = self.processors[i]
            
            sample_input_ids = input_ids[i].unsqueeze(0)  # [1, seq_len]
            sample_scores = scores[i].unsqueeze(0)        # [1, vocab_size]
            
            black_list = None
            if processor is None:
                modified_scores = sample_scores
            else:
                modified_scores, black_list = processor(sample_input_ids, sample_scores)
            batch_black_list[i] = black_list
            processed_scores.append(modified_scores[0])

        return torch.stack(processed_scores, dim=0), batch_black_list
    
class ChunkSearchGenerator():
    def __init__(self, args, tokenizer, model, dataset_name) -> None:
        self.dataset = dataset_name
        self.model_name = args.model
        self.mode = args.mode # watermark mode
        self.init_seed, self.dyna_seed, self.gamma, \
        self.delta, self.bl_type, self.num_beams, self.sampling_temp = args.initial_seed, args.dynamic_seed, args.gamma, args.delta, args.bl_type, args.num_beams, args.sampling_temp
        self.tokenizer = tokenizer
        self.model = model # language model
        self.device = next(self.model.parameters()).device
        self.all_token_ids = list(tokenizer.get_vocab().values())
        self.vocab_size = len(self.all_token_ids)
        self.seeding_scheme = args.seeding_scheme
        self.select_green_tokens = args.select_green_tokens
        self.K = args.K
        self.chunk_size = args.chunk_size
        self.alpha = args.alpha
        self.metrics = TextMetrics(tokenizer, self.device)
        
    @staticmethod
    def simple_hash_safe(x, seed=42):
        x = x & 0xFFFFFFFF
        large_prime = 15485863
        
        result = (x * large_prime + seed) ^ (seed << 1)

        return result
    
    def create_logits_processor_list(self, seeds):
        processors = []
        if self.mode == 'old':
            for seed in seeds:
                bl_processor = BlacklistLogitsProcessor(
                    bad_words_ids=None,
                    eos_token_id=self.tokenizer.eos_token_id,
                    vocab=self.all_token_ids,
                    vocab_size=self.vocab_size,
                    bl_proportion=1-self.gamma,
                    bl_logit_bias=self.delta,
                    bl_type=self.bl_type,
                    initial_seed=self.init_seed,
                    dynamic_seed=self.dyna_seed,
                    hash_seed=seed,
                    logger=None
                )
                processors.append(LogitsProcessorList([bl_processor]))
        elif self.mode == 'gpt':
            for seed in seeds:
                bl_processor = GPTWatermarkLogitsWarper(
                    vocab_size=self.vocab_size,
                    fraction=self.gamma,
                    strength=self.delta,
                    watermark_key=seed
                )
                processors.append(LogitsProcessorList([bl_processor]))
        elif self.mode == 'v2':
            for seed in seeds:
                bl_processor = WatermarkLogitsProcessor(
                    vocab=list(self.tokenizer.get_vocab().values()),
                    gamma=self.gamma,
                    delta=self.delta,
                    seeding_scheme=self.seeding_scheme,
                    select_green_tokens=self.select_green_tokens,
                    hash_seed=seed
                )
                processors.append(LogitsProcessorList([bl_processor]))
        return processors
    
    @staticmethod
    def calculate_score(green_num_fraction, metrics, weights=[0, 1]):
        return weights[0] * green_num_fraction + weights[1] * metrics
    
    def generate_chunk_unified(self, input_ids, chunk_size, watermark_processors):
        ultimate_chunk_size = chunk_size
        chunk_size += 3
        device = self.device
        input_ids = input_ids.to(device)
        
        all_processors = [None] + watermark_processors
        n_total = len(all_processors)
        
        expanded_input_ids = input_ids.repeat(n_total, 1)
        
        parallel_processor = ParallelLogitsProcessor(all_processors)
        if "intern" in self.model_name:
            eos_token_id = [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids(["<eoa>"])[0]
            ]
            
        else:
            eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = self.tokenizer.eos_token_id
        if self.mode == 'gpt':
            outputs, green_num_list = self.model.generate(
                expanded_input_ids,
                max_new_tokens=chunk_size,
                logits_processor=LogitsProcessorList([parallel_processor]),
                do_sample=True,
                top_k=0,
                top_p=0.9,
                use_cache=True,
                return_dict_in_generate=True,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                past_key_values=self.past_key_values,
                repetition_penalty=1.0
            )
        else:
            outputs, green_num_list = self.model.generate(
                expanded_input_ids,
                max_new_tokens=chunk_size,
                logits_processor=LogitsProcessorList([parallel_processor]),
                do_sample=True,
                top_k=0,
                temperature=self.sampling_temp,
                return_dict_in_generate=True,
                use_cache=True,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                past_key_values=self.past_key_values,
                repetition_penalty=1.0
            )
        input_length = input_ids.shape[-1]
        processed_outputs = []
        
        for i in range(n_total):
            b_green_num_list = green_num_list[i]
            available = False
            single_sequence = outputs.sequences[i]
            generated_tokens = single_sequence[input_length:]
            finished = False
            
            if "intern" in self.model_name:
                for eos_id in eos_token_id:
                    if eos_id in generated_tokens:
                        eos_pos = (generated_tokens == eos_id).nonzero(as_tuple=True)[0][0].item()
                        generated_tokens = generated_tokens[:eos_pos]
                        finished = True
                        break
            else:
                if eos_token_id in generated_tokens:
                    eos_pos = (generated_tokens == eos_token_id).nonzero(as_tuple=True)[0][0].item()
                    generated_tokens = generated_tokens[:eos_pos]
                    finished = True
                
            if len(generated_tokens) == 0:
                processed_outputs.append({
                    'text': '',
                    'tokens': [],
                    'sequences': input_ids,
                    'finished': False,
                    'is_standard': (i == 0),
                    'available': available,
                    'green_num_fraction': 0
                })
                continue
            
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            actual_tokens = self.tokenizer.encode(generated_text, return_tensors="pt", truncation=True, add_special_tokens=False)[0].to(device)
            actual_tokens = actual_tokens[:ultimate_chunk_size]
            fraction = 0
            if len(b_green_num_list) > 0:
                if len(actual_tokens) > 0:
                    valid_l = min(len(b_green_num_list), len(actual_tokens))
                    fraction = sum(b_green_num_list[:valid_l]) / len(actual_tokens)
                else:
                    fraction = 0
                
            if len(actual_tokens) == ultimate_chunk_size or finished:
                available = True
            else:
                available = False
                
            final_sequence = torch.cat((input_ids[0], actual_tokens), dim=0)
            
            processed_outputs.append({
                'text': generated_text,
                'tokens': actual_tokens.tolist(), 
                'sequences': final_sequence.unsqueeze(0),
                'finished': finished,
                'is_standard': (i == 0),
                'available': available,
                'green_num_fraction': fraction
            })
        
        return processed_outputs, outputs.past_key_values
    

    
    def generate(self, input_ids, max_new_tokens):
        self.past_key_values = None
        self.seed_pool = list(range(1, 5000))
        
        if self.mode == 'no':
            gen_config = GenerationConfig(
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True
            )
            outputs, _ = self.model.generate(
                input_ids=input_ids, generation_config=gen_config, 
            )
            scores = outputs.scores
            output_ids = outputs.sequences[0, -len(scores):]
            list_data = output_ids.cpu().tolist()
            # compute logprob for each token
            completions_tokens = []
            completions_logprob = 0
            for score, token in zip(scores, output_ids):
                logprobs = F.log_softmax(score[0], dim=-1)
                logprob = logprobs[token].item()
                completions_tokens.append({
                    'text': self.tokenizer.decode(token),
                    'logprob': logprob,
                })
                completions_logprob += logprob
            completions_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
            return completions_text, completions_tokens
        else:
            total_start_time = time.time()
            
            device = self.device
            current_input_ids = input_ids.clone().to(device)
            total_generated = 0
            chunk_count = 0
            
            all_output_ids = []
            
            total_generation_time = 0
            total_rouge_time = 0
            
            
            while total_generated < max_new_tokens:
                remaining_tokens = max_new_tokens - total_generated
                current_chunk_size = min(self.chunk_size, remaining_tokens)
                
                
                pre_seed = 1
                for pre_token in current_input_ids[0][-4:]:
                    pre_seed *= (pre_token.item() + 1)
                random.seed(ChunkSearchGenerator.simple_hash_safe(pre_seed))
                random.shuffle(self.seed_pool)
                cur_seeds = self.seed_pool[:self.K]
                
                processors_list = self.create_logits_processor_list(cur_seeds)
                single_processors = [proc[0] for proc in processors_list]
                
                generation_start_time = time.time()
                
                unified_outputs, self.past_key_values = self.generate_chunk_unified(
                    current_input_ids, 
                    current_chunk_size,
                    single_processors
                )
                generation_time = time.time() - generation_start_time
                total_generation_time += generation_time
                    
                
                standard_outputs = unified_outputs[0]
                watermark_outputs_list = unified_outputs[1:]
                
                if len(standard_outputs['tokens']) == 0:
                    break
                    
                
                rouge_start_time = time.time()

                
                candidate_list = []
                
                for i, watermark_outputs in enumerate(watermark_outputs_list):
                    if watermark_outputs is None or len(watermark_outputs['tokens']) == 0:
                        continue
                    rouge_l = self.metrics.calculate_rouge_l(
                        standard_outputs['text'], 
                        watermark_outputs['text']
                    )
                    f1 = ChunkSearchGenerator.calculate_score(green_num_fraction=watermark_outputs["green_num_fraction"], 
                                metrics=rouge_l['f1'],
                                weights=[self.alpha, 1 - self.alpha])
                    candidate_list.append((i, f1, watermark_outputs["available"]))
                        
                if len(candidate_list) == 0:
                    break
                    
                available_list = []
                for candidate in candidate_list:
                    # available
                    if candidate[2]:
                        # idx, f1
                        available_list.append((candidate[0], candidate[1]))
                if len(available_list) > 0:
                    best_f1 = -1.0
                    best_processor_idx = None
                    for candidate in available_list:
                        if best_f1 < candidate[1]:
                            best_f1 = candidate[1]
                            best_processor_idx = candidate[0]
                else:
                    best_f1 = -1.0
                    best_processor_idx = None
                    for candidate in candidate_list:
                        if best_f1 < candidate[1]:
                            best_f1 = candidate[1]
                            best_processor_idx = candidate[0]
                            
                best_outputs = watermark_outputs_list[best_processor_idx]
                
                rouge_time = time.time() - rouge_start_time
                total_rouge_time += rouge_time
                
                current_input_ids = best_outputs['sequences'].to(device)
                total_generated += len(best_outputs["tokens"])
                all_output_ids.extend(best_outputs["tokens"])
                chunk_count += 1
                
                if best_outputs['finished']:
                    break
                
                original_input_length = current_input_ids.shape[1] - len(best_outputs["tokens"])
                self.update_past_key_values_correctly(
                    best_processor_idx, 
                    current_input_ids, 
                    original_input_length
                )
        
            completions_tokens = []
            
            min_length = len(all_output_ids)
            
            for i in range(min_length):
                token = all_output_ids[i]
                
                completions_tokens.append({
                    'text': self.tokenizer.decode(token),
                })

            completions_text = self.tokenizer.decode(all_output_ids, skip_special_tokens=True)
            total_time = time.time() - total_start_time
            
            return completions_text, completions_tokens
        
    def update_past_key_values_correctly(self, best_processor_idx, current_input_ids, original_input_length):
        if self.past_key_values is None:
            return
        
        single_past = tuple(
            (
                layer_key[best_processor_idx + 1:best_processor_idx + 2, ...],
                layer_value[best_processor_idx + 1:best_processor_idx + 2, ...]
            )
            for layer_key, layer_value in self.past_key_values
        )
        
        truncated_past = tuple(
            (
                layer_k[:, :, :current_input_ids.shape[1] - 1, :].contiguous(),
                layer_v[:, :, :current_input_ids.shape[1] - 1, :].contiguous()
            )
            for layer_k, layer_v in single_past
        )
        
        expanded_past = tuple(
            (
                layer_k.repeat(self.K + 1, *[1] * (layer_k.ndim - 1)),
                layer_v.repeat(self.K + 1, *[1] * (layer_v.ndim - 1))
            )
            for layer_k, layer_v in truncated_past
        )
        
        if "internlm" in self.model_name.lower():
            self.past_key_values = expanded_past
        else:
            def create_dynamic_cache(past_key_values, num_layers):
                cache = DynamicCache()
                for layer_idx, (key, value) in enumerate(past_key_values):
                    cache.update(key, value, layer_idx)
                return cache
            
            num_layers = len(expanded_past)
            self.past_key_values = create_dynamic_cache(expanded_past, num_layers)
        
        del single_past, truncated_past, expanded_past