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
from nltk.tokenize import sent_tokenize

terminators = ['。', '？', '！', '.', '?', '!']

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
            raise ValueError("References and candidates must have the same length.")
        
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
    

class SentenceSearchGenerator():
    
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
        self.seed_pool = list(range(1, 5000))
        self.K = args.K
        self.threshold = 30
        self.full_generated = False
        self.use_new_seed = True
        self.alpha = args.alpha
        self.metrics = TextMetrics(tokenizer, self.device)
        
    @staticmethod
    def simple_hash_safe(x, seed=42):
        x = x & 0xFFFFFFFF  # 保持32位
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
    
    

    def split_sentences(self, text):
        import re
        # 正则匹配每个句子：非贪婪匹配到一个结束标点
        sentence_list = re.findall(r'[^。！？.!?]*[。！？.!?]', text, flags=re.U)
        # 剩余可能没有结束标点的部分
        residual = re.sub(r'[^。！？.!?]*[。！？.!?]', '', text, flags=re.U).strip()
        if residual:
            sentence_list.append(residual)
        return [s.strip() for s in sentence_list if s.strip()]
    
    def generate_sentence_unified(self, input_ids, watermark_processors, max_new_tokens):
        device = self.device
        input_ids = input_ids.to(device)
        
        all_processors = [None] + watermark_processors
        n_total = len(all_processors)
        
        expanded_input_ids = input_ids.repeat(n_total, 1)
        

        parallel_processor = ParallelLogitsProcessor(all_processors)
        
        end_word = ""
        if "internlm" in self.model_name:
            eos_token_id = [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids(["<eoa>"])[0]
            ]
            end_word = "<eoa>"
        else:
            eos_token_id = self.tokenizer.eos_token_id
            
            
        if self.mode == 'gpt':
            outputs, green_num_list = self.model.generate(
                expanded_input_ids,
                max_new_tokens=max_new_tokens,
                logits_processor=LogitsProcessorList([parallel_processor]),
                do_sample=True,
                top_k=0,
                top_p=0.9,
                return_dict_in_generate=True,
                output_scores=True,
                eos_token_id=eos_token_id,
                past_key_values=self.past_key_values,
                pad_token_id=self.tokenizer.eos_token_id
            )
        else:
            outputs, green_num_list = self.model.generate(
                expanded_input_ids,
                max_new_tokens=max_new_tokens,
                logits_processor=LogitsProcessorList([parallel_processor]),
                do_sample=True,
                top_k=0,
                temperature=self.sampling_temp,
                return_dict_in_generate=True,
                output_scores=True,
                eos_token_id=eos_token_id,
                past_key_values=self.past_key_values,
                pad_token_id=self.tokenizer.eos_token_id
            )
        self.past_key_values = outputs.past_key_values
        input_length = input_ids.shape[-1]
        
        processed_outputs = []
        for i in range(n_total):
            b_green_num_list = green_num_list[i]
            single_sequence = outputs.sequences[i]
            original_generated_tokens = single_sequence[input_length:]
            
            full_generated = False
            if len(original_generated_tokens) == max_new_tokens:
                full_generated = True
                
            original_generated_text = self.tokenizer.decode(original_generated_tokens, skip_special_tokens=True)
            sentences = self.split_sentences(original_generated_text)
            
            if not sentences or sentences[0] == end_word:
                sentence_tokens = []
                first_sentence = ""
                finished = True
            elif sentences[0].strip() != original_generated_text.strip():
                first_sentence = sentences[0]
                full_generated = False
                sentence_tokens = self.tokenizer.encode(first_sentence, add_special_tokens=False)
            else:
                if original_generated_text[-1] in terminators:
                    full_generated = False
                
                first_sentence = original_generated_text
                if not isinstance(original_generated_tokens, list):
                    sentence_tokens = original_generated_tokens.tolist()
                else:
                    sentence_tokens = original_generated_tokens
            
            if len(sentence_tokens) == 0:
                processed_outputs.append({
                    'text': '',
                    'tokens': [],
                    'scores': [],
                    'sequences': input_ids,
                    'finished': False,
                    'is_standard': (i == 0),
                    'full_generated': full_generated,
                    'green_num_fraction': 0
                })
                
                continue
            
            fraction = 0
            if len(b_green_num_list) > 0:
                valid_l = min(len(b_green_num_list), len(sentence_tokens))
                fraction = sum(b_green_num_list[:valid_l]) / len(sentence_tokens)
            
            finished = False
            if "internlm" in self.model_name:
                if any(tok in sentence_tokens for tok in eos_token_id):
                    eos_positions = [
                        sentence_tokens.index(tok)
                        for tok in eos_token_id
                        if tok in sentence_tokens
                    ]
                    eos_pos = min(eos_positions)
                    sentence_tokens = sentence_tokens[:eos_pos + 1]
                    finished = True
            else:
                if eos_token_id in sentence_tokens:
                    eos_pos = sentence_tokens.index(eos_token_id)
                    sentence_tokens = sentence_tokens[:eos_pos + 1]
                    finished = True
            
            if outputs.scores:
                max_score_len = len(outputs.scores)
                actual_token_len = len(original_generated_tokens)
                safe_len = min(len(sentence_tokens), max_score_len, actual_token_len)
                
                sentence_scores = []
                for j in range(safe_len):
                    try:
                        sentence_scores.append(outputs.scores[j][i])
                    except (IndexError, TypeError) as e:
                        break
            else:
                sentence_scores = []
            
            sentence_tokens_tensor = torch.tensor(sentence_tokens, device=device, dtype=expanded_input_ids.dtype)
            new_sequence = torch.cat([
                expanded_input_ids[i], 
                sentence_tokens_tensor
            ]).unsqueeze(0)
            
            processed_outputs.append({
                'text': first_sentence,
                'tokens': sentence_tokens,
                'scores': sentence_scores,
                'sequences': new_sequence,
                'finished': finished,
                'is_standard': (i == 0),
                'full_generated': full_generated,
                'green_num_fraction': fraction
            })
        return processed_outputs
    
    @staticmethod
    def calculate_score(green_num_fraction, metrics, weights=[0.7, 0.3]):
        return weights[0] * green_num_fraction + weights[1] * metrics
        
    
    def generate(self, input_ids, max_new_tokens):
        self.past_key_values = None
        self.seed_pool = list(range(1, 5000))
        self.use_new_seed = True
        if self.mode == 'no':
            outputs, _ = self.model.generate(
                input_ids, max_new_tokens=max_new_tokens,
            )
            scores = outputs.scores
            output_ids = outputs.sequences[0, -len(scores):]
            # compute logprob for each token
            completions_tokens = []
            completions_logprob = 0
            for score, token in zip(scores, output_ids, strict=True):
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
            sentence_count = 0
            
            all_scores = []
            all_output_ids = []
            
            total_generation_time = 0
            total_rouge_time = 0
            
            
            # 这里是为了保证每一句话的所有token使用的种子是一致的
            while total_generated < max_new_tokens:
                
                remaining_tokens = max_new_tokens - total_generated
                current_max_tokens = min(self.threshold, remaining_tokens)
                
                if self.use_new_seed:
                    random.seed(42)
                    random.shuffle(self.seed_pool)
                cur_seeds = self.seed_pool[:self.K]
                
                processors_list = self.create_logits_processor_list(cur_seeds)
                single_processors = [proc[0] for proc in processors_list]
                
                generation_start_time = time.time()
                
                unified_outputs = self.generate_sentence_unified(
                    current_input_ids,
                    single_processors,
                    current_max_tokens
                )
                generation_time = time.time() - generation_start_time
                total_generation_time += generation_time
                standard_outputs = unified_outputs[0]
                watermark_outputs_list = unified_outputs[1:]
                
                if len(standard_outputs['tokens']) == 0:
                    if self.full_generated:
                        self.use_new_seed = True
                        self.full_generated = False
                    else:
                        self.use_new_seed = False
                    break
                    
                
                rouge_start_time = time.time()
                
                best_sc = 0
                best_outputs = None
                best_processor_idx = None
                successful_generations = 0
                full_generated = False
                for i, watermark_outputs in enumerate(watermark_outputs_list):
                    if watermark_outputs is None or len(watermark_outputs['tokens']) == 0:
                        continue
                    
                    try:
                        rouge_l = self.metrics.calculate_rouge_l(
                            standard_outputs['text'], 
                            watermark_outputs['text']
                        )
                        f1 = rouge_l['f1']
                        sc = SentenceSearchGenerator.calculate_score(green_num_fraction=watermark_outputs["green_num_fraction"], 
                                                       metrics=f1,
                                                       weights=[self.alpha, 1 - self.alpha])
                        successful_generations += 1
                        if sc > best_sc:
                            best_sc = sc
                            best_outputs = watermark_outputs
                            best_processor_idx = i
                            full_generated = watermark_outputs["full_generated"]
                        
                            
                    except Exception as e:
                        continue
                    
                
                rouge_time = time.time() - rouge_start_time
                total_rouge_time += rouge_time
                
                if best_outputs is None or successful_generations == 0:
                    if self.full_generated and total_generated > 0:
                        self.use_new_seed = True
                        self.full_generated = False
                    else:
                        self.use_new_seed = False
                    break
                
                if 'scores' in best_outputs and best_outputs['scores']:
                    all_scores.extend(best_outputs['scores'])
                    all_output_ids.extend(best_outputs['tokens'])
                
                
                single_past = tuple(
                    (
                        layer_key[best_processor_idx + 1:best_processor_idx + 2, ...],   # shape: (1, num_heads, seq_len, head_dim)
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

                self.past_key_values = tuple(
                    (
                        layer_k.repeat(self.K + 1, *[1] * (layer_k.ndim - 1)),
                        layer_v.repeat(self.K + 1, *[1] * (layer_v.ndim - 1))
                    )
                    for layer_k, layer_v in truncated_past
                )
                
                
                current_input_ids = best_outputs['sequences'].to(device)
                total_generated += len(best_outputs['tokens'])
                sentence_count += 1
                
                
                if len(best_outputs['tokens']) != current_max_tokens or total_generated == max_new_tokens or any(t in best_outputs["text"] for t in terminators):
                    self.use_new_seed = True
                else:
                    self.use_new_seed = False
                
                
                self.full_generated = full_generated
                
                if best_outputs['finished']:
                    self.use_new_seed = True
                    break
                
        
            completions_tokens = []
            completions_logprob = 0
            
            min_length = min(len(all_scores), len(all_output_ids))
            for i in range(min_length):
                score = all_scores[i]
                token = all_output_ids[i]
                logprobs = F.log_softmax(score, dim=-1)
                logprob = logprobs[token].item()
                
                completions_tokens.append({
                    'text': self.tokenizer.decode(token),
                    'logprob': logprob
                })
                completions_logprob += logprob
            completions_text = self.tokenizer.decode(all_output_ids, skip_special_tokens=True)
            total_time = time.time() - total_start_time
            
            return completions_text, completions_tokens
        