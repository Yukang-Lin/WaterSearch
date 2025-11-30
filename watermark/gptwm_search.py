import hashlib
from typing import List

import numpy as np
import torch
# from transformers import LogitsWarper
# new version
from transformers import LogitsProcessor


class GPTWatermarkBase:
    """
    Base class for watermarking distributions with fixed-group green-listed tokens.

    Args:
        fraction: The fraction of the distribution to be green-listed.
        strength: The strength of the green-listing. Higher values result in higher logit scores for green-listed tokens.
        vocab_size: The size of the vocabulary.
        watermark_key: The random seed for the green-listing.
    """
    def __init__(self, fraction: float = 0.5, strength: float = 2.0, vocab_size: int = 32000, watermark_key: int = 0, device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        # 将设备信息保存为实例变量，方便后续使用
        self.device = device
        
        # 使用PyTorch的随机数生成器替代NumPy
        rng = torch.Generator(device=device)
        seed = self._hash_fn(watermark_key)  # 假设_hash_fn返回一个整数
        rng.manual_seed(seed)
        
        # 创建布尔掩码张量
        greenlist_size = int(fraction * vocab_size)
        blacklist_size = vocab_size - greenlist_size
        
        # 创建初始的布尔掩码（前greenlist_size为True，其余为False）
        mask = torch.cat([
            torch.ones(greenlist_size, dtype=torch.bool, device=device),
            torch.zeros(blacklist_size, dtype=torch.bool, device=device)
        ])
        
        # 打乱掩码
        shuffle_indices = torch.randperm(vocab_size, generator=rng, device=device)
        mask = mask[shuffle_indices]
        
        # 获取黑名单索引
        blacklist_indices = torch.where(~mask)[0]
        self.blacklist_tensor = blacklist_indices.to(dtype=torch.long)
        
        # 创建绿名单掩码（转换为float32）
        self.green_list_mask = mask.to(dtype=torch.float32)
        
        self.strength = strength
        self.fraction = fraction

    @staticmethod
    def _hash_fn(x: int) -> int:
        """solution from https://stackoverflow.com/questions/67219691/python-hash-function-that-returns-32-or-64-bits"""
        x = np.int64(x)
        return int.from_bytes(hashlib.sha256(x).digest()[:4], 'little')


class GPTWatermarkLogitsWarper(GPTWatermarkBase, LogitsProcessor):
    """
    LogitsWarper for watermarking distributions with fixed-group green-listed tokens.

    Args:
        fraction: The fraction of the distribution to be green-listed.
        strength: The strength of the green-listing. Higher values result in higher logit scores for green-listed tokens.
        vocab_size: The size of the vocabulary.
        watermark_key: The random seed for the green-listing.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.FloatTensor:
        """Add the watermark to the logits and return new logits."""
        # print("green_list_mask.shape is", self.green_list_mask.shape)
        
        watermark = self.strength * self.green_list_mask
        # print("scores shape is", scores.shape)
        # print("watermark shape is", watermark.shape)
        new_logits = scores + watermark.to(scores.device)
        
        return new_logits, self.blacklist_tensor.to(scores.device)


class GPTWatermarkDetector(GPTWatermarkBase):
    """
    Class for detecting watermarks in a sequence of tokens.

    Args:
        fraction: The fraction of the distribution to be green-listed.
        strength: The strength of the green-listing. Higher values result in higher logit scores for green-listed tokens.
        vocab_size: The size of the vocabulary.
        watermark_key: The random seed for the green-listing.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def _z_score(num_green: int, total: int, fraction: float) -> float:
        """Calculate and return the z-score of the number of green tokens in a sequence."""
        return (num_green - fraction * total) / np.sqrt(fraction * (1 - fraction) * total)

    def detect(self, sequence: List[int]) -> float:
        """Detect the watermark in a sequence of tokens and return the z value."""
        green_tokens = int(sum(self.green_list_mask[i] for i in sequence))

        return self._z_score(green_tokens, len(sequence), self.fraction)
