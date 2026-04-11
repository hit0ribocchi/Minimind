from torch.utils.data import Dataset
import torch
import os
import random
import json
from datasets import load_dataset, Features, Sequence, Value
class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 使用 HuggingFace datesets 的惰性加载，避免一次性读入大文件 
        self.samples = load_dataset('json', data_files=data_path, split='train')

    def __len__(self):
        return len(self.samples)
    
    # 我们拿到的是 jsonl 中的每一行
    def __getitem__(self, index):
        sample = self.samples[index]
        tokens = self.tokenizer(
            str(sample['text']), # jsonl中的“text”字段保存了文本内容
            add_special_tokens=False, # 不加入特殊token
            max_length=self.max_length - 2, # 自己加上EOS和BOS
            truncation=True # 如果长度超过了max_length，自动剪切
            ).input_ids
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id] # 加入EOS和BOS
        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens)) # 填充到 max_length
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        return input_ids, labels