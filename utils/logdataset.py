#© 2023 Nokia
#Licensed under the Creative Commons Attribution Non Commercial 4.0 International license
#SPDX-License-Identifier: CC-BY-NC-4.0
#

from torch.utils.data import Dataset
import torch
from collections import defaultdict
import math

class LogDataset(Dataset):
    def __init__(self, encodings, pad_token_id):
        self.inputs = encodings
        self.attention_masks = [[1] * len(i) for i in encodings]
        self.pad_token_id = pad_token_id
        self.length = [len(i) for i in encodings]

    def __getitem__(self, idx):
        return self.inputs[idx], self.attention_masks[idx], self.length[idx]

    def __len__(self):
        return len(self.inputs)

    def collate_fn(self, batch):
        inputs, masks, seq_len = zip(*batch)
        max_len = max(seq_len)
        inputs = [torch.cat((torch.tensor(i), torch.tensor([self.pad_token_id] * (max_len - len(i)), dtype=torch.long))) for i in inputs]
        masks = [torch.cat((torch.tensor(i), torch.zeros(max_len - len(i), dtype=torch.long))) for i in masks]
        return torch.stack(inputs), torch.stack(masks), torch.tensor(seq_len, dtype=torch.long)