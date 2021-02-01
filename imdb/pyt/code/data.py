from typing import List, Iterable

import torch
from torch.utils.data import Dataset


class IMDBDataset(Dataset):

    def __init__(self, data_file: str, keep_quantile: float = None) -> None:
        self.data_file = data_file
        self.keep_quantile = keep_quantile
        self.data: List[List[int]] = []

    def __getitem__(self, index):
        return self.data[index][0], self.data[index][1:]

    def __len__(self) -> int:
        return len(self.data)

    def load(self):
        with open(self.data_file) as f:
            for line in f:
                self.data.append(list(map(int, line.split(','))))
        if self.keep_quantile:
            length_threshold = calculate_sequence_threshold(self.data, self.keep_quantile)
            self.data = [d for d in self.data if len(d) <= length_threshold]
        return self


def calculate_sequence_threshold(data: Iterable[List], quantile: float) -> int:
    return int(torch.quantile(torch.FloatTensor([
        len(d) for d in data
    ]), quantile).item())


def load_imdb_vocab(vocab_file: str) -> List[str]:
    with open(vocab_file, newline='\n') as f:
        return list(line.strip() for line in f)
