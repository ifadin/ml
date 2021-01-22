from functools import partial
from time import time
from typing import NamedTuple

import torch
from torch import Tensor, stack, FloatTensor, sigmoid, mean, no_grad
from torch.nn import Embedding, Module, Linear, ConstantPad1d, BCELoss
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchtext.experimental.datasets import IMDB
from torchtext.vocab import Vocab, Vectors, GloVe

PADDING_TOKEN = '<pad>'


class Classifier(Module):

    def __init__(self, seq_len: int, embeddings: Embedding):
        super().__init__()
        self.seq_len = seq_len
        self.embeddings = embeddings
        self.fc = Linear(self.seq_len, 1)

    def forward(self, x: Tensor):
        x = self.embeddings(x)
        x = mean(x, -1)
        x = self.fc(x)
        x = sigmoid(x).squeeze()
        return x


class Result(NamedTuple):
    loss: float
    correct: int
    examples: int
    duration: int

    @property
    def accuracy_score(self) -> float:
        return self.correct / self.examples


def calculate_sequence_length(data: Dataset, quantile: float) -> int:
    return int(torch.quantile(torch.FloatTensor(list({
        len(data[i][1]) for i in range(0, len(train_data))
    })), quantile).item())


def pad_batch(batch, pad_idx: int, max_sequence_length: int):
    labels, sequences = zip(*(b for b in batch if len(b[1]) <= max_sequence_length))
    labels = FloatTensor(labels)

    sequences = stack([
        ConstantPad1d((0, max_sequence_length - len(s)), pad_idx)(s)
        for s in sequences
    ])
    return labels, sequences


def get_embeddings(vocab: Vocab, vectors: Vectors, padding_token: str) -> Embedding:
    return Embedding.from_pretrained(torch.stack([vectors[token] for token in vocab.itos]),
                                     padding_idx=vocab[padding_token], freeze=True)


def evaluate(model: Classifier, data: DataLoader) -> Result:
    total_loss, correct_labels = 0.0, 0
    total_examples = 0

    time_start = time()
    model.eval()
    with no_grad():
        for b in data:
            labels, sentences = b
            total_examples += len(labels)
            outputs = model(sentences)

            total_loss += loss.item()
            correct_labels += ((outputs > 0.5) == labels).sum().item()
    time_end = time()
    duration = int(time_end - time_start)
    return Result(total_loss, correct_labels, total_examples, duration)


train_data, test_data = IMDB(ngrams=1)
v: Vocab = train_data.get_vocab()
g = GloVe('twitter.27B', 25)
e = get_embeddings(v, g, PADDING_TOKEN)

seq_len = calculate_sequence_length(train_data, 0.95)
collate_fn = partial(pad_batch, pad_idx=v[PADDING_TOKEN], max_sequence_length=seq_len)

model = Classifier(seq_len, e)
optimizer = SGD(model.parameters(), lr=0.1)
criterion = BCELoss()
scheduler = StepLR(optimizer, 1, gamma=0.9)

test_loader = DataLoader(test_data, shuffle=False, batch_size=1024, collate_fn=collate_fn)
for epoch in range(0, 50):
    model.train()
    loader = DataLoader(train_data, shuffle=True, batch_size=512, collate_fn=collate_fn)
    time_start = time()
    total_loss, correct_labels = 0.0, 0
    total_examples = 0
    for b in loader:
        labels, sentences = b
        total_examples += len(labels)

        optimizer.zero_grad()
        outputs = model(sentences)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct_labels += ((outputs > 0.5) == labels).sum().item()
    time_end = time()
    duration = int(time_end - time_start)
    r = Result(total_loss, correct_labels, total_examples, duration)
    r_test = evaluate(model, test_loader)
    print(f'epoch: {epoch} ({r.duration}/{r_test.duration} sec) loss: {r.loss:.2f} '
          f'acc_train: {r.accuracy_score * 100:.2f}% '
          f'acc_test: {r_test.accuracy_score * 100:.2f}%')
