from functools import partial, reduce
from time import time
from typing import NamedTuple, Iterable, List

import torch
from torch import Tensor, stack, FloatTensor, sigmoid, no_grad, mean, cat
from torch.nn import Embedding, Module, Linear, ConstantPad1d, BCELoss, LSTM
from torch.nn.utils.rnn import pad_sequence
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import StepLR
from torch.types import Device
from torch.utils.data import DataLoader
from torchtext.vocab import GloVe, Vectors
from tqdm import tqdm

from config import get_config
from data import IMDBDataset, load_imdb_vocab


class Classifier(Module):

    def __init__(self, embeddings: Embedding, rnn_hidden_size: int = 32, rnn_num_layers: int = 1):
        super().__init__()

        self.embeddings = embeddings
        self.rnn = LSTM(self.embeddings.embedding_dim, rnn_hidden_size, rnn_num_layers,
                        batch_first=True, bidirectional=True)
        self.fc = Linear(self.rnn.hidden_size * 2 * 2, 1)

    def forward(self, x: Tensor):
        x: Tensor = self.embeddings(x)
        x, hidden = self.rnn(x)
        avg_pool = mean(x, 1)
        max_pool, _ = torch.max(x, 1)
        x = cat((avg_pool, max_pool), 1)
        x = self.fc(x)
        x = sigmoid(x).squeeze()
        return x, None


class Result(NamedTuple):
    loss: float
    correct: int
    examples: int
    duration: int

    @property
    def accuracy_score(self) -> float:
        return self.correct / self.examples


def pad_batch(batch, pad_idx: int):
    labels, sequences = zip(*batch)

    return torch.FloatTensor(labels), pad_sequence(list(map(torch.tensor, sequences)), batch_first=True,
                                                   padding_value=pad_idx)


def pad_batch_manual(batch, pad_idx: int, max_sequence_length: int):
    labels, sequences = zip(*(b for b in batch if len(b[1]) <= max_sequence_length))
    labels = FloatTensor(labels)

    sequences = stack([
        ConstantPad1d((0, max_sequence_length - len(s)), pad_idx)(s)
        for s in sequences
    ])
    return labels, sequences


def get_embeddings(vocab: Iterable[str], vectors: Vectors, padding_idx: int) -> Embedding:
    pad_vector = vectors[padding_idx]
    return Embedding.from_pretrained(torch.stack([pad_vector] + [vectors[token] for token in vocab]),
                                     padding_idx=padding_idx, freeze=True)


def train(model: Classifier, loader: DataLoader, criterion: Module, device: Device,
          optimizer: Optimizer = None) -> Result:
    evaluation_mode = optimizer is None
    total_loss, correct_labels = 0.0, 0
    total_examples = 0

    time_start = time()

    model.eval() if evaluation_mode else model.train()
    for labels, sentences in tqdm(loader, desc=('Evaluating' if evaluation_mode else 'Training')):
        labels, sentences = labels.to(device), sentences.to(device)
        total_examples += len(labels)

        if not evaluation_mode:
            optimizer.zero_grad()
        outputs, _ = model(sentences)
        loss = criterion(outputs, labels)

        if not evaluation_mode:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        correct_labels += ((outputs > 0.5) == labels).sum().item()
    time_end = time()
    duration = int(time_end - time_start)
    return Result(total_loss, correct_labels, total_examples, duration)


class EarlyStopper:

    def __init__(self, state: List[float] = None, patience: int = 3) -> None:
        self.state: List[float] = state if state else []
        self.patience = patience

    def add(self, value: float):
        self.state = self.state[-2:] + [value]

    def get_condition(self) -> bool:
        if not self.state or len(self.state) < self.patience:
            return False
        return reduce(lambda acc, v: (v, acc[1] and v <= acc[0]), self.state, (self.state[0], True))[1]

    def evaluate(self):
        if self.get_condition():
            self.stop()

    def stop(self):
        print(f'Early stopping activated. Last 3 scores: {self.state}')
        exit()


if __name__ == '__main__':
    stopper = EarlyStopper()
    config = get_config()

    time_start = time()
    train_data, test_data = IMDBDataset(config.train, keep_quantile=0.9).load(), IMDBDataset(config.test).load()
    vocab = load_imdb_vocab(config.vocab)
    print(f'train/test: {len(train_data)}/{len(test_data)} ({int(time() - time_start)} sec)')

    time_start = time()

    padding_index = 0
    e = get_embeddings(vocab, GloVe('42B', cache=config.cache), padding_index)
    print(f'embeddings: {e.num_embeddings}x{e.embedding_dim} ({int(time() - time_start)} sec)')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Classifier(e, config.rnn_hidden_size, config.rnn_num_layers).to(device)
    optimizer = SGD(model.parameters(), lr=0.1)
    criterion = BCELoss()
    scheduler = StepLR(optimizer, 1, gamma=0.9)

    print(f'parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    collate_fn = partial(pad_batch, pad_idx=padding_index)
    test_loader = DataLoader(test_data, shuffle=False, batch_size=64, collate_fn=collate_fn)
    for epoch in range(0, config.epochs):
        loader = DataLoader(train_data, shuffle=True, batch_size=config.batch_size, collate_fn=collate_fn)
        r = train(model, loader, criterion, device, optimizer)
        with no_grad():
            r_test = train(model, test_loader, criterion, device)
        print(f'epoch: {epoch} ({r.duration}/{r_test.duration} sec) loss: {r.loss:.2f} '
              f'#train_accuracy: {r.accuracy_score * 100:.2f}% '
              f'#test_accuracy: {r_test.accuracy_score * 100:.2f}%')
        stopper.add(r.accuracy_score)
        stopper.evaluate()
