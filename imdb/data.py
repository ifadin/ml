import csv
import glob
import logging
import os
from collections import OrderedDict
from operator import itemgetter
from typing import List, Tuple, Dict, Mapping, Iterable

import boto3
from botocore.exceptions import ClientError
from torchtext.data import get_tokenizer
from tqdm import tqdm


class IMDBDataSerializer:

    def __init__(self, split_ratio: float = 0.5,
                 data_dir: str = 'data',
                 out_dir: str = 'out',
                 max_sentence_length: int = None) -> None:
        self.split_ratio = split_ratio
        self.data_dir = data_dir
        self.out_dir = out_dir
        self.max_sentence_length = max_sentence_length
        self.tokenizer = get_tokenizer('basic_english')

        self.labels: List[int] = []
        self.reviews = []
        self.stoi = OrderedDict()

        self.vocab_file = os.path.join(self.out_dir, 'imdb.vocab')
        self.train_file = os.path.join(self.out_dir, 'imdb.train.csv')
        self.test_file = os.path.join(self.out_dir, 'imdb.test.csv')

    def load(self):
        counter = {}
        reviews = []

        for data_type in ['train', 'test']:
            for sentiment in ['pos', 'neg']:
                files_path = os.path.join(self.data_dir, data_type, sentiment, '*.txt')
                for f in tqdm(glob.glob(files_path)):
                    with open(f) as review:
                        tokens = self.transform_line(review.read().strip())
                        if not self.max_sentence_length or len(tokens) <= self.max_sentence_length:
                            for token in tokens:
                                self.update_counter(counter, token)

                            reviews.append(tokens)
                            self.labels.append(1 if sentiment == 'pos' else 0)

        self.stoi = self.get_stoi(counter)
        self.reviews = self.get_index_based_array(reviews, self.stoi)

        return self

    def transform_line(self, line: str) -> List[str]:
        return self.tokenizer(line)

    @staticmethod
    def update_counter(counter: Dict[str, int], value: str) -> Dict[str, int]:
        if value in counter:
            counter[value] += 1
        else:
            counter[value] = 1

        return counter

    @staticmethod
    def get_stoi(counter: Dict[str, int]) -> Mapping[str, int]:
        # 0 is reserved for padding index
        return OrderedDict([(key, index + 1) for index, (key, _) in
                            enumerate(sorted(counter.items(), key=itemgetter(1), reverse=True))])

    @staticmethod
    def get_index_based_array(reviews: List[List[str]], stoi: Mapping[str, int]) -> List[List[int]]:
        return [[stoi[token] for token in line] for line in reviews]

    def save(self):
        def save_data(labels: Iterable[int], reviews: Iterable[List[int]], file_name: str):
            with open(file_name, 'w') as f:
                w = csv.writer(f)
                for d in ([label] + reviews for label, reviews in zip(labels, reviews)):
                    w.writerow(d)

        split_index = int(self.split_ratio * len(self.labels))
        inversed_split_index = -(len(self.labels) - split_index)

        self.save_vocab(self.vocab_file)
        save_data(self.labels[0:split_index], self.reviews[0:split_index], self.train_file)
        save_data(self.labels[inversed_split_index:], self.reviews[inversed_split_index:], self.test_file)

    def save_vocab(self, file_name: str):
        with open(file_name, 'w') as f:
            f.write('\n'.join(self.stoi.keys()))


def load_reviews(data_dir: str) -> List[Tuple[str, int]]:
    reviews = []

    for sentiment in ['pos', 'neg']:
        files_path = os.path.join(data_dir, sentiment, '*.txt')
        for f in glob.glob(files_path):
            with open(f) as review:
                reviews.append((review.read().strip(), 1 if sentiment == 'pos' else 0))

    return reviews


def upload_to_s3(file_name, bucket, object_name=None, s3_client=boto3.client('s3')):
    if object_name is None:
        object_name = file_name

    try:
        s3_client.upload_file(file_name, bucket, object_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True
