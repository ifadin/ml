import csv
import multiprocessing
import os
import string
from itertools import chain
from multiprocessing import Pool
from random import shuffle, seed
from typing import Tuple, List

import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from tqdm.auto import tqdm

from imdb.blz.config import train_file, s3_bucket, test_file
from imdb.data import load_reviews, upload_to_s3

contractions = {
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "I'd": "I would",
    "I'd've": "I would have",
    "I'll": "I will",
    "I'll've": "I will have",
    "I'm": "I am",
    "I've": "I have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "that'd": "that had",
    "that'd've": "that would have",
    "there'd": "there would",
    "there'd've": "there would have",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when've": "when have",
    "where'd": "where did",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who've": "who have",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have",
    "n't": 'not',
    "'ll": 'will',
    "'re": 'are',
    "'ve": 'have',
    "'d": 'would',
    "'d've": 'would have',
    "'ll've": 'will have',
    "'m": 'am'
}


def to_blz_format(data_row: Tuple[str, int]) -> List[str]:
    review, label = data_row
    review = BeautifulSoup(review, 'html.parser').get_text()  # Remove HTML tags
    stoplist = (set(stopwords.words('english')) - {'not', 'no'}) | set(string.punctuation) | {
        "''", '``', '..', '...', '....', '.....', '......', '.......', '--', "'s", '“', '”'
    }
    lem = nltk.WordNetLemmatizer()

    return [f'__label__{label}'] + list(
        lem.lemmatize(i) for i in chain.from_iterable((
            [w] if w not in contractions else contractions[w].split()
            for w in nltk.word_tokenize(review.lower())))
        if i not in stoplist
    )


def create_input_file(reviews: List[Tuple[str, int]], output_file: str) -> str:
    pool = Pool(processes=multiprocessing.cpu_count())
    transformed_rows = pool.map(to_blz_format, tqdm(reviews))
    pool.close()
    pool.join()

    shuffle(transformed_rows)
    with open(output_file, 'w') as outfile:
        csv_writer = csv.writer(outfile, delimiter=' ', lineterminator='\n')
        csv_writer.writerows(transformed_rows)

    return output_file


nltk.download('stopwords')
nltk.download('wordnet')
seed(1234)

train_reviews = load_reviews(os.path.join('imdb', 'data', 'train'))
test_data = load_reviews(os.path.join('imdb', 'data', 'test'))
shuffle(test_data)
test_reviews, additional_train_reviews = (test_data[:len(test_data) // 2],
                                          test_data[len(test_data) // 2:])

create_input_file(train_reviews + additional_train_reviews, train_file)
upload_to_s3(train_file, s3_bucket)

create_input_file(test_reviews, test_file)
upload_to_s3(test_file, s3_bucket)
