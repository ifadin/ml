import os
from argparse import Namespace, ArgumentParser

CACHE_PATH = os.environ.get('SM_CHANNEL_CACHE', '.vector_cache')
DATA_PATH = os.environ.get('SM_CHANNEL_DATA', os.path.join('imdb', 'pyt', 'data'))

TRAIN_DATAFILE = os.path.join(DATA_PATH, 'imdb.train.csv')
TEST_DATAFILE = os.path.join(DATA_PATH, 'imdb.test.csv')
VOCAB_DATAFILE = os.path.join(DATA_PATH, 'imdb.vocab')

S3_DATA_BUCKET = 'ci-dna-data-spool-test'
S3_DATA_PATH = f's3://{S3_DATA_BUCKET}/{DATA_PATH}'
S3_CACHE_PATH = f's3://{S3_DATA_BUCKET}/{DATA_PATH}/.vector_cache'
IAM_ROLE = 'arn:aws:iam::845883797226:role/sagemaker-test-role'


def get_config() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--rnn-hidden-size', type=int, default=32)
    parser.add_argument('--rnn-num-layers', type=int, default=1)

    parser.add_argument('--instance-type', type=str, default='ml.m5.xlarge')
    parser.add_argument('--cache', type=str, default=CACHE_PATH)
    parser.add_argument('--train', type=str, default=TRAIN_DATAFILE)
    parser.add_argument('--test', type=str, default=TEST_DATAFILE)
    parser.add_argument('--vocab', type=str, default=VOCAB_DATAFILE)

    args, _ = parser.parse_known_args()

    return args
