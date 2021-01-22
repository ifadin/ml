import glob
import logging
import os
from typing import List, Tuple

import boto3
from botocore.exceptions import ClientError


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
