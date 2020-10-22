import os

s3_bucket = 'ci-dna-data-spool-test'
train_file = os.path.join('imdb', 'data', 'imdb.train')
validation_file = os.path.join('imdb', 'data', 'imdb.validation')
iam_role = 'arn:aws:iam::845883797226:role/sagemaker-test-role'
