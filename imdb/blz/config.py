import os

s3_bucket = 'ci-dna-data-spool-test'
train_file = os.path.join('imdb', 'data', 'imdb.train')
test_file = os.path.join('imdb', 'data', 'imdb.test')
iam_role = 'arn:aws:iam::845883797226:role/sagemaker-test-role'
