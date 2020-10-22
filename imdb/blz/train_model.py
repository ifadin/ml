from random import randint

import boto3
from sagemaker import image_uris, Session, TrainingInput, HyperparameterTuningJobAnalytics
from sagemaker.estimator import Estimator
from sagemaker.inputs import ShuffleConfig
from sagemaker.parameter import IntegerParameter, ContinuousParameter
from sagemaker.tuner import HyperparameterTuner

from imdb.blz.config import s3_bucket, iam_role, train_file, validation_file


def get_algorithm_image_uri(alg_name: str, region_name: str) -> str:
    return image_uris.retrieve(alg_name, region_name)


def get_estimator(container: str, iam_role: str, output_path: str, job_name: str = None,
                  session: Session = None) -> Estimator:
    return Estimator(container,
                     iam_role,
                     instance_count=1,
                     instance_type='ml.m5.xlarge',
                     volume_size=10,
                     output_path=output_path,
                     base_job_name=job_name,
                     sagemaker_session=session)


def get_tuner(estimator: Estimator, tuning_job_name: str) -> HyperparameterTuner:
    return HyperparameterTuner(estimator=estimator,
                               base_tuning_job_name=tuning_job_name,
                               objective_metric_name='validation:accuracy',
                               hyperparameter_ranges={
                                   'buckets': IntegerParameter(1000 * 1000, 1000 * 1000 * 5),
                                   'learning_rate': ContinuousParameter(0.01, 0.1),
                                   'min_count': IntegerParameter(2, 10),
                                   'vector_dim': IntegerParameter(32, 200),
                                   'word_ngrams': IntegerParameter(1, 3)
                               },
                               max_jobs=50,
                               max_parallel_jobs=10,
                               early_stopping_type='Auto')


def get_tuner_analytics(tuner: HyperparameterTuner) -> HyperparameterTuningJobAnalytics:
    return tuner.analytics().dataframe().sort_values(['FinalObjectiveValue'], ascending=False)


aws_region = boto3.Session().region_name

model_name = 'imdb-blztxt'
model_output = s3_output_location = f's3://{s3_bucket}/imdb/{model_name}/output'

img = get_algorithm_image_uri('blazingtext', aws_region)
e = get_estimator(img, iam_role, model_output, job_name=model_name)
e.set_hyperparameters(mode='supervised',
                      buckets=3500000,
                      learning_rate=0.1,
                      min_count=5,
                      vector_dim=100,
                      word_ngrams=2,
                      epochs=10,
                      early_stopping=True,
                      patience=4,
                      min_epochs=4)
e.fit({
    'train': TrainingInput(f's3://{s3_bucket}/{train_file}',
                           content_type='text/plain', shuffle_config=ShuffleConfig(randint(0, 1000))),
    'validation': TrainingInput(f's3://{s3_bucket}/{validation_file}',
                                content_type='text/plain')
})
