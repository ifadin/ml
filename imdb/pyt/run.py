import os

import boto3
from sagemaker import image_uris, Session, TrainingInput
from sagemaker.parameter import CategoricalParameter
from sagemaker.pytorch import PyTorch
from sagemaker.tuner import HyperparameterTuner

from imdb.pyt.code.config import S3_DATA_PATH, get_config, IAM_ROLE, S3_CACHE_PATH


def get_estimator(cfg, image_uri: str, iam_role: str, output_path: str, job_name: str = None,
                  session: Session = None) -> PyTorch:
    return PyTorch('train.py',
                   source_dir=os.path.join('imdb', 'pyt', 'code'),
                   image_uri=image_uri,
                   role=iam_role,
                   instance_count=1,
                   instance_type=cfg.instance_type,
                   volume_size=20,
                   output_path=output_path,
                   base_job_name=job_name,
                   sagemaker_session=session,
                   metric_definitions=[
                       {'Name': 'train:accuracy', 'Regex': r'#train_accuracy: (\S+)%'},
                       {'Name': 'test:accuracy', 'Regex': r'#test_accuracy: (\S+)%'}
                   ],
                   hyperparameters={
                       'epochs': cfg.epochs,
                       'batch-size': cfg.batch_size,
                       'rnn-hidden-size': cfg.rnn_hidden_size,
                       'rnn-num-layers': cfg.rnn_num_layers
                   })


def get_hyper_tuner(estimator: PyTorch, tuning_job_name: str) -> HyperparameterTuner:
    return HyperparameterTuner(estimator=estimator,
                               base_tuning_job_name=tuning_job_name,
                               metric_definitions=estimator.metric_definitions,
                               objective_metric_name='train:accuracy',
                               hyperparameter_ranges={
                                   'batch-size': CategoricalParameter([2, 8, 16, 32, 64, 128, 256, 512, 1024, 2048])
                               },
                               max_jobs=10,
                               max_parallel_jobs=10,
                               early_stopping_type='Auto')


aws_region = boto3.Session().region_name

model_name = 'imdb-pyt'
model_output = os.path.join(S3_DATA_PATH, 'output')

cfg = get_config()
img = image_uris.retrieve('pytorch', aws_region, version='1.6', py_version='py3',
                          image_scope='training', instance_type=cfg.instance_type)
e = get_estimator(cfg, img, IAM_ROLE, model_output, job_name=model_name)
tuner = get_hyper_tuner(e, e.base_job_name)

e.fit({
    'data': TrainingInput(S3_DATA_PATH),
    'cache': TrainingInput(S3_CACHE_PATH)
})
