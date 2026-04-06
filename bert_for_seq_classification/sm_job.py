import os
import argparse

import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.debugger import TensorBoardOutputConfig

session = sagemaker.Session()
bucket = session.default_bucket()

model_training = os.path.basename(os.getcwd())


parser = argparse.ArgumentParser()

parser.add_argument('--data', type=str, default="data/00_sft",
                    help="dataset to use under data/")
parser.add_argument('--epochs', type=int, default=50,
                    help='epoch limit')
parser.add_argument('--batch-size', type=int, default=64,
                    help="batch size")
parser.add_argument('--lr', type=float, default=0.05,
                    help="learning rate")
parser.add_argument('--accel', action='store_true',
                    help='accelerator to use')
parser.add_argument('--tb-dir', type=str, default='/opt/ml/output/tensorboard',
                    help="tensorboard data directory")

args = parser.parse_args()

tensorboard_output_config = TensorBoardOutputConfig(
    s3_output_path=f's3://{bucket}/{model_training}/tensorboard',
    container_local_output_path=args.tb_dir
)

hyperparameters = {
    'epochs': 20,
    'batch-size': 16,
    'learning-rate': 0.0000001
}

pytorch_estimator = PyTorch(image_uri="", ## TODO: add custom image uri
                            entry_point='main.py',
                            source_dir='.',
                            role="AmazonSageMaker-ExecutionRole-20251012T110132",
                            instance_type="ml.m5.xlarge", # 'ml.g4dn.2xlarge',
                            instance_count=1,
                            output_path=f's3://{bucket}/{model_training}',
                            tensorboard_output_config=tensorboard_output_config,
                            dependencies=[
                                'model.py',
                                'requirements.txt',
                                'class_set.json',
                                "lib/dataset.py"
                            ],
                            hyperparameters=hyperparameters)
pytorch_estimator.fit({'train': f's3://{bucket}/{model_training}/{args.data}/train',
                        'test': f's3://{bucket}/{model_training}/{args.data}/test'})
# TODO: deploy and eval
# pytorch_estimator.deploy()

