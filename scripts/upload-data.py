import os
import sys
import argparse

import boto3
import sagemaker
from sagemaker.s3 import S3Uploader


parser = argparse.ArgumentParser()
parser.add_argument("model_training", help="The model training")
parser.add_argument("dataset", help="The dataset to upload")
args, _ = parser.parse_known_args()

all_files = os.listdir(os.getcwd())
all_directories = [entry for entry in all_files if os.path.isdir(os.path.join(os.getcwd(), entry))]
directories = [file for file in all_directories if not file.startswith(".") and file != "scripts"]

prefix = args.model_training
if prefix.replace("-", "_") not in directories:
    print("[Error] Provided prefix is not a model training. Exiting...")
    sys.exit(1)

prefix = "/".join([prefix, "data", args.dataset])


session = sagemaker.Session()
bucket = session.default_bucket()
s3_client = boto3.client('s3')

train = session.upload_data(
    path=f'{prefix.replace("-", "_")}/train',
    bucket=bucket,
    key_prefix=f'{prefix}/train'
)
test = session.upload_data(
    path=f'{prefix.replace("-", "_")}/test',
    bucket=bucket,
    key_prefix=f'{prefix}/test'
)
s3_client.put_object(Bucket=bucket, Key=f"{prefix}/tensorboard/")


print(f"Training data uploaded to: {train}")
print(f"Test data uploaded to: {test}")
