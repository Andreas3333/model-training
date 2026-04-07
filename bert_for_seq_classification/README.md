# bert_for_seq_classification

This training script specializes base BERT on bank record transaction data for classification using
a custom synthetic dataset. It fine-tunes an MLP head for multi class classification using Linear
Probing then Fine Tunning (LP-FT) described in [Understanding Linear Probing then Fine-tuning Language Models from NTK Perspective](https://arxiv.org/pdf/2405.16747).

## Local usage

### Setup

```bash
# Install deps
cd bert_for_seq_classification
alias uv-venv='uv venv; source .venv/bin/activate'
uv-venv
uv pip install -r requirements.txt 
```

### Start a local training run

```bash
python main.py
```

### Sagemaker AI

TODO: move `sm_job.py` over to sagemaker v3 and ModelTrainer

#### Set up

Build and push a training job cpu image to your ECR registry

```bash
aws ecr get-login-password --region <your-region> | docker login --username AWS --password-stdin <your-account-id>.dkr.ecr.<your-region>.amazonaws.com
docker build -t <your-image-name> .
docker tag <your-image-name>:latest <your-account-id>.dkr.ecr.<your-region>.amazonaws.com/<your-image-name>:latest
docker push <your-account-id>.dkr.ecr.<your-region>.amazonaws.com/<your-image-name>:latest
```

#### Run the train job

- Start tensorboard server locally (optional)
```bash
cd bert_for_seq_classification
F_CPP_MIN_LOG_LEVEL=3 AWS_REGION=<region> tensorboard --logdir s3://sagemaker/bert_for_seq_classification/data/00_sft/tensorboard/
```
- Run training Job
```bash
python sm_job.py
```
