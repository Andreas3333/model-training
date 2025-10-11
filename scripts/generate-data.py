#!/usr/bin/env -S uv run --script

import os
import sys
import time
import json
import argparse

import torch
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

from lib.utils import extract_csv, create_split


usage_info = f"""
Generate synthetic data for an example training. By default uses `Qwen/Qwen3-4B-Instruct-2507`.
The prompt can be provided interactively by setting `INTERACTIVE_SYNTHETIC_GENERATION` to True.
Or from a json file with ("prompt_title" "system" "user") keys. The generated training data is
saved to example data/ directory of the example training.

Usage: {__name__} [ARGUMENTS]
"""

parser = argparse.ArgumentParser(description=usage_info)
parser.add_argument('example_training', nargs='?', metavar='str', type=str, help="The example training to generate data for")
parser.add_argument('prompt_file', nargs='?', metavar='str', type=str, help=f"The prompt file to use as input to the model")
parser.add_argument('train_split', default=.7, nargs='?', metavar='float', type=float, help="The portion for training split")
parser.add_argument('test_split', default=.3, nargs='?', metavar='float', type=float, help="The portion for test split")
args = parser.parse_args()

checkpoint = "Qwen/Qwen3-4B-Instruct-2507"

os.environ['HF_HUB_OFFLINE'] = '1'
if torch.accelerator.is_available():
    device = torch.accelerator.current_accelerator()
else:
    device = torch.device("cpu")


BASE_DATA_DIR = args.example_training
BASE_DATA_DIR += "/data"

configuration_message = f"""
Generating synthetic training data for `{BASE_DATA_DIR}` example training using:

- model: {checkpoint}
- device: {device}
"""
print(configuration_message)


tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, use_safetensors=True, device=device)
model = AutoModelForCausalLM.from_pretrained(checkpoint, use_safetensors=True, local_files_only=True)
model.to(device)

if os.getenv("INTERACTIVE_SYNTHETIC_GENERATION"):
    role_instruction = input("\nInput an optional role instructions for the model>>>\n")
    user_prompt = input("Input prompt>>>\n")
else:
    with open(f"{BASE_DATA_DIR}/prompts/{args.prompt_file}", 'r') as f:
        prompt_data = json.load(f)
        prompt_title = prompt_data["prompt_title"]
        role_instruction = "\n".join(prompt_data["system"])
        user_prompt = "\n".join(prompt_data["user"])
    os.makedirs(f"{BASE_DATA_DIR}/{prompt_title}", exist_ok=True)

conversation = []
if isinstance(role_instruction, str) and isinstance(user_prompt, str):
    conversation.append({"role": "user", "content": role_instruction})
    conversation.append({"role": "user", "content": user_prompt})

start_time = time.time()
text = tokenizer.apply_chat_template(
    conversation,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=16384
)
end_time = time.time()
elapsed_time = end_time - start_time

print(f"Generated in {elapsed_time:.2f} seconds.")
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
content = tokenizer.decode(output_ids, skip_special_tokens=True)
print(f"{checkpoint} output: ", content)

df = extract_csv(content["Category"])
classes = df["Category"].unique().tolist()
number_of_examples = []
for cat in classes:
    number_of_examples.append(len(df[df["Category"] == cat]))

df_tmp = pd.DataFrame({"Category": classes, "Number of examples": number_of_examples})
df_tmp.plot.bar(x="Category", y="Number of examples", rot=0, figsize=(8,8), ylim=(0, 40), color='lightcoral', title=f"Number of examples per class {checkpoint}")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show(block=False)
plt.pause(0.001)

print(f"Classes and number of examples: (total, {sum(number_of_examples)})")
print(classes)
print(number_of_examples)

user_response = input(f"Create splits and save the generated synthetic data? (y/N)")

if user_response == "y":
    class_set_data = { "prompt_title": prompt_title, "description": "Transaction classes", "classes": classes}
    with open(f"{BASE_DATA_DIR}/class_set.json", "x") as f:
        json.dump(class_set_data, f)

    create_split(df, f"{BASE_DATA_DIR}/{prompt_title}", (.7,.3), True)
else:
    print("Finished...")
