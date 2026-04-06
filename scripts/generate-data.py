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
from huggingface_hub import scan_cache_dir, snapshot_download

from lib.utils import extract_csv, create_split
from rich import print


usage_info = f"""
Generate synthetic data for a model training. Uses `Qwen/Qwen3-4B-Instruct-2507` by default.
The prompt can be provided interactively by setting the -i option or from a json file with
("prompt_title" "system" "user") keys. The generated training data is split and saved to
data/ directory of the model training.
"""

parser = argparse.ArgumentParser(description=usage_info, usage=f"{os.path.basename(__file__)} [Positional Args]")

parser.add_argument("-m", "--model", type=str, help="Model to use for generation")
parser.add_argument('-i', '--interactive', action='store_true', help="Interactive mode")
parser.add_argument('model_training', type=str, help="The model training to generate data for (required)")
parser.add_argument('prompt_file', nargs='?', type=str, help="The prompt file to use as input to the model (required)")
parser.add_argument('train_split', default=.7, nargs='?', type=float, help="The portion for training split (optional: default .7)")
parser.add_argument('test_split', default=.3, nargs='?', type=float, help="The portion for test split (optional: default .3)")

args = parser.parse_args()

if args.model:
    checkpoint = args.model
    cache = scan_cache_dir()
    for repo in cache.repos:
        if checkpoint == repo.repo_id:
            # Model already cached
            os.environ['HF_HUB_OFFLINE'] = '1'
            break
        else:
            # del os.environ['HF_HUB_OFFLINE']
            snapshot_download(repo_id=checkpoint)
            os.environ['HF_HUB_OFFLINE'] = '1'
else:
    checkpoint = "Qwen/Qwen3-4B-Instruct-2507"

if torch.accelerator.is_available():
    device = torch.accelerator.current_accelerator()
else:
    device = torch.device("cpu")


BASE_DIR = args.model_training
BASE_DATA_DIR = BASE_DIR + "/data"

configuration_message = f"""
Generating synthetic training data for `{BASE_DIR}` model training using:

- model: {checkpoint}
- device: {device}
"""
print(configuration_message)


tokenizer = AutoTokenizer.from_pretrained(checkpoint, dtype=torch.float16, legacy=False, local_files_only=True, use_safetensors=True, device=device)
model = AutoModelForCausalLM.from_pretrained(checkpoint, dtype=torch.float16, use_safetensors=True, local_files_only=True)
model.to(device)

generation_args = {
    "num_beams": 2,
    "early_stopping": True,
    "max_new_tokens": 128000,
    "do_sample": False,
}

if args.interactive:
    sys_instruct = input("\nInput an optional role instructions for the model>>>\n")
    user_prompt = input("Input prompt>>>\n")
else:
    with open(f"{BASE_DATA_DIR}/prompts/{args.prompt_file}", 'r') as f:
        prompt_data = json.load(f)
        prompt_title = prompt_data["prompt_title"]
        sys_instruct = "\n".join(prompt_data["system"])
        user_prompt = "\n".join(prompt_data["user"])
    os.makedirs(f"{BASE_DATA_DIR}/{prompt_title}", exist_ok=True)

conversation = []
if isinstance(sys_instruct, str) and isinstance(user_prompt, str):
    conversation.append({"role": "system", "content": sys_instruct})
    conversation.append({"role": "user", "content": user_prompt})

start_time = time.time()
text = tokenizer.apply_chat_template(
    conversation,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    **generation_args
)
end_time = time.time()
elapsed_time = end_time - start_time

output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
content = tokenizer.decode(output_ids, skip_special_tokens=True)
print(f"{checkpoint} output: ", content)

df = extract_csv(content, ["Output:", "CSV", "Synthetic Bank Transaction Data", "bank_transactions.csv"])
if df is None:
    out_file = f"{BASE_DATA_DIR}/gen_out_{prompt_title}.md"
    print(f"Header `Output:` can not be found in generated output. Saving output to {out_file}")
    with open(out_file, "a") as f:
        f.write(content)
    sys.exit()

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

user_response = input(f"Create splits and save the generated synthetic data? (y/N)\n>>> ")

if user_response == "y":
    class_set_data = { "prompt_title": prompt_title, "description": "Transaction classes", "classes": classes}
    with open(f"{BASE_DIR}/class_set.json", "x") as f:
        json.dump(class_set_data, f)

    create_split(df, f"{BASE_DATA_DIR}/{prompt_title}", (.7,.3), True)

print("Finished...")
print(f"Generated in {elapsed_time:.2f} seconds.")
