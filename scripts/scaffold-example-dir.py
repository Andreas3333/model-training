#!/usr/bin/env -S uv run --script


import os
import sys
from pathlib import Path


if len(sys.argv) == 1 or len(sys.argv) in ["help", "--help", "-h"]:
    usage_info = """
    Build out directory structure for example training. For example `text_classification_bert_sft` produces:

    text_classification_bert_sft/   - base directory
        ├── data                    - data (optional)
        ├── inference.py            - load and use ft model
        ├── main.py                 - train script
        ├── model.py                - model
        └── README.md
    
    Usage: `example_training_directory`
    """
    print(usage_info)
    sys.exit(0)


PROJECT_ROOT = "example-trainings"

if os.getcwd().split("/")[-1] != PROJECT_ROOT:
    print(f"[Error] Run script from the project root directory, {PROJECT_ROOT}")
    sys.exit(1)

BASE_DIR = sys.argv[1]

proceed = input(f"Proceed creating example training directory? (y/N)\n\n- {BASE_DIR}\n")

if proceed == 'y':
    os.mkdir(BASE_DIR)
    os.mkdir(f"{BASE_DIR}/data")
    files = ["inference.py", "main.py", "model.py"]
    for file in files:
        Path(f"{BASE_DIR}/{file}").touch()
    Path(f"{BASE_DIR}/README.md").write_text(f"# {BASE_DIR}\n")
    print(f"Created {BASE_DIR} example training directory")
else:
    print("Exiting...")

