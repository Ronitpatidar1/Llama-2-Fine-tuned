# Llama 2 Instruction Fine-Tuning Project

This project shows a practical fine-tuning workflow for a large language model using QLoRA and the Hugging Face ecosystem.

## Project Overview

The repository contains a complete example of adapting `NousResearch/Llama-2-7b-chat-hf` to follow instruction-style prompts using a curated dataset. It includes dataset preparation, model configuration, and training logic.

Key skills demonstrated:
- Efficient LLM fine-tuning with parameter-efficient LoRA adapters
- 4-bit quantization using `bitsandbytes` to reduce GPU memory usage
- Data preparation and prompt formatting for instruction-following tasks
- Training automation with a reusable CLI script
- Inference sample generation for validation

## Repository Contents

- `fine_tune_llama_2.py`: main script for fine-tuning with command-line arguments
- `requirements.txt`: project dependencies pinned for reproducibility
- `.gitignore`: ignores model artifacts, logs, and temporary files
- `README.md`: project description and setup instructions

## Setup

1. Clone the repository:

```bash
git clone <your-repo-url>
cd "Finetuned llama 2"
```

2. Create a Python environment and install dependencies:

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

3. Run fine-tuning:

```bash
python fine_tune_llama_2.py --use-4bit --train-batch-size 4 --num-train-epochs 1
```

> Note: A CUDA-enabled GPU is required to fine-tune large LLMs. This project is designed for GPU-based training.

## Example workflow

The script performs the following steps:
- loads the base Llama 2 chat model
- downloads and formats the instruction dataset
- configures LoRA adapters for efficient fine-tuning
- trains the model and saves the final weights
- generates a sample answer to verify output quality

## Extensions

Possible improvements:
- add validation and evaluation metrics
- support Hugging Face Hub model publishing
- include a training notebook with results
- add a small demo for model inference
