
import argparse
import logging
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Pipeline,
    TrainingArguments,
    pipeline,
)
from trl import SFTTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune Llama 2 using QLoRA and a Hugging Face instruction dataset."
    )
    parser.add_argument(
        "--model-name",
        default="NousResearch/Llama-2-7b-chat-hf",
        help="Base Hugging Face model name to fine-tune.",
    )
    parser.add_argument(
        "--dataset-name",
        default="mlabonne/guanaco-llama2-1k",
        help="Hugging Face dataset name used for instruction fine-tuning.",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory where checkpoints and training logs are stored.",
    )
    parser.add_argument(
        "--final-model-dir",
        default="llama2-finetuned",
        help="Directory where the final merged model is saved.",
    )
    parser.add_argument(
        "--num-train-epochs",
        type=int,
        default=1,
        help="Number of fine-tuning epochs.",
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=4,
        help="Per-GPU training batch size.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps to simulate larger batches.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Initial learning rate for the optimizer.",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=64,
        help="Rank for LoRA low-rank adapters.",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=16,
        help="LoRA scaling alpha.",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.1,
        help="Dropout probability for LoRA layers.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=1024,
        help="Maximum sequence length during fine-tuning.",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=25,
        help="Log every N steps during training.",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=0,
        help="Save intermediate checkpoints every N steps (0 disables frequent checkpoints).",
    )
    parser.add_argument(
        "--use-4bit",
        action="store_true",
        help="Enable 4-bit quantization to reduce memory usage.",
    )
    parser.add_argument(
        "--bnb-4bit-quant-type",
        default="nf4",
        help="4-bit quantization type used by bitsandbytes.",
    )
    parser.add_argument(
        "--bnb-4bit-compute-dtype",
        default="float16",
        choices=["float16", "bfloat16"],
        help="Compute dtype for 4-bit quantized weights.",
    )
    return parser.parse_args()


def prepare_dataset(dataset_name: str):
    dataset = load_dataset(dataset_name, split="train")

    if "text" in dataset.column_names:
        return dataset

    def format_example(example):
        instruction = example.get("instruction") or example.get("prompt") or ""
        output = example.get("output") or example.get("response") or ""
        if instruction and output:
            return {"text": f"<s>[INST] {instruction} [/INST] {output}"}

        if "input" in example and "output" in example:
            return {"text": f"<s>[INST] {example['input']} [/INST] {example['output']}"}

        return {"text": example.get("text", "")}

    return dataset.map(format_example, remove_columns=dataset.column_names)


def build_bitsandbytes_config(args: argparse.Namespace):
    if not args.use_4bit:
        return None

    compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)
    if args.bnb_4bit_compute_dtype == "bfloat16" and torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        if major < 8:
            logging.warning(
                "bfloat16 is only supported on Ampere or newer GPUs. Falling back to float16."
            )
            compute_dtype = torch.float16

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )


def build_model_and_tokenizer(args: argparse.Namespace, quant_config):
    device_map = {"": 0} if args.use_4bit else ("auto" if torch.cuda.is_available() else None)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=quant_config,
        device_map=device_map,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer


def run_training(args: argparse.Namespace):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Loading dataset: %s", args.dataset_name)
    dataset = prepare_dataset(args.dataset_name)
    logging.info("Dataset loaded with %s samples", len(dataset))

    quant_config = build_bitsandbytes_config(args)
    logging.info("Loading model: %s", args.model_name)
    model, tokenizer = build_model_and_tokenizer(args, quant_config)

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.001,
        fp16=args.use_4bit and args.bnb_4bit_compute_dtype == "float16",
        bf16=args.use_4bit and args.bnb_4bit_compute_dtype == "bfloat16",
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        report_to="tensorboard",
        group_by_length=True,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        tokenizer=tokenizer,
        args=training_args,
        packing=False,
    )

    logging.info("Beginning fine-tuning...")
    trainer.train()

    final_dir = Path(args.final_model_dir)
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    logging.info("Saved final model to %s", final_dir)

    return model, tokenizer


def generate_example(model, tokenizer, prompt: str, max_length: int = 200) -> str:
    prompt_text = f"<s>[INST] {prompt} [/INST]"
    generator = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=max_length,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated = generator(prompt_text)
    return generated[0]["generated_text"]


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not torch.cuda.is_available():
        logging.error("CUDA is required for fine-tuning large LLMs. Please run on a GPU instance.")
        raise SystemExit(1)

    model, tokenizer = run_training(args)

    prompt = "Explain why cost-efficient fine-tuning is important for production LLM workflows."
    example_output = generate_example(model, tokenizer, prompt)
    logging.info("Example generation output:\n%s", example_output)


if __name__ == "__main__":
    main()

