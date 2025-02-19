import argparse
import torch
from transformers import TrainingArguments
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer
from functions import load_csv_data, process_data

def main(args):
	# Load Model
	model, tokenizer = FastLanguageModel.from_pretrained(
		model_name=args.model_name,
		max_seq_length=args.max_seq_length,
		dtype=None,
		load_in_4bit=args.load_in_4bit,
		)

	model = FastLanguageModel.get_peft_model(
		model,
		r=args.r,
		target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
		                "gate_proj", "up_proj", "down_proj"],
		lora_alpha=args.lora_alpha,
		lora_dropout=0,
		bias="none",
		use_gradient_checkpointing="unsloth",
		random_state=args.seed,
		use_rslora=False,
		loftq_config=None,
		)

	# Load Data
	train_dataset = load_csv_data(args.train_csv)
	valid_dataset = load_csv_data(args.valid_csv)

	train_dataset = train_dataset.rename_column("data", "text")
	train_dataset = train_dataset.map(lambda examples: {'text': examples['text']})
	valid_dataset = valid_dataset.rename_column("data", "text")
	valid_dataset = valid_dataset.map(lambda examples: {'text': examples['text']})

	# Apply processing
	train_dataset = train_dataset.map(process_data)
	valid_dataset = valid_dataset.map(process_data)

	# Training Configuration
	training_args = TrainingArguments(
		per_device_train_batch_size=args.batch_size,
		per_device_eval_batch_size=args.eval_batch_size,
		evaluation_strategy=args.eval_strategy,
		num_train_epochs=args.num_train_epochs,
		gradient_accumulation_steps=args.gradient_accumulation_steps,
		warmup_steps=args.warmup_steps,
		learning_rate=args.learning_rate,
		fp16=not is_bfloat16_supported(),
		bf16=is_bfloat16_supported(),
		logging_steps=args.logging_steps,
		optim=args.optim,
		weight_decay=args.weight_decay,
		lr_scheduler_type=args.lr_scheduler_type,
		seed=args.seed,
		output_dir=args.output_dir,
		save_strategy=args.save_strategy,
		save_total_limit=args.save_total_limit,
		report_to=args.report_to,
		)

	trainer = SFTTrainer(
		model=model,
		tokenizer=tokenizer,
		train_dataset=train_dataset,
		eval_dataset=valid_dataset,
		dataset_text_field="instruction",
		max_seq_length=args.max_seq_length,
		dataset_num_proc=2,
		packing=False,
		args=training_args,
		)

	# Train Model
	trainer.train()

	# Save Model
	model.save_pretrained(args.output_model)
	tokenizer.save_pretrained(args.output_model)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Fine-tune LLaMA3 with Unsloth")

	# Model & Data Paths
	parser.add_argument("--model_name", type=str, default="unsloth/llama-3-8b-bnb-4bit", help="Pretrained model name")
	parser.add_argument("--train_csv", type=str, default="/data/train_llama_formatted.csv", help="Training dataset path")
	parser.add_argument("--valid_csv", type=str, default="/data/validation_llama_formatted.csv", help="Validation dataset path")
	parser.add_argument("--output_model", type=str, default="lora_model", help="Output directory for saving the model")

	# LoRA Parameters
	parser.add_argument("--r", type=int, default=16, help="LoRA rank")
	parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha scaling factor")

	# Training Parameters
	parser.add_argument("--max_seq_length", type=int, default=2048, help="Max sequence length for training")
	parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
	parser.add_argument("--eval_batch_size", type=int, default=16, help="Batch size for evaluation")
	parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs")
	parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
	parser.add_argument("--warmup_steps", type=int, default=5, help="Warmup steps for training")
	parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate for training")
	parser.add_argument("--eval_strategy", type=str, default="epoch", choices=["epoch", "steps"], help="Evaluation strategy")
	parser.add_argument("--logging_steps", type=int, default=10, help="Logging steps during training")

	# Optimization Parameters
	parser.add_argument("--optim", type=str, default="adamw_8bit", help="Optimizer type")
	parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for optimizer")
	parser.add_argument("--lr_scheduler_type", type=str, default="linear", help="Learning rate scheduler type")

	# Save & Reporting
	parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory for model checkpoints")
	parser.add_argument("--save_strategy", type=str, default="epoch", choices=["epoch", "steps"], help="Save strategy for checkpoints")
	parser.add_argument("--save_total_limit", type=int, default=2, help="Total number of saved checkpoints")
	parser.add_argument("--report_to", type=str, default="none", help="Reporting tool (e.g., wandb, tensorboard)")

	# Miscellaneous
	parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
	parser.add_argument("--load_in_4bit", type=bool, default=True, help="Whether to load the model in 4-bit quantization")

	args = parser.parse_args()
	main(args)