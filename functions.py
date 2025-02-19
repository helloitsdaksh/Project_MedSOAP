from datasets import Dataset
import pandas as pd
import torch
import evaluate
import json
import nltk
import re
from transformers import AutoTokenizer, TextStreamer
from unsloth import FastLanguageModel
from torch.utils.data import DataLoader

# Ensure necessary NLTK packages are available for METEOR
nltk.download("wordnet")
nltk.download("punkt")

# Load evaluation metrics
rouge = evaluate.load("rouge")
bleu = evaluate.load("sacrebleu")
meteor = evaluate.load("meteor")

def load_csv_data(csv_path):
	"""

	:param csv_path:
	:return:
	"""
	df = pd.read_csv(csv_path)
	dataset = Dataset.from_pandas(df)
	return dataset


def process_data(example):
	"""Extracts input conversation and expected SOAP note from dataset.
	:param example:
	:return: """
	try:
		conversation, soap_note = example["text"].split("[/INST]")
		conversation = conversation.replace("<s>[INST]", "").strip()
		soap_note = soap_note.replace("</s>", "").strip()
		return {"instruction": conversation, "output": soap_note}
	except ValueError:
		return {"instruction": example["text"], "output": ""}


def load_test_data(file_path, sample_size=50):
	"""Loads the first `sample_size` samples from the test dataset."""
	df = pd.read_csv(file_path)

	if "data" not in df.columns:
		raise ValueError(f"CSV must contain a 'data' column. Found: {df.columns}")

	input_texts = []
	reference_outputs = []

	df = df.head(sample_size)

	for idx, text in enumerate(df["data"]):
		try:
			instruction_match = re.search(r"\[INST\](.*?)\[/INST\]", text, re.DOTALL)
			instruction = instruction_match.group(1).strip() if instruction_match else ""

			soap_note_match = re.search(r"\[/INST\](.*)", text, re.DOTALL)
			soap_note = soap_note_match.group(1).replace("</s>", "").strip() if soap_note_match else ""

			input_texts.append(instruction)
			reference_outputs.append(soap_note)

		except Exception as e:
			print(f"Error processing row {idx}: {e}")
			continue

	return input_texts, reference_outputs


def load_model(model_path, max_seq_length=2048, dtype=torch.float16, load_in_4bit=True):
	"""Loads the Unsloth fine-tuned model."""
	model, tokenizer = FastLanguageModel.from_pretrained(
		model_name=model_path,
		max_seq_length=max_seq_length,
		dtype=dtype,
		load_in_4bit=load_in_4bit,
		)

	FastLanguageModel.for_inference(model)
	return model, tokenizer


def generate_responses_batch(model, tokenizer, prompts, batch_size=4, max_length=2048):
	"""Generates SOAP notes for a batch of test samples using Unsloth."""
	model.eval()
	predictions = []

	dataloader = DataLoader(prompts, batch_size=batch_size, shuffle=False)

	with torch.no_grad():
		for batch_idx, batch in enumerate(dataloader):
			try:
				inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to("cuda")

				outputs = model.generate(
					inputs.input_ids,
					max_new_tokens=max_length,
					pad_token_id=tokenizer.eos_token_id
					)

				batch_predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)

				cleaned_predictions = []
				for pred in batch_predictions:
					pred = pred.split("S:", 1)[1].strip() if "S:" in pred else pred.strip()
					pred = "S: " + pred
					cleaned_predictions.append(pred)

				predictions.extend(cleaned_predictions)

				print(f"Completed Batch {batch_idx + 1}/{len(dataloader)}")

			except Exception as e:
				print(f"Error in batch {batch_idx + 1}: {e}")
				continue

	return predictions


def evaluate_model(references, predictions):
	"""Computes BLEU, ROUGE, and METEOR scores."""
	references = [ref.strip() for ref in references]
	predictions = [pred.strip() for pred in predictions]

	rouge_results = rouge.compute(predictions=predictions, references=references)
	bleu_results = bleu.compute(predictions=predictions, references=references, smooth_method="exp")
	meteor_results = meteor.compute(predictions=predictions, references=references)

	return {
		"ROUGE": rouge_results,
		"BLEU": bleu_results["score"],
		"METEOR": meteor_results["meteor"],
		}