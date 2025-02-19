import argparse
import torch
from functions import *
import json

def main(args):
	"""Loads model, runs batch inference on test samples, and evaluates performance."""
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	model, tokenizer = load_model(args.model_path)
	test_inputs, test_outputs = load_test_data(args.test_data_path, args.sample_size)

	predictions = generate_responses_batch(model, tokenizer, test_inputs, batch_size=args.batch_size)

	eval_results = evaluate_model(test_outputs, predictions)

	print("\nEvaluation Metrics")
	print(f"BLEU Score: {eval_results['BLEU']:.2f}")
	print(f"ROUGE Scores: {eval_results['ROUGE']}")
	print(f"METEOR Score: {eval_results['METEOR']:.2f}")

	with open("evaluation_results.json", "w") as f:
		json.dump(eval_results, f, indent=4)

	print("\nEvaluation results saved to 'evaluation_results.json'")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Evaluate fine-tuned Unsloth model")

	parser.add_argument("--model_path", type=str, required=True, help="Path to the fine-tuned model")
	parser.add_argument("--test_data_path", type=str, required=True, help="Path to the test dataset CSV file")
	parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
	parser.add_argument("--sample_size", type=int, default=50, help="Number of test samples to evaluate")

	args = parser.parse_args()
	main(args)