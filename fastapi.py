import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from unsloth import FastLanguageModel
from transformers import AutoTokenizer

# Initialize FastAPI
app = FastAPI(title="Unsloth LLM API", description="Serving LoRA fine-tuned Unsloth model with FastAPI")

# Load Model and Tokenizer
MODEL_PATH = "models/lora_model"  # Adjust path if needed
MAX_SEQ_LENGTH = 2048
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading model...")
model, tokenizer = FastLanguageModel.from_pretrained(
	model_name=MODEL_PATH,
	max_seq_length=MAX_SEQ_LENGTH,
	dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
	load_in_4bit=False if DEVICE == "cpu" else True
	)
FastLanguageModel.for_inference(model)
model.to(DEVICE)
print("Model loaded successfully!")

# Define Request Model
class TextRequest(BaseModel):
	text: str
	max_tokens: int = 2048 # Default max output tokens

# API Endpoint: Generate Response
@app.post("/generate")
def generate_response(request: TextRequest):
	try:
		# Tokenize Input
		input_ids = tokenizer(request.text, return_tensors="pt").input_ids.to(DEVICE)

		# Generate Response
		outputs = model.generate(input_ids, max_new_tokens=request.max_tokens, pad_token_id=tokenizer.eos_token_id)
		generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

		return {"input": request.text, "generated_text": generated_text}

	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))

# API Health Check
@app.get("/")
def health_check():
	return {"status": "API is running", "device": DEVICE}