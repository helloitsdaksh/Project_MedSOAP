# 🚀 Fine-Tuning and Evaluating Unsloth-Based LLM

This repository provides a streamlined pipeline for **fine-tuning** and **evaluating** a **LoRA-adapted Unsloth model** using **TRL's SFTTrainer** and **Hugging Face Transformers**. It supports **Docker-based deployment** for easy execution in isolated environments.

find the model and its checkpoints here: <https://drive.google.com/drive/folders/1FhOqbE5cm_P7sIVGea2N-xnxru3kjV7P?usp=drive_link>

## 📂 Project Structure

```
├── Dockerfile              # Docker setup for containerized execution
├── requirements.txt        # Python dependencies
├── trainer.py              # Training script for fine-tuning the model
├── evaluation.py           # Evaluation script with BLEU, ROUGE, and METEOR scores
├── functions.py            # Utility functions for training and evaluation
├── data/
│   ├── train_llama_formatted.csv         # Training dataset
│   ├── validation_llama_formatted.csv    # Validation dataset
│   ├── test_llama_formatted.csv          # Test dataset
├── models/
│   ├── lora_model/       # Fine-tuned LoRA model (output)
├── generated_output.txt   # Output from the model
├── evaluation_results.json # Metrics from evaluation
└── README.md              # Documentation
```

Here’s a detailed README with clear sections on how to train the model from scratch or use the FastAPI-based inference with Docker.

🚀 Fine-Tuning & Serving Unsloth LLM with FastAPI

This repository provides a streamlined workflow for:
1.	Fine-tuning a LoRA-adapted Unsloth model using TRL’s SFTTrainer and Hugging Face Transformers.
2.	Deploying the model via a FastAPI server, enabling real-time inference.
3.	Running everything inside a Docker container for portability.

# 📌 Choose Your Approach
### Option 1: Fine-tune the model from scratch (Trainer & Evaluation Section).
### Option 2: Use Docker & FastAPI for inference (API Section).



# 📌 Option 1: Fine-Tuning the Model from Scratch

🛠 Installation & Setup

1️⃣ Install Dependencies

`pip install -r requirements.txt`

2️⃣ Prepare Data

Ensure your training, validation, and test datasets are stored in the data/ directory and formatted as required.

🏋️ Fine-Tuning the Model

Run Trainer Script

`python trainer.py --train_data "data/train_llama_formatted.csv" --valid_data "data/validation_llama_formatted.csv"`

	*	This fine-tunes the LoRA-adapted Unsloth model.
	*	The trained model will be saved inside the models/lora_model/ directory.

📊 Evaluating the Fine-Tuned Model

Run Evaluation Script

`python evaluation.py --model_path "models/lora_model" --test_data "data/test_llama_formatted.csv"`

	*	This script computes BLEU, ROUGE, and METEOR scores for evaluating model performance.
	*	The evaluation results are saved in evaluation_results.json.

# 📌 Option 2: Running the API with Docker

🚀 Deploy Model Using FastAPI

Once the model is trained or downloaded, you can serve it via FastAPI.

1️⃣ Build the Docker Image

`docker build -t MedSOAP .`

2️⃣ Run the API in a Docker Container

`docker run --gpus all -p 8000:8000 -it unsloth-llm-api`

	*	The API will be accessible at: http://localhost:8000

📡 API Endpoints

1️⃣ Health Check
*	Check if the API is running and detect device (CPU/GPU).
*	Endpoint: GET /

*	Response:
```json
 {
    "status": "API is running",
    "device": "cuda"
 }
```


2️⃣ Generate Text
*	Generate a response from the fine-tuned LLM model.
*	Endpoint: POST /generate

*	Request Body (JSON):

```json
{
"text": "Doctor: Hello, can you please tell me about your past medical history?",
"max_tokens": 2048
}
```

*	Response (JSON):
```json
{
    "input": "Doctor: Hello, can you please tell me about your past medical history?",
    "generated_text": "The patient reports a history of..."
}
```


# 📌 How to Test the API

1️⃣ Using curl
```
curl -X 'POST' \
'http://localhost:8000/generate' \
-H 'Content-Type: application/json' \
-d '{"text": "Doctor: What brings you in today?", "max_tokens": 2048}'
```

2️⃣ Using Python (requests)
```python
import requests

url = "http://localhost:8000/generate"
payload = {"text": "Doctor: What brings you in today?", "max_tokens": 512}

response = requests.post(url, json=payload)
print(response.json())

```



# 📌 Notes
*	If you want to fine-tune the model, use Option 1.
*	If you just need to run inference, use Docker and FastAPI (Option 2).
*	Ensure you have an NVIDIA GPU for optimal performance.
*	Modify trainer.py to adjust LoRA hyperparameters as needed.
*	The dataset format should align with functions.py preprocessing logic.

# 💡 Future Work
*	Implement a streaming inference API for longer texts.
*	Optimize memory usage & add CPU support for macOS Metal.

📝 License

This project is open-source under the MIT License.

🚀 Happy fine-tuning & inferencing! 🚀
