from transformers import AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM
import torch
import json

# Configuration
model_name = "gpt2"  # Replace with model name
dataset_path = "path_to_your_custom_math500_dataset.json"  # Replace with file path
output_dir = "./sft-model"  # Set output dir
apply_custom_prompt = True  # Set to False to use default prompt format
use_structure_tuning = False  # Set to True to enable structure tuning

# Define prompt templates
custom_prompt_templates = {
    "default": "{question} Please provide the initial step towards resolving the question. This step may serve as a foundation but might not encompass the entire solution.",
}

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)

# Load and preprocess dataset
raw_data = []
with open(dataset_path, "r") as f:
    for line in f:
        raw_data.append(json.loads(line))

# Flatten the data: 500 questions * 250 responses
flattened_data = []
for item in raw_data:
    question = item["problem"]
    q_type = item.get("mode", "default")
    for response in item["response"]:
        if apply_custom_prompt:
            if use_structure_tuning:
                template = custom_prompt_templates.get(
                    q_type, custom_prompt_templates["default"]
                )
            else:
                template = custom_prompt_templates["default"]
            prompt_text = template.format(question=question, response=response)
        else:
            prompt_text = f"Question: {question}\nAnswer: {response}"
        flattened_data.append({"text": prompt_text})

# Convert to Hugging Face Dataset
dataset = Dataset.from_list(flattened_data)


# Preprocess dataset
def tokenize_function(examples):
    return tokenizer(
        examples["text"], truncation=True, padding="max_length", max_length=512
    )


tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=4,
    num_train_epochs=3,
    logging_dir=f"{output_dir}/logs",
    logging_steps=10,
    save_steps=50,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
)

# Initialize SFTTrainer
trainer = SFTTrainer(
    model=model,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    args=training_args,
)

# Train the model
trainer.train()

# Save the model and tokenizer
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

print("Training complete. Model saved to:", output_dir)
