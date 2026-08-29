from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from pathlib import Path
import os


# Local model folder - convert to forward slashes for compatibility
MODEL_PATH = str(Path("./models/qwen-0.5b").resolve()).replace("\\", "/")


print("====================================")
print("       QWEN LOCAL AI MODEL")
print("====================================")

# Verify model path exists
if not os.path.isdir(MODEL_PATH):
    print(f"\nERROR: Model directory not found at: {MODEL_PATH}")
    print("Please ensure the model files are in: ./models/Qwen-0.5/")
    exit(1)

print(f"\nModel path: {MODEL_PATH}")
print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
    use_cache=False
)

print("Tokenizer loaded.")

print("\nLoading model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float32,
    local_files_only=True,
    use_cache=False
)

print("Model loaded successfully!")


# ------------------------------------------------
# Ask the user for input
# ------------------------------------------------

while True:

    user_input = input("\nYou: ")

    if user_input.lower() in ["exit", "quit"]:

        print("\nAI: Goodbye!")
        break

    # Create chat message
    messages = [
        {
            "role": "user",
            "content": user_input
        }
    ]

    # Convert chat into model format
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Convert text to tokens
    inputs = tokenizer(
        text,
        return_tensors="pt"
    )

    print("AI is thinking...")

    # Generate answer
    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )

    # Get only newly generated tokens
    generated_tokens = outputs[
        0
    ][
        inputs["input_ids"].shape[1]:
    ]

    # Convert tokens to text
    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    print("\nAI:", response)