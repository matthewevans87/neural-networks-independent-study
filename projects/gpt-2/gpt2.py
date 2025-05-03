import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import termios
import tty

# pick MPS if available, otherwise fall back to CPU
device = "mps" if torch.backends.mps.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("gpt2-xl")
model     = AutoModelForCausalLM.from_pretrained("gpt2-xl").to(device)



# REPL loop
while True:

    # Read the user prompt
    user_text = input("Enter your prompt: ").strip()

    # If no user input, fall back to prompt.txt
    if not user_text:
        try:
            with open("prompt.md", "r") as f:
                user_text = f.read().strip()
        except FileNotFoundError:
            print("prompt.md not found, please enter a prompt.")
            continue

    # Load preprompt if it exists
    try:
        with open("preprompt.md", "r") as f:
            raw = f.read().strip()
            # escape backslashes first, then double and single quotes
            preprompt = (raw
                         .replace("\\", "\\\\")
                         .replace('"', '\\"')
                         .replace("'", "\\'"))
    except FileNotFoundError:
        preprompt = ""
    # Combine preprompt and user prompt
    if preprompt:
        full_prompt = f"{preprompt}\n{user_text}"
    else:
        full_prompt = user_text

    # Tokenize and generate
    inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,
        num_beams=1,
        eos_token_id=tokenizer.eos_token_id,
        # temperature=0.8,
        # top_k=50,
        # top_p=0.9
    )
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

