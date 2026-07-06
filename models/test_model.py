from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained("./kalainova_final_v3")
tokenizer = AutoTokenizer.from_pretrained("./kalainova_final_v3")

prompts = [
    "How do I create a StatefulWidget in Flutter?",
    "What is the difference between StatelessWidget and StatefulWidget?",
    "How do I make an HTTP GET request in Flutter?",
]

for p in prompts:
    inputs = tokenizer(p, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=200)
    print("Q:", p)
    print("A:", tokenizer.decode(out[0], skip_special_tokens=True))
    print("---")
