import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

model_name = "Qwen/Qwen2.5-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name, device_map="mps")

lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

dataset = load_dataset("json", data_files="flutter_training_data.jsonl", split="train")

def format_example(example):
    q = example.get("question") or example.get("instruction") or ""
    a = example.get("answer") or example.get("output") or ""
    return {"text": f"### Question\n{q}\n\n### Answer\n{a}<|endoftext|>"}

dataset = dataset.map(format_example)

def tokenize_function(examples):
    tokenized = tokenizer(examples["text"], truncation=True, max_length=512, padding="max_length")
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

training_args = TrainingArguments(
    output_dir="./kalainova_finetuned_v3",
    num_train_epochs=2,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=1e-4,
    fp16=False,
    save_steps=100,
    logging_steps=10,
    optim="adamw_torch",
    report_to="none",
    remove_unused_columns=False,
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

print("Starting improved training...")
trainer.train()

model.save_pretrained("./kalainova_final_v3")
tokenizer.save_pretrained("./kalainova_final_v3")
print("✅ Improved training finished!")
