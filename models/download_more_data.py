from datasets import load_dataset
import os

print("Downloading datasets...")
all_text = []

# Dataset 1 - Python instructions (already have, but redo for completeness)
print("1/3: Python instructions...")
ds1 = load_dataset('iamtarun/python_code_instructions_18k_alpaca', split='train')
for item in ds1:
    all_text.append(item['output'])
print(f"  Added {len(ds1)} examples")

# Dataset 2 - More general code
print("2/3: CodeAlpaca...")
try:
    ds2 = load_dataset('sahil2801/CodeAlpaca-20k', split='train')
    for item in ds2:
        if 'output' in item:
            all_text.append(item['output'])
    print(f"  Added {len(ds2)} examples")
except Exception as e:
    print(f"  Skipped: {e}")

# Dataset 3 - Evol instruct code
print("3/3: Evol Instruct Code...")
try:
    ds3 = load_dataset('mlabonne/Evol-Instruct-Python-26k', split='train')
    for item in ds3:
        if 'output' in item:
            all_text.append(item['output'])
    print(f"  Added {len(ds3)} examples")
except Exception as e:
    print(f"  Skipped: {e}")

print(f"\nTotal examples: {len(all_text)}")

with open('coding_data_v2.txt', 'w') as f:
    for text in all_text:
        f.write(text + '\n\n')

size = os.path.getsize('coding_data_v2.txt')
print(f"Saved coding_data_v2.txt: {size/1024/1024:.1f} MB")