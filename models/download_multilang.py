from datasets import load_dataset
import os

print("=" * 50)
print("KalaiNova Assistant - Dataset Download")
print("Downloading: Flutter, JavaScript, Python, General")
print("=" * 50)

all_qa = []

# Dataset 1 - Flutter Q&A (most important for your work)
print("\n1/4: Flutter Code Q&A...")
try:
    ds1 = load_dataset('NoirZangetsu/Flutter-Code-with-Questions-Dataset-English', split='train')
    for item in ds1:
        q = item.get('questions', '').strip()
        a = item.get('code', '').strip()
        if q and a:
            all_qa.append({
                'instruction': q,
                'output': a,
                'language': 'Flutter/Dart'
            })
    print(f"  Added {len(ds1)} Flutter examples")
except Exception as e:
    print(f"  Error: {e}")

# Dataset 2 - Flutter Questions Answers
print("\n2/4: Flutter Questions Answers...")
try:
    ds2 = load_dataset('durrah/flutter-questions-answers', split='train')
    print(f"  Columns: {ds2.column_names}")
    count = 0
    for item in ds2:
        # try common column names
        q = item.get('question', item.get('Question', item.get('prompt', ''))).strip()
        a = item.get('answer', item.get('Answer', item.get('response', item.get('completion', '')))).strip()
        if q and a:
            all_qa.append({
                'instruction': q,
                'output': a,
                'language': 'Flutter/Dart'
            })
            count += 1
    print(f"  Added {count} Flutter Q&A examples")
except Exception as e:
    print(f"  Error: {e}")

# Dataset 3 - CodeAlpaca (JavaScript, Python, general)
print("\n3/4: CodeAlpaca 20k (JS + Python + general)...")
try:
    ds3 = load_dataset('sahil2801/CodeAlpaca-20k', split='train')
    for item in ds3:
        q = item.get('instruction', '').strip()
        a = item.get('output', '').strip()
        if q and a:
            all_qa.append({
                'instruction': q,
                'output': a,
                'language': 'General'
            })
    print(f"  Added {len(ds3)} general coding examples")
except Exception as e:
    print(f"  Error: {e}")

# Dataset 4 - Python instructions
print("\n4/4: Python Instructions 18k...")
try:
    ds4 = load_dataset('iamtarun/python_code_instructions_18k_alpaca', split='train')
    for item in ds4:
        q = item.get('instruction', '').strip()
        a = item.get('output', '').strip()
        if q and a:
            all_qa.append({
                'instruction': q,
                'output': a,
                'language': 'Python'
            })
    print(f"  Added {len(ds4)} Python examples")
except Exception as e:
    print(f"  Error: {e}")

print(f"\nTotal Q&A pairs collected: {len(all_qa)}")

# Count by language
from collections import Counter
lang_counts = Counter(item['language'] for item in all_qa)
for lang, count in lang_counts.items():
    print(f"  {lang}: {count}")

# Save as formatted instruction text
print("\nSaving instruction_data_multilang.txt...")
with open('instruction_data_multilang.txt', 'w') as f:
    for item in all_qa:
        instruction = item['instruction']
        output = item['output']
        lang = item['language']
        text = f"Question: {instruction}\nAnswer: {output}\n\n"
        f.write(text)

size = os.path.getsize('instruction_data_multilang.txt')
print(f"Saved: instruction_data_multilang.txt ({size/1024/1024:.1f} MB)")
print(f"Total examples: {len(all_qa)}")
print("\nDone! Ready for instruction fine-tuning.")