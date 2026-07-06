"""
Real Q&A test -- run this against the FINE-TUNED checkpoint, after
finetune.py finishes. This is the actual test the whole project has
been building toward: real Flutter questions in, real answers out --
not code completion, not a loss number.

Note: some of these questions are similar to what's in the training
set (flutter_training_data.jsonl had near-identical phrasing for a
few of them), so a good answer to those confirms the QA behavior was
learned, not necessarily that it generalizes to brand-new questions.
Worth also trying questions of your own that definitely aren't in the
training data, to see how it handles those.
"""

import sys

import torch
from tokenizers import Tokenizer

from model import KalaiNovaModel

CHECKPOINT = sys.argv[1] if len(sys.argv) > 1 else "checkpoints_finetune/finetune_best.pt"
TOKENIZER_PATH = "kalainova_tokenizer/tokenizer.json"
MAX_NEW_TOKENS = 200

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
eot_id = tokenizer.token_to_id("<|endoftext|>")

model = KalaiNovaModel().to(device)
ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=True)
model.load_state_dict(ckpt["model"])
model.eval()
print(f"Loaded checkpoint from step {ckpt['step']}, val_loss={ckpt.get('val_loss', 'n/a')}")

QUESTIONS = [
    "What is StatefulWidget in Flutter?",
    "How do I use setState in Flutter?",
    "What is the difference between StatelessWidget and StatefulWidget?",
    "How do I create a ListView in Flutter?",
    "What is a Scaffold widget used for?",
]

for q in QUESTIONS:
    ids = tokenizer.encode(q).ids
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(x, max_new_tokens=MAX_NEW_TOKENS, temperature=0.7, top_p=0.9)

    new_ids = out[0, len(ids):].tolist()
    if eot_id in new_ids:
        new_ids = new_ids[: new_ids.index(eot_id)]  # trim anything after the stop token
    answer = tokenizer.decode(new_ids)

    print("=" * 60)
    print("Q:", q)
    print("A:", answer)
    print()