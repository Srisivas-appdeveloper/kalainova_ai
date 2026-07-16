"""
Shared Q&A extraction + quality filtering for the from-scratch pipeline
(pretrain.py, prepare_finetune_data.py).

instruction_data_multilang.txt turned out to contain real contamination
once sampled directly (not caught by the technical-doc audit, which only
checked file provenance, not content):
  - stretches of degenerate repeated phrasing (looks like broken
    back-translation/paraphrase noise from an upstream HF dataset)
  - off-topic dumps unrelated to code/Flutter (e.g. IANA language subtag
    registry entries: "Type: language\nSubtag: ...")
Feeding that straight into pretraining would actively teach the model to
produce exactly the kind of "wrong/irrelevant" output it's already
accused of, so every source gets filtered through here before use.
"""

import json
import re
from collections import Counter

_JUNK_MARKERS = (
    "Subtag:",
    "Type: language",
    "Type: extlang",
    "Type: variant",
)


def _is_repetitive(text, n=4, max_repeat_fraction=0.15, stutter_gap=3):
    """Flag text with degenerate repetition.

    Two patterns, checked separately because they look different:
    - a "stutter": the same n-word phrase recurs right after itself
      (e.g. "re -testing and how to re -testing and how to ..."), which
      a whole-document frequency ratio misses since it may only repeat
      3-4 times total in an otherwise-long paragraph.
    - one phrase dominating the entire document (frequency ratio).
    """
    words = text.split()
    if len(words) < n * 2:
        return False

    grams = Counter()
    last_seen = {}
    for i in range(len(words) - n + 1):
        gram = tuple(words[i : i + n])
        grams[gram] += 1
        if i - last_seen.get(gram, -(n + stutter_gap + 1)) <= n + stutter_gap:
            return True
        last_seen[gram] = i

    # Only meaningful once a gram has actually repeated a few times in a
    # long-enough text -- with few total n-grams even one unique phrase's
    # trivial count of 1 can exceed a small ratio threshold by pure chance.
    total_grams = len(words) - n + 1
    if total_grams < 20:
        return False
    most_common = grams.most_common(1)[0][1]
    return most_common >= 3 and most_common / total_grams > max_repeat_fraction


def is_low_quality(text):
    """True if `text` looks like noise rather than real code/QA content."""
    if not text or len(text.strip()) < 8:
        return True
    if any(marker in text for marker in _JUNK_MARKERS):
        return True
    printable = sum(1 for c in text if c.isprintable() or c in "\n\t")
    if printable / len(text) < 0.9:
        return True
    if _is_repetitive(text):
        return True
    return False


def iter_question_answer_txt(path):
    """Yields (question, answer) from a 'Question: ...\\nAnswer: ...\\n\\n'
    formatted file (instruction_data_multilang.txt)."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        blob = f.read()
    for chunk in blob.split("\n\n"):
        chunk = chunk.strip()
        if not chunk.startswith("Question:"):
            continue
        m = re.match(r"Question:\s*(.*?)\nAnswer:\s*(.*)", chunk, re.DOTALL)
        if not m:
            continue
        yield m.group(1).strip(), m.group(2).strip()


def iter_question_answer_markdown(path):
    """Yields (question, answer) from a '### Question\\n...\\n\\n### Answer\\n...'
    formatted file (grok_flutter_qa.txt)."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        blob = f.read()
    for block in blob.split("### Question\n")[1:]:
        if "### Answer" not in block:
            continue
        q, a = block.split("### Answer", 1)
        yield q.strip(), a.lstrip("\n").strip()


def iter_question_answer_jsonl(path):
    """Yields (question, answer) from the curated Track-A jsonl files."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            q = obj.get("question") or obj.get("instruction") or ""
            a = obj.get("answer") or obj.get("output") or ""
            if q and a:
                yield q.strip(), a.strip()


def clean_qa_pairs(pairs):
    """Drop low-quality and duplicate (question, answer) pairs."""
    seen = set()
    for q, a in pairs:
        if is_low_quality(q) or is_low_quality(a):
            continue
        key = (q, a)
        if key in seen:
            continue
        seen.add(key)
        yield q, a
