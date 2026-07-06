"""
Generate synthetic Flutter/Dart Q&A training pairs using the Claude API.
Runs on CPU, no GPU needed. Costs API tokens (cheap at scale, check pricing).

Setup: pip install anthropic
Set ANTHROPIC_API_KEY env var.
"""

from openai import OpenAI
import json
import time
import os

client = OpenAI(
    api_key=os.environ.get("XAI_API_KEY", ""),
    base_url="https://api.x.ai/v1",
)

OUTPUT_FILE = "flutter_synthetic_qa.jsonl"

TOPICS = [
    "StatefulWidget lifecycle", "setState and rebuilds", "Provider state management",
    "Riverpod providers", "BLoC pattern", "go_router navigation", "Dio HTTP requests",
    "Firebase Auth integration", "Firestore CRUD", "ListView and GridView",
    "Custom animations", "Form validation", "async/await in Dart", "Streams and StreamBuilder",
    "Null safety", "Extension methods", "Mixins", "Isolates", "Platform channels",
    "Local storage (shared_preferences)", "flutter_secure_storage", "Custom painters",
    "Theming and Material 3", "Responsive layouts", "GetX state management",
    "Dependency injection with GetIt", "Unit testing widgets", "Integration testing",
    "Error handling patterns", "Localization (intl)",
]

GENERAL_CODE_TOPICS = [
    "Python list comprehensions", "Python decorators", "Python OOP classes",
    "JavaScript async/await", "JavaScript array methods (map/filter/reduce)",
    "JavaScript closures", "Java collections (List/Map/Set)", "Java OOP inheritance",
    "Java exception handling", "C++ pointers and memory", "C++ STL vectors",
    "SQL joins and queries", "SQL indexing basics", "Go goroutines and channels",
    "Rust ownership and borrowing", "React hooks (useState/useEffect)",
    "Node.js Express routing", "REST API design basics", "Git branching and merging",
    "Data structures: linked lists", "Data structures: binary trees",
    "Sorting algorithms", "Recursion basics", "Regular expressions",
    "HTML forms and validation", "CSS flexbox and grid", "CSS responsive design",
    "DOM manipulation with JavaScript", "Java Spring Boot REST API", "Java servlets basics",
]

BUSINESS_TOPICS = [
    "How to build a resume for a small business owner", "How to write a business plan",
    "How to price a mobile app project", "How to choose between website vs app",
    "What is MSME registration", "How to get GST registration in India",
    "How KalaiNova Infotech pricing works", "What services KalaiNova offers",
    "How long does it take to build a mobile app", "How to write a project proposal",
    "How to market a small business on Instagram", "How to write a client invoice",
    "What is a maintenance/support contract", "How to choose a tech stack for a startup",
    "How to write a cold outreach email to clients", "Basics of digital marketing for small business",
    "How to set up Google My Business", "What is SEO and why it matters for small business",
    "How to collect client requirements for an app project", "How to handle client feedback and revisions",
    "Step-by-step GST registration process in India", "Step-by-step MSME/Udyam registration process",
    "Step-by-step how to file GST returns", "Step-by-step how to register a company in India",
    "Step-by-step how to open a business bank account in India",
]

GENERAL_KNOWLEDGE_TOPICS = [
    "Popular smartphone brands and models comparison", "iPhone vs Android differences",
    "Popular bike/motorcycle brands in India", "Best bikes for daily commute in India",
    "Popular car brands and models in India", "Electric cars in India",
    "Basic science facts for kids", "Fun facts about animals for kids",
    "Basic geography facts", "World capitals and countries",
    "Basic history facts (India)", "Basic history facts (world)",
    "Simple math concepts for kids", "Basic solar system facts for kids",
    "Popular laptop brands and models", "Popular headphone/earbuds brands",
]

QUESTIONS_PER_TOPIC = 200


def generate_batch(topic, n=15, kind="code"):
    if kind == "code":
        prompt = f"""Generate {n} distinct, realistic developer questions and complete, correct code
answers about Flutter/Dart topic: "{topic}".

Format strictly as JSON lines, one object per line, no markdown, no extra text:
{{"question": "...", "answer": "..."}}

Answers must be working Dart/Flutter code with brief inline comments where helpful.
Vary difficulty (beginner to advanced) and question phrasing."""
    else:
        prompt = f"""Generate {n} distinct, realistic questions a small business owner or client
might ask about: "{topic}". This is for a business/IT-services chatbot on a software agency website.

Format strictly as JSON lines, one object per line, no markdown, no extra text:
{{"question": "...", "answer": "..."}}

Answers should be clear, practical, 3-6 sentences, no code. Vary question phrasing."""

    try:
        resp = client.chat.completions.create(
            model="grok-4-fast",
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as e:
        if "429" in str(e) or "resource-exhausted" in str(e):
            time.sleep(15)
            resp = client.chat.completions.create(
                model="grok-4-fast",
                max_tokens=4000,
                messages=[{"role": "user", "content": prompt}],
            )
        else:
            raise
    text = resp.choices[0].message.content
    pairs = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if "question" in obj and "answer" in obj:
                pairs.append(obj)
        except json.JSONDecodeError:
            continue
    return pairs


def main():
    total = 0
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for topic in TOPICS:
            print(f"Generating (code): {topic}")
            try:
                pairs = generate_batch(topic, QUESTIONS_PER_TOPIC, kind="code")
            except Exception as e:
                print(f"  Error: {e}")
                continue
            for p in pairs:
                out.write(json.dumps(p) + "\n")
            total += len(pairs)
            print(f"  Got {len(pairs)} pairs (total: {total})")
            time.sleep(1)

        for topic in GENERAL_CODE_TOPICS:
            print(f"Generating (code): {topic}")
            try:
                pairs = generate_batch(topic, QUESTIONS_PER_TOPIC, kind="code")
            except Exception as e:
                print(f"  Error: {e}")
                continue
            for p in pairs:
                out.write(json.dumps(p) + "\n")
            total += len(pairs)
            print(f"  Got {len(pairs)} pairs (total: {total})")
            time.sleep(1)

        for topic in BUSINESS_TOPICS:
            print(f"Generating (business): {topic}")
            try:
                pairs = generate_batch(topic, QUESTIONS_PER_TOPIC, kind="business")
            except Exception as e:
                print(f"  Error: {e}")
                continue
            for p in pairs:
                out.write(json.dumps(p) + "\n")
            total += len(pairs)
            print(f"  Got {len(pairs)} pairs (total: {total})")
            time.sleep(1)

        for topic in GENERAL_KNOWLEDGE_TOPICS:
            print(f"Generating (general): {topic}")
            try:
                pairs = generate_batch(topic, QUESTIONS_PER_TOPIC, kind="business")
            except Exception as e:
                print(f"  Error: {e}")
                continue
            for p in pairs:
                out.write(json.dumps(p) + "\n")
            total += len(pairs)
            print(f"  Got {len(pairs)} pairs (total: {total})")
            time.sleep(1)

    print(f"\nDone. Wrote {total} examples to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()