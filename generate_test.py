"""
generate_testset.py

Manually generates QA pairs from your PDF one chunk at a time.
Respects Groq free tier rate limits (30 req/min).
Saves to validation_set.csv — run this ONCE.

Run:
    python generate_testset.py
"""

import os
import time
import json
import random
import pandas as pd
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from groq import Groq

load_dotenv()

DATA_PATH  = "document/gk-book.pdf"
OUTPUT_CSV = "validation_set.csv"
TEST_SIZE  = 30     # number of QA pairs to generate
SLEEP_SEC  = 2.5    

groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])


# ── 1. Load and chunk PDF

def load_chunks():
    print("Loading PDF...")
    loader = PyPDFLoader(DATA_PATH)
    docs   = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)
    print(f"  {len(docs)} pages -> {len(chunks)} chunks")
    return chunks


# 2. Pick good chunks to generate questions from 

def sample_chunks(chunks, n):
    good = [
        c for c in chunks
        if len(c.page_content.strip()) > 150
        and not c.page_content.strip().isdigit()
    ]

    if len(good) < n:
        print(f"  Warning: only {len(good)} usable chunks, adjusting TEST_SIZE")
        n = len(good)

    selected = random.sample(good, n)
    print(f"  Selected {n} chunks to generate questions from")
    return selected


#  3. Ask Groq to make one QA pair from one chunk 

def generate_qa_from_chunk(chunk_text, page):
    prompt = f"""You are a question generator for a study quiz.

Read this text excerpt and write ONE clear factual question that can be answered
directly from the text. Then write the correct answer.

Rules:
- Question must be answerable ONLY from the text below
- Answer must be a complete sentence
- Do not ask vague questions like "What is discussed here?"
- If the text has no useful facts, respond with: SKIP

Text:
\"\"\"{chunk_text}\"\"\"

Respond in this exact JSON format (no extra text):
{{
  "question": "your question here",
  "answer": "your answer here"
}}

If skipping: SKIP"""

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.3,
        )

        content = response.choices[0].message.content.strip()

        if content == "SKIP" or content.startswith("SKIP"):
            return None

        data = json.loads(content)
        q = data.get("question", "").strip()
        a = data.get("answer", "").strip()

        if not q or not a or len(q) < 10 or len(a) < 10 or q == a:
            return None

        return {
            "question":     q,
            "ground_truth": a,
            "page":         page,
            "source_text":  chunk_text[:300],
        }

    except json.JSONDecodeError:
        return None
    except Exception as e:
        print(f"    Error: {e}")
        return None


# Generate all QA pairs with rate limiting 

def generate_all(chunks):
    rows = []
    selected = sample_chunks(chunks, TEST_SIZE)

    print(f"\nGenerating {len(selected)} QA pairs ({SLEEP_SEC}s delay between calls)...")
    print(f"Estimated time: ~{int(len(selected) * SLEEP_SEC / 60) + 1} minutes\n")

    for i, chunk in enumerate(selected):
        page = chunk.metadata.get("page", "?")
        text = chunk.page_content.strip()

        print(f"  [{i+1}/{len(selected)}] Page {page} -- ", end="", flush=True)

        result = generate_qa_from_chunk(text, page)

        if result:
            rows.append(result)
            print(f"OK  ->  {result['question'][:55]}...")
        else:
            print("skipped (no useful fact found)")

        if i < len(selected) - 1:
            time.sleep(SLEEP_SEC)

    return rows


# 5. Save to CSV 

def save(rows):
    if not rows:
        print("\nERROR: No QA pairs were generated. Check your PDF path and API key.")
        return

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["question"])
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\n{'='*50}")
    print(f"Saved {len(df)} QA pairs to {OUTPUT_CSV}")
    print(f"{'='*50}")
    print("\nSample questions generated:")
    for _, row in df.head(5).iterrows():
        print(f"\n  Q: {row['question']}")
        print(f"  A: {row['ground_truth'][:100]}")
    print(f"\nNow run:  python eval.py")


def main():
    chunks = load_chunks()
    rows   = generate_all(chunks)
    save(rows)


if __name__ == "__main__":
    main()