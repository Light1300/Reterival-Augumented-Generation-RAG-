"""
eval.py — runs locally via Ollama (no API limits, completely free)

Setup:
    curl -fsSL https://ollama.com/install.sh | sh
    ollama pull llama3.2:3b
    ollama serve          # keep this running in another terminal

Then:
    python eval.py
"""

import os
import time
import json
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI   

load_dotenv()

INDEX_NAME     = "quickstart"
VALIDATION_CSV = "validation_set.csv"
RESULTS_FILE   = "eval_results.json"

OLLAMA_MODEL   = "llama3.2:3b"   
SLEEP_SEC      = 0.5             

# ── Ollama client (points to your local server) 
llm = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",   # required by the client but ignored locally
)

# ── Pinecone + embeddings 
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

def get_vectorstore(embeddings):
    return PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)


#  call local LLM 
def call_llm(prompt: str, max_tokens: int = 300) -> str:
    response = llm.chat.completions.create(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0,
    )
    return response.choices[0].message.content.strip()


#  RAG pipeline 
def retrieve(vectorstore, query, k=4):
    results = vectorstore.similarity_search_with_score(query, k=k)
    results = sorted(results, key=lambda x: x[1])
    return [doc for doc, _ in results]

def build_prompt(question, chunks):
    parts = []
    total = 0
    for doc in chunks:
        text = doc.page_content.strip()
        if total + len(text) > 3000:   # smaller context = faster local inference
            break
        parts.append(f"[Page {doc.metadata.get('page','?')}]\n{text}")
        total += len(text)
    context = "\n\n---\n\n".join(parts)
    return f"""Answer ONLY from the context below.
If the answer is not in the context, say: "I don't have enough information."

Context:
{context}

Question: {question}
Answer:"""

def ask(vectorstore, question):
    chunks = retrieve(vectorstore, question, k=4)
    if not chunks:
        return "No relevant data found.", []
    answer = call_llm(build_prompt(question, chunks), max_tokens=200)
    return answer, chunks


# ── Step 1: run RAG on all questions ──────────
def run_rag(vs, df):
    print(f"\nStep 1/2 — Running RAG on {len(df)} questions...")
    rows = []

    for i, row in df.iterrows():
        q  = row["question"]
        gt = row["ground_truth"]
        print(f"  [{i+1}/{len(df)}] {q[:65]}...")

        try:
            answer, chunks = ask(vs, q)
            contexts = [c.page_content for c in chunks]
        except Exception as e:
            print(f"    error: {e}")
            answer, contexts = "Error.", []

        rows.append({
            "question":     q,
            "answer":       answer,
            "contexts":     contexts,
            "ground_truth": gt,
        })

        time.sleep(SLEEP_SEC)

    return rows


# Step 2: score each row
SCORING_PROMPT = """You are evaluating a RAG system. Score these 4 metrics from 0.0 to 1.0.

faithfulness:      Is the generated answer supported by the context? (1.0=fully, 0.0=hallucinated)
answer_relevancy:  Does the answer address the question? (1.0=yes, 0.0=off-topic)
context_precision: Are the retrieved chunks relevant to the question? (1.0=all relevant, 0.0=none)
context_recall:    Does the context contain what's needed to answer? (1.0=yes, 0.0=no)

QUESTION: {question}

CONTEXT (first 1500 chars):
{context}

GENERATED ANSWER: {answer}

GROUND TRUTH: {ground_truth}

Reply ONLY with JSON, no explanation:
{{"faithfulness": 0.0, "answer_relevancy": 0.0, "context_precision": 0.0, "context_recall": 0.0}}"""

def score_one(row):
    context_str = "\n---\n".join(row["contexts"])[:1500]
    prompt = SCORING_PROMPT.format(
        question=row["question"],
        context=context_str,
        answer=row["answer"],
        ground_truth=row["ground_truth"],
    )
    try:
        content = call_llm(prompt, max_tokens=80)
        # find the JSON part even if model adds extra text
        start = content.find("{")
        end   = content.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError("No JSON found in response")
        data = json.loads(content[start:end])
        return {k: max(0.0, min(1.0, float(v))) for k, v in data.items()}
    except Exception as e:
        print(f"    scoring parse error: {e} | raw: {content[:100]}")
        return None

def score_all(rows):
    print(f"\nStep 2/2 — Scoring {len(rows)} rows locally...")
    metrics    = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    all_scores = {m: [] for m in metrics}
    detailed   = []

    for i, row in enumerate(rows):
        print(f"  [{i+1}/{len(rows)}] scoring... ", end="", flush=True)
        result = score_one(row)
        if result:
            for m in metrics:
                all_scores[m].append(result.get(m, 0.5))
            detailed.append({**row, **result})
            vals = " ".join(f"{m[:3]}={result.get(m,0):.2f}" for m in metrics)
            print(vals)
        else:
            print("skipped")
        time.sleep(SLEEP_SEC)

    final = {m: round(sum(v) / len(v), 4) for m, v in all_scores.items() if v}
    return final, detailed


# Print results 
def print_results(scores):
    print("\n" + "=" * 54)
    print("  EVALUATION RESULTS")
    print("=" * 54)

    items = [
        ("faithfulness",      "Faithfulness       did LLM stay in context?"),
        ("answer_relevancy",  "Answer relevancy   does it answer the question?"),
        ("context_precision", "Context precision  were retrieved chunks relevant?"),
        ("context_recall",    "Context recall     were all needed chunks found?"),
    ]
    for key, label in items:
        val = scores.get(key)
        if val is None:
            continue
        bar    = "█" * int(val * 24) + "░" * (24 - int(val * 24))
        status = "OK " if val >= 0.7 else ("LOW" if val >= 0.5 else "BAD")
        print(f"\n  {status}  {label}")
        print(f"        [{bar}]  {val:.2f}")

    print("\n" + "=" * 54)
    tips = {
        "faithfulness":      (0.7, "tighten prompt — add 'answer ONLY from context'"),
        "answer_relevancy":  (0.7, "improve prompt clarity"),
        "context_precision": (0.6, "reduce chunk size or add reranking"),
        "context_recall":    (0.6, "try hybrid search (BM25 + dense)"),
    }
    issues = [(k, t) for k, (thresh, t) in tips.items() if scores.get(k, 1) < thresh]
    if issues:
        print("  What to fix:")
        for key, tip in issues:
            print(f"  -> {key} = {scores[key]:.2f}  =>  {tip}")
    else:
        print("  All scores look good!")
    print("=" * 54)

def save_results(scores, detailed):
    out = {
        "timestamp":     datetime.now().isoformat(),
        "model":         OLLAMA_MODEL,
        "num_questions": len(detailed),
        "scores":        scores,
        "rows": [{
            "question":          r["question"],
            "answer":            r["answer"],
            "ground_truth":      r["ground_truth"],
            "faithfulness":      r.get("faithfulness"),
            "answer_relevancy":  r.get("answer_relevancy"),
            "context_precision": r.get("context_precision"),
            "context_recall":    r.get("context_recall"),
        } for r in detailed],
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {RESULTS_FILE}")
    print("Re-run eval.py after any change to compare scores.")


def main():
    if not os.path.exists(VALIDATION_CSV):
        print(f"ERROR: {VALIDATION_CSV} not found. Run generate_testset.py first.")
        return

    df = pd.read_csv(VALIDATION_CSV).dropna(subset=["question", "ground_truth"])
    print(f"Loaded {len(df)} QA pairs")
    print(f"Using local Ollama model: {OLLAMA_MODEL}")
    print(f"Make sure Ollama is running:  ollama serve")

    print("\nLoading embeddings and connecting to Pinecone...")
    embeddings = get_embeddings()
    vs         = get_vectorstore(embeddings)

    rows             = run_rag(vs, df)
    scores, detailed = score_all(rows)

    print_results(scores)
    save_results(scores, detailed)


if __name__ == "__main__":
    main()