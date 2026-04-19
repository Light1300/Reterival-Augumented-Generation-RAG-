import os
from dotenv import load_dotenv

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

from openai import OpenAI

load_dotenv()

INDEX_NAME = "quickstart"
OLLAMA_MODEL = "llama3.2:3b"

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

#  OLLAMA CLIENT 
llm = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

#  EMBEDDINGS 
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

#  VECTORSTORE 
def get_vectorstore(embeddings):
    return PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings
    )

#  RETRIEVE 
def retrieve(vectorstore, query: str, k=5):
    results = vectorstore.similarity_search_with_score(query, k=k)

    # lower = better (distance)
    results = sorted(results, key=lambda x: x[1])

    return [doc for doc, _ in results]

#  BUILD CONTEXT 
def build_context(chunks):
    parts = []
    total = 0
    MAX_CHARS = 4000   # smaller → faster + better grounding

    for doc in chunks:
        text = doc.page_content.strip()

        if total + len(text) > MAX_CHARS:
            break

        parts.append(
            f"[Page {doc.metadata.get('page','?')}]\n{text}"
        )
        total += len(text)

    return "\n\n---\n\n".join(parts)

#  PROMPT 
def build_prompt(question: str, context: str):
    return f"""You are a strict retrieval-based QA system.

Rules:
- Answer ONLY from the context
- Do NOT use prior knowledge
- If answer not found, say: "I don't have enough information"

Context:
{context}

Question: {question}

Answer (concise, factual):"""

#  LLM CALL 
def call_llm(prompt: str):
    response = llm.chat.completions.create(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0
    )

    return response.choices[0].message.content.strip()

#  ASK 
def ask(vectorstore, question: str):
    chunks = retrieve(vectorstore, question)

    if not chunks:
        return "No relevant data found.", []

    context = build_context(chunks)
    prompt = build_prompt(question, context)

    answer = call_llm(prompt)

    return answer, chunks


def main():
    print("Loading embeddings...")
    embeddings = get_embeddings()

    print("Connecting to Pinecone...")
    vs = get_vectorstore(embeddings)

    print("Using Ollama model:", OLLAMA_MODEL)
    print("Make sure running: ollama serve\n")

    while True:
        q = input("Ask (exit to quit): ").strip()

        if q.lower() == "exit":
            break

        answer, chunks = ask(vs, q)

        print("\n--- Answer ---\n")
        print(answer)

        pages = sorted(
            set(str(doc.metadata.get("page", "?")) for doc in chunks)
        )

        print(f"\nSources: pages {', '.join(pages)}\n")


if __name__ == "__main__":
    main()