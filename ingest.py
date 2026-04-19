import os
from dotenv import load_dotenv

from pinecone import Pinecone, ServerlessSpec

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

DATA_PATH = "document/gk-book.pdf"
INDEX_NAME = "quickstart"
EMBEDDING_DIM = 384
BATCH_SIZE = 100

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])


def load_pdf(file_path: str):
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    for i, doc in enumerate(docs):
        doc.metadata["page"] = i

    return docs


def chunk_data(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20
    )
    return splitter.split_documents(docs)


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


def init_index():
    existing = [i["name"] for i in pc.list_indexes()]

    if INDEX_NAME not in existing:
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        print("Index created")
    else:
        print("Index already exists")


def store(chunks, embeddings):
    vs = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings
    )

    for i in range(0, len(chunks), BATCH_SIZE):
        vs.add_documents(chunks[i:i + BATCH_SIZE])
        print(f"Batch {i // BATCH_SIZE + 1} inserted")


def main():
    embeddings = get_embeddings()

    docs = load_pdf(DATA_PATH)
    print(f"Pages: {len(docs)}")

    chunks = chunk_data(docs)
    print(f"Chunks: {len(chunks)}")

    init_index()
    store(chunks, embeddings)

    print("Ingestion done")


if __name__ == "__main__":
    main()