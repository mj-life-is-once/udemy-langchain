import os

import pinecone
from consts import INDEX_NAME
from dotenv import dotenv_values
from langchain.document_loaders import ReadTheDocsLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.pinecone import Pinecone

config = dotenv_values(".env")
os.environ["OPENAI_API_KEY"] = config["OPENAI_API_KEY"]
os.environ["PINECONE_API_KEY"] = config["PINECONE_API_KEY"]
os.environ["PINECONE_ENV_REGION"] = config["PINECONE_ENV_REGION"]

pinecone.init(
    api_key=os.environ["PINECONE_API_KEY"],
    environment=os.environ["PINECONE_ENV_REGION"],
)


def ingest_docs() -> None:
    loader = ReadTheDocsLoader(path="langchain-docs/api.python.langchain.com/en/latest")
    raw_documents = loader.load()
    print(f"loaded {len(raw_documents)} documents")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100, separators=["\n\n", "\n", " ", ""]
    )
    # rule of thumb (at most 4-5 contexts)
    # split documents into chunks
    documents = text_splitter.split_documents(documents=raw_documents)
    print(f"split into {len(documents)} chunks")

    for doc in documents:
        old_path = doc.metadata["source"]
        new_url = old_path.replace("langchain-docs", "https://")
        doc.metadata.update({"source": new_url})

    print(f"Going to insert {len(documents)} to pinecone")
    embeddings = OpenAIEmbeddings()
    Pinecone.from_documents(documents, embeddings, index_name=INDEX_NAME)


if __name__ == "__main__":
    ingest_docs()
