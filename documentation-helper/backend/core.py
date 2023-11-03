import os
import sys
from pathlib import Path

import pinecone
from dotenv import dotenv_values
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.pinecone import Pinecone

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
from consts import INDEX_NAME

config = dotenv_values(".env")
os.environ["OPENAI_API_KEY"] = config["OPENAI_API_KEY"]
os.environ["PINECONE_API_KEY"] = config["PINECONE_API_KEY"]
os.environ["PINECONE_ENV_REGION"] = config["PINECONE_ENV_REGION"]

pinecone.init(
    api_key=os.environ["PINECONE_API_KEY"],
    environment=os.environ["PINECONE_ENV_REGION"],
)


def run_llm(query: str) -> any:
    embeddings = OpenAIEmbeddings()
    docsearch = Pinecone.from_existing_index(
        index_name=INDEX_NAME, embedding=embeddings
    )
    chat = ChatOpenAI(verbose=True, temperature=0)  # get more funky answers
    qa = RetrievalQAWithSourcesChain.from_chain_type(
        llm=chat,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
    )
    return qa({"question": query})


if __name__ == "__main__":
    print(run_llm(query="What is LangChain?"))
