import os
from typing import Any, Dict, List

from dotenv import dotenv_values
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader, SeleniumURLLoader
from langchain.embeddings import OpenAIEmbeddings

# from langchain.llms.openai import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS

config = dotenv_values(".env")
os.environ["OPENAI_API_KEY"] = config["OPENAI_API_KEY"]


urls = [
    "https://www.minjoocho.com/",
    "https://www.minjoocho.com/projects/a1?category=artistic",
    "https://www.minjoocho.com/projects/a2?category=artistic",
    "https://www.minjoocho.com/projects/c1?category=creative",
    "https://www.minjoocho.com/projects/c2?category=creative",
    "https://www.minjoocho.com/projects/p1?category=pragmatic",
    "https://www.minjoocho.com/projects/p2?category=pragmatic",
]

cv_path = "./documents/resume_minjoo_2023.pdf"
documents = []


def run_llm(query: str, chat_history: List[Dict[str, Any]]) -> Any:
    cv_loader = PyPDFLoader(file_path=cv_path)
    documents.extend(cv_loader.load())
    web_loader = SeleniumURLLoader(urls=urls)
    documents.extend(web_loader.load())
    text_splitter = CharacterTextSplitter(
        chunk_size=1000, chunk_overlap=30, separator="\n"
    )
    docs = text_splitter.split_documents(documents=documents)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("faiss_index_career")

    chat = ChatOpenAI(verbose=True, temperature=0)
    new_vectorstore = FAISS.load_local("faiss_index_career", embeddings)
    chat = ChatOpenAI(verbose=True, temperature=0)
    qa = ConversationalRetrievalChain.from_llm(
        llm=chat,
        retriever=new_vectorstore.as_retriever(),
        return_source_documents=True,
    )
    res = qa({"question": query, "chat_history": chat_history})
    return res


if __name__ == "__main__":
    print(run_llm(query="What is Minjoo most strong at?", chat_history=[]))
