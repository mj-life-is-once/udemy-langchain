import os

import pinecone
from dotenv import dotenv_values
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms.openai import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.pinecone import Pinecone

config = dotenv_values(".env")
os.environ["OPENAI_API_KEY"] = config["OPENAI_API_KEY"]
os.environ["PINECONE_API_KEY"] = config["PINECONE_API_KEY"]

pinecone.init(api_key=os.environ["PINECONE_API_KEY"], environment="gcp-starter")

if __name__ == "__main__":
    loader = TextLoader("./mediumblogs/mediumblog1.txt")
    document = loader.load()

    # chunk_size , chunk_overlap are important parameters to tune
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    # print(len(texts))

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
    docsearch = Pinecone.from_documents(
        texts, embeddings, index_name="medium-blogs-embeddings-index"
    )

    # RetrievalQA is a class that combines LLM and retriever
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever()
    )
    query = "What is a vector DB? Give me a 15 word answer from a beginner"
    result = qa({"query": query})
    print(result)
