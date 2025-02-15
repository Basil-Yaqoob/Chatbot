import os
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
import pathlib
from langchain.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

load_dotenv()

open_ai_api_key = os.getenv("OPENAI_API_KEY")

LLM = ChatOpenAI(
    model_name="gpt-4o-mini", temperature=0, streaming=True, openai_api_key=open_ai_api_key
)

def load_document(temp_filepath):
    ext = pathlib.Path(temp_filepath).suffix
    loaded = PyPDFLoader(temp_filepath)
    docs = loaded.load()
    return docs

def init_memory():
    return ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True,
        output_keys='answer'
    )

MEMORY = init_memory()

def configure_retriever(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    embedding = OpenAIEmbeddings(openai_api_key=open_ai_api_key)
    vectordb = DocArrayInMemorySearch.from_documents(splits, embedding)

    retriever = vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 5,
            "fetch_k": 7,
            "include_metadata": True
        }
    )
    return retriever

def configure_chain(retriever):
    params = dict(
        llm=LLM, retriever=retriever,
        memory=MEMORY,
        verbose=True,
        max_tokens_limit=4000
    )
    return ConversationalRetrievalChain.from_llm(**params)

def configure_retrieval_chain(uploaded_files):
    docs = []
    temp_dir = "./UserFiles"
    
    for file in uploaded_files:
        temp_filepath = os.path.join(temp_dir, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())
        
        docs.extend(load_document(temp_filepath))
    
    retriever = configure_retriever(docs)
    chain = configure_chain(retriever)
    return chain
