from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import os

load_dotenv()

CHROMA_PATH = "./chroma_db"
KNOWLEDGE_PATH = "./knowledge_base"

def load_knowledge_base():
    loader = DirectoryLoader(
        KNOWLEDGE_PATH,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} documents")
    return documents

def create_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=40
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")
    return chunks

def create_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    print("Knowledge base saved.")
    return vectorstore

def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )
    return vectorstore

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def build_spark(vectorstore):
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.3,
        api_key=os.getenv("GROQ_API_KEY")
    )

    prompt = ChatPromptTemplate.from_template("""
You are Spark, an AI skill coach for corporate professionals in India.
Your job is to help employees learn and grow in the flow of their work.
Be practical, warm, and specific to the Indian workplace where relevant.

If the context below does not cover the question well, say honestly:
"I don't have specific guidance on that yet — I'd suggest speaking
with your manager or HR directly."

Never make up information. Never give medical, legal, or financial advice.

Context:
{context}

Question: {question}

Answer (3 to 5 sentences unless more is genuinely needed):
""")

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain

if __name__ == "__main__":
    print("Building Spark knowledge base...")
    docs = load_knowledge_base()
    chunks = create_chunks(docs)
    create_vectorstore(chunks)
    print("Done. Spark is ready.")