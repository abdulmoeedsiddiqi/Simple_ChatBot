# Simple_ChatBot
📘 Introduction to AI – RAG with LangChain + Google Gemini

This project implements a Retrieval-Augmented Generation (RAG) pipeline using LangChain and Google Gemini.

It uses a JSON file 📄 "Introduction_to_AI.json" as the knowledge base and allows you to ask AI-related questions such as:

What is supervised learning?

What is meant by labeled data?

The system retrieves the relevant context from the file and generates accurate answers using Gemini.

🚀 Features

✅ Load and process JSON data
✅ Chunk documents for better LLM understanding
✅ Create embeddings using GoogleGenerativeAIEmbeddings
✅ Store embeddings in ChromaDB for efficient retrieval
✅ Use a Retriever to fetch top relevant chunks
✅ Build a RetrievalQA Chain with Gemini
✅ Custom Prompt Template to restrict answers only to available context

⚙️ Installation

Install dependencies:

pip install -U langchain langchain-community langchain-core langchain-google-genai chromadb

🔑 Environment Setup

Set your Google Gemini API Key:

import os
os.environ["GOOGLE_API_KEY"] = "your_api_key_here"


Replace "your_api_key_here" with your actual Gemini API key.

📂 Project Workflow

📌 Step 1 – Load JSON file

from langchain_community.document_loaders import JSONLoader
file_path = "/content/Introduction_to_Ai.json"


📌 Step 2 – Chunk the data

from langchain.text_splitter import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)


📌 Step 3 – Generate embeddings

from langchain_google_genai import GoogleGenerativeAIEmbeddings
embedder = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


📌 Step 4 – Store in ChromaDB

from langchain_community.vectorstores import Chroma
vector_db = Chroma.from_documents(documents=doc_chunks, embedding=embedder)


📌 Step 5 – Set up Retriever + LLM

from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
retriever = vector_db.as_retriever(search_kwargs={"k": 3})
gemini_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-thinking-exp-01-21", temperature=0.2)


📌 Step 6 – Custom Prompt + QA Chain

from langchain.prompts import PromptTemplate
rag_prompt = """
You are an expert on AI. Use the context below to answer questions.
Context: {context}
Question: {question}
Answer:
"""
custom_prompt = PromptTemplate(template=rag_prompt, input_variables=["context", "question"])
qa_chain = RetrievalQA.from_chain_type(llm=gemini_llm, retriever=retriever, chain_type_kwargs={"prompt": custom_prompt})


📌 Step 7 – Ask Questions

def Machine_Learning_Question(question):
    response = qa_chain.invoke({"query": question})
    print(response['result'])

questions = [
    "What is supervised learning?",
    "What is meant by labeled data?"
]

for q in questions:
    Machine_Learning_Question(q)

🧠 Example Output
Question: What is supervised learning?
Answer: Supervised learning is a type of machine learning where the model is trained using labeled data.

📌 Tech Stack

LangChain – RAG framework

Google Gemini API – LLM & embeddings

ChromaDB – Vector database

Python – Implementation language

🌟 Future Improvements

Add support for multiple datasets

Build a user-friendly front-end for querying

Enhance prompt engineering for better accuracy

🙌 Contribution

Pull requests are welcome! If you’d like to suggest features or report issues, please open an issue.

📜 License

This project is licensed under the MIT License.
