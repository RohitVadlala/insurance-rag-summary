## 🛡️ Insurance RAG Summary – AI-Powered Document Summarization App

This is an end-to-end Retrieval-Augmented Generation (RAG) system built for summarizing insurance documents using Amazon Bedrock (Claude Instant), FAISS vector search, and a Streamlit frontend. The application ingests PDF claim files, extracts and chunks the content, stores embeddings in FAISS, and allows users to ask contextual questions with instant answers.

## 🚀 Features

- 🔍 **Semantic Search** using MiniLM + FAISS
- 🤖 **LLM-based Answers** via Claude Instant from Amazon Bedrock
- 📄 **PDF Ingestion** and pre-processing pipeline
- 🧠 **Contextual Querying** using Retrieval-Augmented Generation (RAG)
- 🌐 **Streamlit UI** for interactive user experience
- 🐳 **Dockerized** for portable deployment

## 🧱 Tech Stack

| Layer            | Tools / Services                                      |
|------------------|--------------------------------------------------------|
| LLM Backend      | Amazon Bedrock (Claude Instant)                        |
| Embedding Model  | `all-MiniLM-L6-v2` (SentenceTransformers)              |
| Vector DB        | FAISS                                                  |
| Frontend         | Streamlit                                              |
| Containerization | Docker                                                 |
| Deployment Ready | GitHub + Dockerfile                                    |


 
