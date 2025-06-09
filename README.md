# Multi-Modal PDF Chatbot: An Enterprise-Grade RAG System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-green)

An advanced, enterprise-ready RAG (Retrieval-Augmented Generation) system designed to transform dense, multi-modal PDF documents into an interactive conversational experience. This chatbot can parse and understand text, tables, and charts within your documents, allowing users to ask complex questions and receive accurate, context-aware answers.

---

## 🚀 Key Features

* **📄 Multi-Modal Processing:** Ingests and comprehends not just text, but also **tables** and **images/charts** within PDF documents using `unstructured.io`.
* **🧠 Advanced RAG Architecture:** Implements a `MultiVectorRetriever` strategy. It creates concise summaries of document chunks (text, tables, and images) for efficient searching, while retrieving the original, full-detail chunks for final answer generation.
* **🤖 Intelligent Summarization:** Leverages `gpt-4o-mini` to create structured, analytical summaries of every document component, ensuring rich, searchable context.
* **⚡️ High-Performance Backend:** Built with **FastAPI**, providing a robust, scalable, and asynchronous API for all RAG operations.
* ** streamlit  Interactive Frontend:** A clean, professional, and user-friendly chat interface built with **Streamlit**, designed for a seamless user experience.
* **☁️ Cloud-Native & Scalable:** Designed for production deployment with a decoupled frontend/backend architecture, using Render for backend hosting with persistent storage and Streamlit Community Cloud for the UI.

---

## 🛠️ Tech Stack & Architecture

This project uses a modern, decoupled architecture to ensure scalability and maintainability.

* **Frontend:** Streamlit
* **Backend:** FastAPI, Uvicorn
* **LLM & Embeddings:** OpenAI (gpt-4o-mini)
* **Core Logic:** LangChain
* **Vector Database:** ChromaDB
* **PDF Parsing:** `unstructured`
* **Deployment:**
    * Backend on **Render** (with Persistent Disks)
    * Frontend on **Streamlit Community Cloud**

### Architecture Diagram

```mermaid
graph TD
    A[End User] --> B(Streamlit Frontend on Streamlit Cloud);
    B -->|"API Request /query"| C{FastAPI Backend on Render};
    C <--> D[ChromaDB on Render Persistent Disk];
    D -->|Retrieved Summaries| C;
    C -->|"Fetch Full Content"| F["Docstore (.pkl) on Persistent Disk"];
    F -->|Original Text/Image| C;
    C -->|Augmented Prompt| E(OpenAI API);
    E -->|Generated Answer| C;
    C -->|Final Response| B;
