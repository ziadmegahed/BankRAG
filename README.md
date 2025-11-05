# BankRAG

BankRAG is a **Retrieval-Augmented Generation (RAG)** application built using **FastAPI** and **LangChain**.  
It allows users to ask questions about bank-related documents and retrieves relevant information from a vector database to generate accurate answers.  

---

## Features

- Query bank documents with natural language questions.
- Retrieve relevant documents using **Chroma vector store**.
- Generate precise answers with context using **LLMs**.
- Web interface powered by **FastAPI** and **Jinja2 templates**.
- Returns both the generated answer and the source document for reference.

---

## Technologies Used

- **Python 3.12**
- **FastAPI** – Web framework for building the API.
- **LangChain / langchain-core** – For LLM orchestration and RAG chains.
- **Chroma** – Vector database for document retrieval.
- **Jinja2** – HTML templating engine.
- **JSONResponse** – Structured JSON responses for frontend consumption.

---

## Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/BankRAG.git
cd BankRAG