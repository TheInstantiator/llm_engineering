# RAG Project Overview

This document summarizes the core components of the RAG (Retrieval-Augmented Generation) system implemented in this directory. It serves as a blueprint for building your own knowledge-based AI applications.

## 1. `ingest.py` (The Knowledge Builder)
**Purpose:** Prepares your data pipeline. It takes raw text files and converts them into a searchable mathematical format (vectors) stored in a database.

**How it works:**
1.  **Loading:** It scans the `knowledge-base` folder for markdown files.
2.  **Agentic Chunking (Smart Splitting):** Instead of blindly chopping text every 500 characters, it uses a small, fast LLM (`gpt-4.1-nano`) to intelligently break documents into logical sections.
    *   It creates a **Headline** and **Summary** for each chunk to improve searchability.
3.  **Embedding:** It sends these text chunks to OpenAI's `text-embedding-3-large` model to Create "embeddings" (lists of numbers representing semantic meaning).
4.  **Storage:** It saves these embeddings and metadata into **ChromaDB**, a vector database that allows for fast similarity searching later.
5.  **Parallelism:** It uses multiprocessing to run these tasks on multiple files effectively at once.

**Notes:**
*  To make it more professional it needs to not be a nuke and reset.  It needs to know what it has seen before and not reprocess it. 
* It needs to handle duplicates and not process the same file multiple times.
* It needs to handle the case where the knowledge base is updated and the system needs to reprocess the files.
* It needs to handle the case where the knowledge base is deleted and the system needs to reprocess the files.
---

## 2. `answer.py` (The Chatbot Brain)
**Purpose:** The intelligent engine that takes a user's question, finds the right data, and generates an answer.

**How it works:**
1.  **Query Rewriting (Context Awareness):**
    *   If you ask "What is her salary?", the system looks at chat history to figure out who "her" is (e.g., "Avery").
    *   It rewrites the query to "What is Avery Lancaster's salary?" for better database searching.
2.  **Hybrid Retrieval:**
    *   It searches the database for *both* the original question and the rewritten question to ensure nothing is missed (`RETRIEVAL_K = 20` results).
3.  **Reranking (Quality Control):**
    *   Vector search is fast but sometimes imprecise. The system takes the top 20 results and asks an LLM to strictly rank them by relevance, keeping only the best 10 (`FINAL_K = 10`).
4.  **Answer Generation:**
    *   It combines the **Persona** (System Prompt), **Retrieved Data** (Context), **Chat History**, and **User Question**.
    *   It sends this package to a high-quality LLM (`gpt-oss-120b`) to generate the final, accurate response.

---

## 3. `app.py` (The User Interface)
**Purpose:** Provides a web-based chat interface so users can easily interact with the RAG system without writing code.

**How it works:**
1.  **Gradio:** Uses the `gradio` library to build a clean web UI with just a few lines of Python.
2.  **Split View:** Creates a two-column layout:
    *   **Left Column:** The Chatbot window for the conversation.
    *   **Right Column:** A "Context" window that shows exactly what documents the AI retrieved to answer your question (great for transparency/debugging).
3.  **Flow:**
    *   User types a message -> `put_message_in_chatbot` updates the UI instantly.
    *   The system calls `chat` -> which calls `answer.py` -> which does the heavy lifting.
    *   The UI updates with the AI's answer AND the source documents found.
