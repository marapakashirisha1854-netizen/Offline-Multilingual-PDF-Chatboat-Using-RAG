Offline Multilingual PDF Chatbot Using RAG

An AI-powered chatbot that allows users to upload PDF documents and ask questions about their content in English, Telugu, and Hindi. The application uses Retrieval-Augmented Generation (RAG) and locally running AI models to provide answers with relevant document sources.

Features

- Upload and process PDF documents.
- Ask questions about uploaded documents.
- Multilingual interaction: English, Telugu, and Hindi.
- Retrieve relevant information using vector search.
- Generate answers using a locally running Large Language Model (LLM).
- Display relevant document sources.
- Support text extraction and OCR for scanned PDFs, depending on local configuration.
- Designed for local, offline processing after the required models and dependencies have been installed.

Technologies Used

- Python
- FastAPI
- HTML, CSS, JavaScript
- Retrieval-Augmented Generation (RAG)
- ChromaDB
- Sentence Transformers
- Ollama
- PDF text extraction and OCR tools

Project Workflow

1. Upload PDF documents.
2. Extract text from the PDFs.
3. Split the extracted text into chunks.
4. Generate embeddings for the text chunks.
5. Store embeddings in ChromaDB.
6. Retrieve relevant chunks based on the user's question.
7. Generate an answer using a local language model.
8. Display the answer and relevant sources.

Project Structure

Offline-Multilingual-PDF-Chatbot-Using-RAG/
├── backend/
├── frontend/
├── app.py
├── requirements.txt
├── LOCAL_SETUP.md
└── README.md

Setup Instructions

1. Install Python and the required local AI tools.
2. Install the Python dependencies listed in "requirements.txt".
3. Configure Ollama and download the required language models.
4. Configure any required OCR and speech models.
5. Follow the detailed instructions in "LOCAL_SETUP.md".
6. Start the application using the configured startup command.

Offline Capability

The application is designed to run locally using downloaded dependencies and AI models. Initial setup may require an internet connection to install packages and download models. Once everything required is available locally, supported features can run without an internet connection.

Purpose

This project demonstrates the practical use of RAG, embeddings, vector databases, local language models, and multilingual document question answering.

Author

M. Shirisha

B.Tech – Computer Science and Engineering
