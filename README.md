# Multi-purpose-RAG-for-PDF


## Overview

This project is a web-based application built using Streamlit that enables users to upload, process, and query PDF documents using a locally hosted Language Model (LLM) integrated with a Retrieval-Augmented Generation (RAG) system. The application utilizes Chroma as the vector store for managing document embeddings, and it allows users to dynamically create, manage, and query multiple document databases.

## Key Features

1. **PDF Upload and Processing**:
   - Users can upload multiple PDF files simultaneously.
   - The application processes the uploaded PDFs to extract text and create embeddings.
   - A new Chroma database is created for each set of uploaded PDFs, named based on the filenames.

2. **Dynamic Database Management**:
   - Users can select from a list of existing Chroma databases.
   - The application provides an option to delete selected Chroma databases directly from the UI.

3. **Query Interface**:
   - Users can enter natural language queries to search the selected Chroma database.
   - The application retrieves relevant document chunks and uses the LLM to generate responses based on the retrieved context.

## Technologies and Libraries Used

- **Streamlit**: A framework for building interactive web applications.
- **Langchain Community**: Libraries for integrating with vector stores and language models.
- **PyPDF2**: A library for PDF file handling and text extraction.
- **Chroma**: A vector store for managing document embeddings.
- **Ollama**: A library for interfacing with the language model.

## Dependencies

The project requires the following Python packages:

- `streamlit`
- `langchain-community`
- `PyPDF2`
- `chroma`
- `ollama`

The project uses LLM Llama3, and uses the model nomic-embed-text for embeddings. To install, visit https://ollama.com/download and https://ollama.com/blog/embedding-models, and follow the installation guide.

## Installation

To set up the project, follow these steps:

1. **Clone the Repository** (if applicable):
   ```bash
   git clone <repository_url>
   cd <repository_directory>

2. **Visit the Ollama website and download ollama** (if applicable):
    ```bash
    ollama run llama3
    ollama pull nomic-embed-text
    ollama serve

note: you can use any model you want, just make sure to change the model specified in main.py and embeddingfun.py

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt

4. **Run Application**:
   ```bash
   streamlit run main.py

## How to Use
1. **Upload PDFs**:
   - Use the file uploader in the web interface to upload PDF documents.
2. **Process PDFs**:
    - Click the "Process PDFs" button to process the uploaded documents and create a new Chroma database.
3. **Manage Databases**:
   - Select an existing Chroma database from the dropdown menu. Use the "Delete Selected Chroma Database" button to delete any selected database.
4. **Query Database**:
   - Enter a query in natural language and click the "Submit" button to retrieve information from the selected Chroma database.
