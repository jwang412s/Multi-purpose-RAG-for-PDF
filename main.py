import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.llms.ollama import Ollama
from langchain.prompts import ChatPromptTemplate
from embeddingfunc import get_embedding_function
import os
import subprocess
import shutil

# Set page config as the first Streamlit command
st.set_page_config(page_title="Dynamic RAG Application", layout="wide")

# Constants
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

DATA_DIR = "data"
CHROMA_BASE_PATH = "chroma"

def query_rag(query_text: str, chroma_path: str):
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=chroma_path, embedding_function=embedding_function)

    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    
    model = Ollama(model="llama3")
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    return formatted_response

def upload_and_process_pdfs(uploaded_files):
    # Generate a unique Chroma path based on the uploaded file names
    chroma_name = "_".join([os.path.splitext(file.name)[0] for file in uploaded_files])
    chroma_path = os.path.join(CHROMA_BASE_PATH, chroma_name)
    os.makedirs(chroma_path, exist_ok=True)
    
    # Clear the data directory
    if os.path.exists(DATA_DIR):
        shutil.rmtree(DATA_DIR)
    os.makedirs(DATA_DIR)
    
    # Save uploaded files
    for uploaded_file in uploaded_files:
        file_path = os.path.join(DATA_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    
    # Run database.py to process the uploaded PDFs
    subprocess.run(["python", "database.py", "--data_path", DATA_DIR, "--chroma_path", chroma_path], check=True)
    return chroma_path

def delete_chroma(chroma_path):
    try:
        # Attempt to remove the directory
        if os.path.exists(chroma_path):
            shutil.rmtree(chroma_path)
            return True
    except Exception as e:
        st.error(f"Failed to delete Chroma database. Error: {str(e)}")
        return False

# Streamlit app
st.title("Dynamic RAG Application with Locally Hosted LLM")

# Upload and Process PDFs Section
st.header("Upload and Process PDFs")
uploaded_files = st.file_uploader("Upload PDF files", accept_multiple_files=True, type="pdf")
if st.button("Process PDFs"):
    if uploaded_files:
        with st.spinner('Processing uploaded PDFs...'):
            chroma_path = upload_and_process_pdfs(uploaded_files)
        st.success(f"PDFs processed successfully! New database created at {chroma_path}")
    else:
        st.error("Please upload at least one PDF file.")

# Select and Manage Chroma Database Section
st.header("Manage Chroma Databases")
chroma_dirs = [os.path.join(CHROMA_BASE_PATH, d) for d in os.listdir(CHROMA_BASE_PATH) if os.path.isdir(os.path.join(CHROMA_BASE_PATH, d))]
selected_chroma_dir = st.selectbox("Select Chroma Database", chroma_dirs)

if st.button("Delete Selected Chroma Database"):
    if selected_chroma_dir:
        with st.spinner('Deleting selected Chroma database...'):
            success = delete_chroma(selected_chroma_dir)
        if success:
            st.success(f"Chroma database {selected_chroma_dir} deleted successfully!")
            chroma_dirs = [os.path.join(CHROMA_BASE_PATH, d) for d in os.listdir(CHROMA_BASE_PATH) if os.path.isdir(os.path.join(CHROMA_BASE_PATH, d))]
            st.experimental_rerun()  # Refresh the UI to reflect the deletion
        else:
            st.error("Failed to delete Chroma database.")

# Query Section
st.header("Query Chroma Database")
query_text = st.text_input("Enter your query:")
if st.button("Submit"):
    if query_text and selected_chroma_dir:
        with st.spinner('Searching and generating response...'):
            response = query_rag(query_text, selected_chroma_dir)
        st.text_area("Response:", value=response, height=200)
    else:
        st.error("Please enter a query and select a Chroma database.")
