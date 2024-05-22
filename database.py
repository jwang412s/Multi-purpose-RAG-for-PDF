import argparse
import os
import shutil

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.schema.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from embeddingfunc import get_embedding_function

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    parser.add_argument("--data_path", type=str, default="data", help="Path to the data directory.")
    parser.add_argument("--chroma_path", type=str, required=True, help="Path to the Chroma directory.")
    args = parser.parse_args()

    if args.reset:
        print("✨ Clearing Database")
        clear_database(args.chroma_path)

    # Create (or update) the data store.
    documents = load_documents(args.data_path)
    chunks = split_documents(documents)
    add_to_chroma(chunks, args.chroma_path)

def load_documents(data_path):
    document_loader = PyPDFDirectoryLoader(data_path)
    return document_loader.load()

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

def add_to_chroma(chunks: list[Document], chroma_path):
    os.makedirs(chroma_path, exist_ok=True)
    db = Chroma(
        persist_directory=chroma_path, embedding_function=get_embedding_function()
    )

    chunks_with_ids = calculate_chunk_ids(chunks)

    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    new_chunks = [chunk for chunk in chunks_with_ids if chunk.metadata["id"] not in existing_ids]

    if new_chunks:
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
    else:
        print("✅ No new documents to add")

def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        chunk.metadata["id"] = chunk_id

    return chunks

def clear_database(chroma_path):
    if os.path.exists(chroma_path):
        shutil.rmtree(chroma_path)

if __name__ == "__main__":
    main()
