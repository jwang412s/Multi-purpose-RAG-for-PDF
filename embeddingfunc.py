from langchain_community.embeddings.ollama import OllamaEmbeddings

#ollama embeddings models: "mxbai-embed-large", "nomic-embed-text", "all-minilm"	
def get_embedding_function():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings