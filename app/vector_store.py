from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Split the full text into chunks
def chunk_text(text, chunk_size=500):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# Create a FAISS vector index
def create_faiss_index(text_chunks):
    embeddings = model.encode(text_chunks)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index, embeddings, text_chunks

# Query the FAISS index
def retrieve_similar_chunks(query, index, text_chunks, k=3):
    query_embedding = model.encode([query])
    _, top_indices = index.search(np.array(query_embedding), k)
    return [text_chunks[i] for i in top_indices[0]]

