from app.ingest import extract_text_from_s3
from app.vector_store import chunk_text, create_faiss_index, retrieve_similar_chunks


# Step 1: Get text from S3
bucket_name = "insurance-rag-demo-bucket"
file_name = "sample_claim.pdf"
print(f"📥 Extracting text from {file_name} in {bucket_name}...\n")
full_text = extract_text_from_s3(bucket_name, file_name)

# Step 2: Chunk text
chunks = chunk_text(full_text)
print(f"📄 Total Chunks Created: {len(chunks)}")

# Step 3: Create vector index
index, _, stored_chunks = create_faiss_index(chunks)

# Step 4: Test a query
user_query = "Summarize the claim details"
top_chunks = retrieve_similar_chunks(user_query, index, stored_chunks)

print("\n🔍 Top Relevant Chunks:")
for chunk in top_chunks:
    print(f"- {chunk}\n")

from app.rag_query import query_bedrock_claude

# Format the prompt for Claude
rag_prompt = (
    "You are an intelligent assistant that summarizes insurance claim documents.\n\n"
    "Here is the relevant extracted text from the document:\n\n"
    + "\n\n".join(top_chunks)
    + "\n\nPlease provide a clear and concise summary of this claim."
)

# Query Claude via Bedrock
print("\n🤖 Claude's Summary:\n")
print(query_bedrock_claude(rag_prompt))
