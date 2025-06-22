import streamlit as st
import boto3
import json
import tempfile
import os
from app.ingest import extract_text_from_s3
from app.vector_store import chunk_text, create_faiss_index, retrieve_similar_chunks
from app.rag_query import query_bedrock_claude

# ---- S3 Settings ----
BUCKET_NAME = "insurance-rag-demo-bucket"

# ---- Streamlit UI ----
st.set_page_config(page_title="Insurance Claim GenAI", layout="centered")
st.title("📄 Insurance Claim Summarizer & Q&A")

uploaded_file = st.file_uploader("Upload a PDF claim document", type=["pdf"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        local_path = tmp.name

    s3 = boto3.client("s3")
    s3_key = uploaded_file.name
    s3.upload_file(local_path, BUCKET_NAME, s3_key)
    os.remove(local_path)

    st.success(f"✅ Uploaded to S3: `{s3_key}`")
    if st.button("🧠 Generate Summary"):
        # Step 1: Extract & chunk
        full_text = extract_text_from_s3(BUCKET_NAME, s3_key)
        chunks = chunk_text(full_text)
        index, _, stored_chunks = create_faiss_index(chunks)

        query = "Summarize the claim details"
        top_chunks = retrieve_similar_chunks(query, index, stored_chunks)

        rag_prompt = (
            "You are an intelligent assistant that summarizes insurance claim documents.\n\n"
            "Here is the relevant extracted text from the document:\n\n"
            + "\n\n".join(top_chunks)
            + "\n\nPlease provide a clear and concise summary of this claim."
        )

        with st.spinner("Claude is summarizing..."):
            response = query_bedrock_claude(rag_prompt)
        st.subheader("📋 Summary:")
        st.write(response)

    # Optional Q&A
    user_question = st.text_input("💬 Ask a question about the claim")
    if user_question:
        full_text = extract_text_from_s3(BUCKET_NAME, s3_key)
        chunks = chunk_text(full_text)
        index, _, stored_chunks = create_faiss_index(chunks)
        top_chunks = retrieve_similar_chunks(user_question, index, stored_chunks)

        q_prompt = (
            f"You are a helpful assistant. Based on the claim text below, answer the question.\n\n"
            f"Document Content:\n{chr(10).join(top_chunks)}\n\n"
            f"Question: {user_question}\n\nAnswer:"
        )

        with st.spinner("Claude is answering..."):
            answer = query_bedrock_claude(q_prompt)
        st.subheader("🧠 Claude's Answer:")
        st.write(answer)
