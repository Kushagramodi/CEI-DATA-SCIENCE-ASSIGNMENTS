import csv
import io
import streamlit as st
from vectorstore import VectorStore
from chatbot import Chatbot


def main():
    st.set_page_config(page_title="RAG Document QA", page_icon="🤖", layout="wide")
    st.title("Document QA Bot 🤖")
    st.write("Upload a PDF, input your API keys, and ask questions!")

    # ── Session state ────────────────────────────────────────────────────
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    if "chatbot" not in st.session_state:
        st.session_state["chatbot"] = None
    if "vectorstore" not in st.session_state:
        st.session_state["vectorstore"] = None

    # ── Sidebar ──────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("API Keys 🔑")
        cohere_api_key   = st.text_input("Cohere API Key",   type="password")
        pinecone_api_key = st.text_input("Pinecone API Key", type="password")

        st.divider()
        st.header("⚙️ Settings")
        chunk_size  = st.slider("Chunk Size", 300, 2000, 1000, 100)
        use_rerank  = st.toggle("Use Reranking", value=True)
        use_hybrid = st.toggle("Use Hybrid Search", value=True)
        embed_model = st.selectbox("Embedding Model", [
            "embed-english-v3.0",
            "embed-english-light-v3.0",
            "embed-multilingual-v3.0",
        ])
        chat_model = st.selectbox("Chat Model", [
            "command-a-03-2025",
            "command-r-plus-08-2024",
            "command-r-08-2024",
        ])

        st.divider()
        if st.button("🗑️ Clear Chat"):
            st.session_state["chat_history"] = []
            st.rerun()

        # CSV export — shown only when chat history exists
        if st.session_state["chat_history"]:
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(["Question", "Answer"])
            for q, a, _ in st.session_state["chat_history"]:
                writer.writerow([q, a])
            st.download_button(
                label="📥 Export Chat (CSV)",
                data=output.getvalue(),
                file_name="chat_history.csv",
                mime="text/csv",
            )

    # ── Main area ────────────────────────────────────────────────────────
    source_type = st.radio("Input Source", ["PDF", "TXT File", "HuggingFace Dataset"])
    uploaded_file = None
    hf_dataset = ""

    if source_type in ("PDF", "TXT File"):
        ext = "pdf" if source_type == "PDF" else "txt"
        uploaded_file = st.file_uploader(f"Upload a {source_type}", type=[ext])
    else:
        hf_dataset = st.text_input(
            "HuggingFace Dataset Name",
            placeholder="e.g. rajpurkar/squad"
        )
        st.caption(
            "ℹ️ Use `namespace/name` format. Try: `rajpurkar/squad`, `vectara/open_ragbench`")
    user_query = st.text_input("Ask a question based on the document")

    if st.button("Submit"):
        # Validation checks before processing
        if source_type in ("PDF", "TXT File") and not uploaded_file:
            st.warning(f"⚠️ Please upload a {source_type}.")
        elif source_type == "HuggingFace Dataset" and not hf_dataset.strip():
            st.warning("⚠️ Please enter a HuggingFace dataset name.")
        elif not cohere_api_key:
            st.warning("⚠️ Please enter your Cohere API key.")
        elif not pinecone_api_key:
            st.warning("⚠️ Please enter your Pinecone API key.")
        elif not user_query.strip():
            st.warning("⚠️ Please enter a question before submitting.")
        else:
            source_label = "PDF" if source_type == "PDF" else "TXT" if source_type == "TXT File" else "HuggingFace Dataset"
            with st.spinner(f"Processing {source_label} — extracting, chunking, embedding..."):
                if source_type == "PDF":
                    path = "uploaded_document.pdf"
                    with open(path, "wb") as f:
                        f.write(uploaded_file.read())

                elif source_type == "TXT File":
                    path = "uploaded_document.txt"
                    with open(path, "wb") as f:
                        f.write(uploaded_file.read())

                else:
                    path = f"hf://{hf_dataset}"  # VectorStore handles this

                vectorstore = VectorStore(
                    path, cohere_api_key,
                    pinecone_api_key, chunk_size, use_rerank, embed_model, use_hybrid
                )
                chatbot = Chatbot(vectorstore, cohere_api_key, chat_model)

                # Store in session state so they persist across reruns
                st.session_state["vectorstore"] = vectorstore
                st.session_state["chatbot"]     = chatbot

                st.success(f"✅ Indexed {len(vectorstore.chunks)} chunks successfully!")

            # ── Criteria 2: Validation Logs ──────────────────────────
            st.subheader("📋 Validation Logs")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Text Length", f"{len(vectorstore.pdf_text)} chars")
            with col2:
                st.metric("Total Chunks", len(vectorstore.chunks))
            with col3:
                st.metric("Embeddings Created", len(vectorstore.embeddings))

            with st.expander("🔍 Sample Chunks"):
                for i, chunk in enumerate(vectorstore.chunks[:3]):
                    st.text_area(f"Chunk {i+1}", chunk, height=80, disabled=True)

            chunk_lengths = [len(c) for c in vectorstore.chunks]
            with st.expander("📊 Chunk Distribution"):
                st.write(
                    f"Min: {min(chunk_lengths)} | "
                    f"Max: {max(chunk_lengths)} | "
                    f"Avg: {int(sum(chunk_lengths)/len(chunk_lengths))} chars"
                )

            # ── Criteria 3: System Metrics ────────────────────────────
            st.sidebar.divider()
            st.sidebar.subheader("📊 System Metrics")
            st.sidebar.markdown(f"""
| Parameter | Value |
|---|---|
| **Chunk Size** | {chunk_size} |
| **Total Chunks** | {len(vectorstore.chunks)} |
| **Embedding Model** | {embed_model} |
| **Embed Dimensions** | {"384" if "light" in embed_model else "1024"} |
| **Vector Store** | Pinecone Serverless |
| **Similarity Metric** | Cosine |
| **Retrieve Top-K** | {vectorstore.retrieve_top_k} |
| **Rerank Top-K** | {vectorstore.rerank_top_k} |
| **Reranking** | {"✅ ON" if use_rerank else "❌ OFF"} |
| **Hybrid Search** | {"✅ ON" if use_hybrid else "❌ OFF"} |
| **Chat Model** | {chat_model} |
            """)

            # ── Generate answer ───────────────────────────────────────
            with st.spinner("Retrieving context and generating answer..."):
                response, retrieved_docs = chatbot.respond(user_query)

                accumulated_response = ""
                for event in response:
                    if getattr(event, "event_type", None) == "text-generation":
                        accumulated_response += event.text

                st.session_state["chat_history"].append(
                    (user_query, accumulated_response, retrieved_docs)
                )

    # ── Display chat history ─────────────────────────────────────────────
    if st.session_state["chat_history"]:
        for q, ans, docs in st.session_state["chat_history"]:
            st.write(f"**You:** {q}")
            st.write(f"**Bot:** {ans}")
            with st.expander("📌 Retrieval Validation — Sources used"):
                for i, doc in enumerate(docs):
                    st.caption(f"[{i+1}] {doc['text'][:300]}...")
            st.divider()


if __name__ == "__main__":
    main()
