import streamlit as st
import numpy as np
from typing import List, Dict, Tuple
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# --------------- Config ---------------
# Embeddings model: light, fast, good quality
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# Generative model: small enough for CPU demo
GEN_MODEL_NAME = "google/flan-t5-small"

# --------------- Utilities ---------------

def extract_text_from_pdf(file_bytes) -> List[Tuple[int, str]]:
    """
    Read PDF, return list of (page_number, text).
    """
    reader = PdfReader(file_bytes)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        # Basic cleaning
        text = " ".join(text.split())
        if text.strip():
            pages.append((i + 1, text))
    return pages

def chunk_text(pages: List[Tuple[int, str]], chunk_size: int = 1000, overlap: int = 150) -> List[Dict]:
    """
    Split each page's text into overlapping chunks.
    Each chunk keeps metadata: page and span.
    """
    chunks = []
    for page_num, text in pages:
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            if chunk.strip():
                chunks.append({
                    "page": page_num,
                    "start": start,
                    "end": min(end, len(text)),
                    "text": chunk
                })
            # Move forward with overlap
            start += chunk_size - overlap
            if start < 0:
                break
    return chunks

def build_embeddings(chunks: List[Dict], model: SentenceTransformer) -> np.ndarray:
    """
    Compute embeddings for all chunks.
    """
    texts = [c["text"] for c in chunks]
    emb = model.encode(texts, normalize_embeddings=True)  # shape: (n, d)
    return emb

def embed_query(query: str, model: SentenceTransformer) -> np.ndarray:
    """
    Compute embedding for user query.
    """
    return model.encode([query], normalize_embeddings=True)[0]

def cosine_sim_matrix(query_vec: np.ndarray, doc_matrix: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity scores between query and all doc vectors.
    (They are already normalized, so dot product = cosine similarity)
    """
    return doc_matrix @ query_vec

def retrieve_top_k(chunks: List[Dict], doc_matrix: np.ndarray, query_vec: np.ndarray, k: int = 4) -> List[Dict]:
    """
    Return top-k chunks with similarity scores.
    """
    scores = cosine_sim_matrix(query_vec, doc_matrix)
    idx = np.argsort(scores)[::-1][:k]
    results = []
    for i in idx:
        item = dict(chunks[i])
        item["score"] = float(scores[i])
        results.append(item)
    return results

def make_prompt(context_chunks: List[Dict], question: str) -> str:
    """
    Build a strict, grounded prompt for the generative model.
    """
    header = (
        "You are a helpful assistant answering strictly from the provided document context.\n"
        "- If the answer is not found in the context, say: 'I don't know based on the document.'\n"
        "- Cite page numbers when relevant.\n\n"
        "Context:\n"
    )
    ctx_texts = []
    for c in context_chunks:
        ctx_texts.append(f"[Page {c['page']}, score {c['score']:.3f}]\n{c['text']}\n")
    ctx_block = "\n---\n".join(ctx_texts)
    user_q = f"\nQuestion: {question}\n\nAnswer:"
    prompt = header + ctx_block + user_q
    # Truncate if too long for small models
    if len(prompt) > 6000:
        prompt = prompt[-6000:]
    return prompt

# --------------- Streamlit App ---------------

@st.cache_resource(show_spinner=False)
def load_models():
    emb_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    gen_pipe = pipeline("text2text-generation", model=GEN_MODEL_NAME)
    return emb_model, gen_pipe

def main():
    st.set_page_config(page_title="PDF Q&A (RAG) Demo", page_icon="📄", layout="wide")
    st.title("📄 PDF Q&A with Generative AI (RAG)")

    st.markdown(
        "Upload a PDF, then ask any question grounded in the document. "
        "The app retrieves relevant passages and generates an answer."
    )

    # Sidebar settings
    with st.sidebar:
        st.header("Settings")
        chunk_size = st.slider("Chunk size (characters)", 600, 1600, 1000, 50)
        overlap = st.slider("Chunk overlap", 50, 400, 150, 10)
        top_k = st.slider("Top-k retrieved chunks", 2, 8, 4, 1)
        temperature = st.slider("Generation temperature (0 = factual)", 0.0, 1.0, 0.2, 0.1)
        max_tokens = st.slider("Max generated tokens", 64, 512, 256, 32)

    emb_model, gen_pipe = load_models()

    st.subheader("Step 1: Upload PDF")
    file = st.file_uploader("Upload a single PDF", type=["pdf"])
    if file:
        # Extract and chunk
        pages = extract_text_from_pdf(file)
        if not pages:
            st.warning("No extractable text found in the PDF. Try a text-based PDF (not scanned) or use OCR.")
            return

        st.success(f"Parsed {len(pages)} pages.")
        chunks = chunk_text(pages, chunk_size=chunk_size, overlap=overlap)
        st.info(f"Created {len(chunks)} chunks.")

        # Embeddings
        with st.spinner("Computing embeddings..."):
            doc_matrix = build_embeddings(chunks, emb_model)

        st.subheader("Step 2: Ask a question")
        question = st.text_input("Enter your question about the document")

        if question:
            q_vec = embed_query(question, emb_model)
            top_chunks = retrieve_top_k(chunks, doc_matrix, q_vec, k=top_k)

            # Show retrieved sources
            with st.expander("Show retrieved sources"):
                for i, c in enumerate(top_chunks, 1):
                    st.markdown(f"**Source {i} — Page {c['page']} (score {c['score']:.3f})**")
                    st.caption(f"Span: {c['start']}–{c['end']}")
                    st.write(c["text"])

            # Build prompt and generate answer
            prompt = make_prompt(top_chunks, question)
            with st.spinner("Generating grounded answer..."):
                out = gen_pipe(
                    prompt,
                    do_sample=(temperature > 0.0),
                    temperature=max(temperature, 1e-6),
                    max_new_tokens=max_tokens,
                )
            answer = out[0]["generated_text"]

            st.subheader("Answer")
            st.write(answer)

            # Trust signals
            st.markdown("— Answer generated using retrieved document context.")
    else:
        st.info("Upload a PDF to begin.")

if __name__ == "__main__":
    main()