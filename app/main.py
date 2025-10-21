import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
from rouge import Rouge
import numpy as np
import nltk
from PyPDF2 import PdfReader
import docx

# --- Setup ---
st.set_page_config(page_title="BERT Extractive Summarizer", layout="wide")
st.title("🧠 BERT Extractive Text Summarizer")

nltk.download("punkt")
nltk.download("punkt_tab")

@st.cache_resource
def load_model():
    """Load Sentence-BERT model once for performance."""
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# --- Utility functions ---
def read_pdf(file):
    reader = PdfReader(file)
    text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text

def read_docx(file):
    doc = docx.Document(file)
    text = " ".join([para.text for para in doc.paragraphs])
    return text

def summarize_text(text, compression_ratio=0.3):
    """Extractive summary using cosine similarity and BERT embeddings."""
    sentences = sent_tokenize(text)
    if len(sentences) < 3:
        return text  # skip very short inputs
    
    embeddings = model.encode(sentences)
    sim_matrix = cosine_similarity(embeddings)
    scores = sim_matrix.sum(axis=1)

    n_sentences = max(1, int(len(sentences) * compression_ratio))
    top_idx = np.argsort(scores)[-n_sentences:]
    top_idx.sort()

    summary = " ".join([sentences[i] for i in top_idx])
    return summary


# --- Sidebar options ---
st.sidebar.header("⚙️ Settings")
compression_ratio = st.sidebar.slider("Compression ratio (how short should the summary be)", 0.1, 0.9, 0.3, 0.05)
show_rouge = st.sidebar.checkbox("Show ROUGE evaluation (if reference provided)", value=False)

# --- File upload or text input ---
st.subheader("📄 Input Text or Upload a File")

uploaded_file = st.file_uploader("Upload a document (.txt, .pdf, .docx)", type=["txt", "pdf", "docx"])
user_text = st.text_area("Or paste your text here:", height=200)
reference_summary = st.text_area("Optional: Paste reference summary (for ROUGE evaluation)", height=150)

# --- Read file content if uploaded ---
if uploaded_file:
    file_name = uploaded_file.name.lower()
    if file_name.endswith(".pdf"):
        text = read_pdf(uploaded_file)
    elif file_name.endswith(".docx"):
        text = read_docx(uploaded_file)
    else:
        text = uploaded_file.read().decode("utf-8", errors="ignore")
else:
    text = user_text

# --- Generate Summary ---
if st.button("🚀 Summarize"):
    if not text.strip():
        st.warning("Please enter or upload a document first.")
    else:
        with st.spinner("Generating summary..."):
            summary = summarize_text(text, compression_ratio)
        
        st.subheader("✨ Generated Summary:")
        st.write(summary)

        # --- Optional ROUGE Evaluation ---
        if show_rouge and reference_summary.strip():
            rouge = Rouge()
            scores = rouge.get_scores(summary, reference_summary)
            st.subheader("📊 ROUGE Evaluation")
            st.json(scores[0])
        elif show_rouge:
            st.info("Provide a reference summary above to compute ROUGE scores.")

# --- Footer ---
st.markdown("---")
st.caption("Built with ❤️ using Streamlit and Sentence-BERT | by Emmanuel")
