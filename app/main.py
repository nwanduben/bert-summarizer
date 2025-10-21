from fastapi import FastAPI, UploadFile, File, Form
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader
from nltk.tokenize import sent_tokenize
import nltk
import numpy as np

# Ensure NLTK resources are ready
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# Initialize FastAPI app
app = FastAPI(title="BERT Extractive Summarizer API")

# Load BERT model for sentence embeddings
model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

# Core Summarization Function
def summarize_text(text: str, compression_ratio: float = 0.3) -> str:
    sentences = sent_tokenize(text)
    if len(sentences) < 3:
        return text.strip()

    # Compute embeddings for each sentence
    embeddings = model.encode(sentences)
    sim_matrix = cosine_similarity(embeddings)

    # Calculate sentence scores
    scores = sim_matrix.sum(axis=1)

    # Select top N sentences based on compression ratio
    n_sentences = max(1, int(len(sentences) * compression_ratio))
    top_idx = np.argsort(scores)[-n_sentences:]
    top_idx.sort()

    summary = " ".join([sentences[i] for i in top_idx])
    return summary.strip()

# Endpoint: Text Summarization
@app.post("/summarize")
async def summarize_text_endpoint(
    text: str = Form(...),
    ratio: float = Form(0.3)
):
    """Summarize plain text input."""
    summary = summarize_text(text, ratio)
    clean_summary = summary.replace("\n", " ").strip()
    return {"summary": clean_summary}


# Endpoint: PDF Upload Summarization
@app.post("/summarize/pdf")
async def summarize_pdf(file: UploadFile = File(...), ratio: float = Form(0.3)):
    """Summarize text extracted from a PDF file."""
    pdf_reader = PdfReader(file.file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"

    summary = summarize_text(text, ratio)
    clean_summary = summary.replace("\n", " ").strip()
    return {"summary": clean_summary}


# Health Check
@app.get("/")
async def root():
    return {"message": "BERT Summarizer API is running!"}
