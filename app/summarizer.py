from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
import numpy as np
import nltk

nltk.download("punkt")
nltk.download("punkt_tab")

# Load BERT model
model = SentenceTransformer("all-MiniLM-L6-v2")

def summarize_text(text: str, compression_ratio: float = 0.3):
    sentences = sent_tokenize(text)
    if len(sentences) < 3:
        return text
    
    embeddings = model.encode(sentences)
    sim_matrix = cosine_similarity(embeddings)
    scores = sim_matrix.sum(axis=1)

    n_sentences = max(1, int(len(sentences) * compression_ratio))
    top_idx = np.argsort(scores)[-n_sentences:]
    top_idx.sort()

    summary = " ".join([sentences[i] for i in top_idx])
    return summary
