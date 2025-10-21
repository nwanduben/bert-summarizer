## BERT Extractive Text Summarizer

An interactive NLP app that uses BERT sentence embeddings and cosine similarity to automatically summarize long documents while preserving key meaning.
Built with Streamlit, Sentence-Transformers, and Scikit-learn.


Live Demo

https://bert-summarizer-ho95.streamlit.app/


---

##  Features

- **Model** — Sentence-BERT (all-MiniLM-L6-v2) for sentence embeddings.  
- **Approach** — Extractive summarization using cosine similarity to rank key sentences.  
- **Dataset Used** — BBC News Summary Dataset (Kaggle).  
- **Evaluation** — ROUGE-1, ROUGE-2, and ROUGE-L  
- **Deployment** — Streamlit

---





##  Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/bert-summarizer.git
cd bert-summarizer
```

### 2️⃣ Create a Virtual Environment
```bash
python -m venv env
source env/bin/activate  # On macOS/Linux
env\Scripts\activate    # On Windows
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Streamlit App
```bash
streamlit run streamlit_app/app.py
```

---

### How It Works

- The app splits the document into sentences using NLTK.

- Generates embeddings for each sentence using Sentence-BERT.

- Computes pairwise cosine similarity to measure importance.

- Selects the top sentences (based on a user-set compression ratio).

-Displays the concise summary with optional ROUGE evaluation against a reference summary

## 📢 Contributing
Feel free to fork the repository, create a new branch, and submit a pull request with your improvements!








## Authors

- [@nwanduben](https://www.github.com/nwanduben)

