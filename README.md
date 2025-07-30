# ScholarSphere 🎓

AI-powered Academic Assistant for document analysis, summarization, and quiz generation.

## Features

- **Document Summarization**: Generate concise summaries from PDFs, DOCX, and TXT files
- **Similarity Analysis**: Compare documents and visualize relationships
- **Quiz Generator**: Create multiple-choice questions from documents
- **Research Explorer**: Extract key topics and find related papers

## Live Demo

🚀 [Try ScholarSphere on Hugging Face Spaces](https://huggingface.co/spaces/YOUR_USERNAME/scholarsphere)

## Local Installation

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Technologies Used

- Streamlit for the web interface
- Transformers (BART, T5) for summarization and question generation
- Sentence Transformers for similarity analysis
- KeyBERT for keyword extraction
- Semantic Scholar API for research paper recommendations

## NOTE

- In this i made  use of Hugging face and other ML models offline on my pc (Locally) : IMPROV this and access them via API.
- Once Done DEPLOY IT: via RENDER (Backend) and VERCEL (FrontEnd)
