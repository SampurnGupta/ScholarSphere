from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader
from docx import Document
import streamlit as st
import io
import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
import time
import os
import hashlib
import pickle
from transformers import pipeline as hf_pipeline
import json
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import random
import nltk
from nltk.corpus import wordnet
from keybert import KeyBERT
import requests

nltk.download("wordnet")

# --- Cache Configuration ---
CACHE_DIR = "cache_embeddings"
os.makedirs(CACHE_DIR, exist_ok=True)

def get_file_hash(file_bytes: bytes) -> str:
    """Generate MD5 hash of file contents for caching"""
    return hashlib.md5(file_bytes).hexdigest()

def load_cached_data(file_hash: str, data_type: str) -> Optional[object]:
    """Load cached data from disk"""
    cache_path = os.path.join(CACHE_DIR, f"{file_hash}_{data_type}.pkl")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            st.warning(f"Cache loading failed: {str(e)}")
            return None
    return None

def save_cached_data(file_hash: str, data_type: str, data: object) -> None:
    """Save data to cache"""
    cache_path = os.path.join(CACHE_DIR, f"{file_hash}_{data_type}.pkl")
    try:
        with open(cache_path, "wb") as f:
            pickle.dump(data, f)
    except Exception as e:
        st.error(f"Failed to cache data: {str(e)}")

# --- Streamlit Config ---
st.set_page_config(
    page_title="ScholarSphere", 
    page_icon="🎓", 
    layout="wide",
    menu_items={
        'Get Help': 'https://github.com/SampurnGupta',
        'Report a bug': "https://github.com/SampurnGupta",
        'About': "# ScholarSphere\nAcademic Writing Assistant"
    }
)

# --- Cache Resources ---
# Modified load_models function
@st.cache_resource(show_spinner=False)
def load_models(use_gpu: bool):
    # 1) detect real CUDA availability
    cuda_available = torch.cuda.is_available()
    # 2) choose device index for transformer pipelines (-1 for CPU)
    device_idx = 0 if use_gpu and cuda_available else -1

    summarizer = hf_pipeline(
        "summarization",
        model="facebook/bart-large-cnn",
        device=device_idx
    )
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    similarity_model = SentenceTransformer('all-MiniLM-L6-v2')

    # quiz components
    quiz_model = None
    quiz_tokenizer = None
    qa_pipeline = None
    try:
        quiz_model_name = "mrm8488/t5-base-finetuned-question-generation-ap"
        quiz_tokenizer = T5Tokenizer.from_pretrained(quiz_model_name)
        quiz_model = T5ForConditionalGeneration.from_pretrained(quiz_model_name)
        # only move to CUDA if it actually exists
        if use_gpu and cuda_available:
            quiz_model.to("cuda")
        else:
            quiz_model.to("cpu")
        
        # add QA pipeline
        qa_pipeline = hf_pipeline("question-answering", device=0 if use_gpu and cuda_available else -1)
    except ImportError:
        st.warning("SentencePiece not installed → quiz disabled. `pip install sentencepiece`")

    # **return both** summarizer, tokenizer, sim-model and a (model,tokenizer) tuple
    return summarizer, tokenizer, similarity_model, (quiz_model, quiz_tokenizer, qa_pipeline)

# Updated quiz generation function
def generate_quiz(text: str, quiz_components, similarity_model, num_questions=5) -> List[Dict]:
    """Generate multiple choice questions from text with real answers & distractors"""
    quiz_model, quiz_tokenizer, qa_pipeline = quiz_components
    questions = []
    text = " ".join(text.split())
    if len(text) < 100:
        return []

    # generate raw questions
    inputs = quiz_tokenizer(
        f"generate questions: {text[:5000]}",
        return_tensors="pt", truncation=True, max_length=512
    ).to(quiz_model.device)
    outputs = quiz_model.generate(
        **inputs,
        max_length=128,
        num_return_sequences=num_questions,
        do_sample=True,
        temperature=0.7
    )

    for output in outputs:
        q = quiz_tokenizer.decode(output, skip_special_tokens=True).strip()
        if not q.endswith("?"):
            q += "?"

        # 1) find the correct answer span via QA
        ans = qa_pipeline(question=q, context=text)["answer"]
        
        # 1) synonyms via WordNet
        syns = set()
        for ss in wordnet.synsets(ans):
            for lm in ss.lemmas():
                w = lm.name().replace("_"," ")
                if w.lower() != ans.lower():
                    syns.add(w)
            if len(syns)>=3: break
        syns = list(syns)
        
        # 2) context sentences as candidates
        sents = [s for s in text.split(". ") if ans.lower() not in s.lower() and len(s)>20]
        # embed answer + sentences
        ans_emb = similarity_model.encode(ans)
        sent_embs = similarity_model.encode(sents)
        sims = cosine_similarity([ans_emb], sent_embs)[0]
        # pick those with medium similarity (0.3–0.6)
        ctx_opts = [sents[i] for i,sim in enumerate(sims) if 0.3 < sim < 0.6]
        
        # assemble final 3 distractors
        distractors = []
        # take up to 2 synonyms
        distractors += syns[:2]
        # fill remaining from ctx_opts
        for sent in ctx_opts:
            if len(distractors) == 3:
                break
            distractors.append(sent)  # keep full sentence

        # fallback: random choice
        while len(distractors) < 3:
            rand = random.choice(sents) if sents else ans[::-1]
            distractors.append(rand)  # no truncation

        opts = [ans] + distractors
        random.shuffle(opts)
        questions.append({"question": q, "options": opts, "answer": ans})

    return questions

# Update quiz_tab function
def quiz_tab(quiz_components, similarity_model):
    """Render the quiz generation tab"""
    quiz_model, quiz_tokenizer, qa_pipeline = quiz_components

    st.subheader("📝 Quiz Generator")
    st.markdown("Upload documents to generate practice quizzes for students or create assessments.")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        uploaded_files = st.file_uploader(
            "Select documents for quiz generation",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            key="quiz_uploader"
        )
    
    with col2:
        num_questions = st.number_input(
            "Questions per document",
            min_value=1,
            max_value=10,
            value=3,
            help="Number of questions to generate per document"
        )
    
    if uploaded_files and st.button("Generate Quiz", type="primary"):
        all_questions = []
        with st.status("Generating quiz questions...", expanded=True) as status:
            for file in uploaded_files:
                st.write(f"📄 Processing {file.name}...")
                file_text, _ = extract_text(file)
                
                if not file_text:
                    st.error(f"Could not extract text from {file.name}")
                    continue
                
                # PASS the tuple (quiz_model,quiz_tokenizer)
                questions = generate_quiz(file_text, quiz_components, similarity_model, num_questions)
                if questions:
                    all_questions.extend(questions)
                    st.success(f"Generated {len(questions)} questions from {file.name}")
                else:
                    st.warning(f"No questions generated from {file.name}")
                
                time.sleep(0.5)  # Rate limiting
            
            status.update(label="Quiz generation complete!", state="complete")
        
        if all_questions:
            st.subheader("Generated Quiz Questions")
            
            # Display questions
            for i, q in enumerate(all_questions, 1):
                with st.expander(f"Question {i}: {q['question']}", expanded=False):
                    st.radio(
                        "Options",
                        options=q["options"],
                        key=f"question_{i}"
                    )
            
            # Download options
            st.markdown("---")
            st.subheader("Export Quiz")
            
            col1, col2 = st.columns(2)
            with col1:
                # JSON export
                quiz_json = json.dumps(all_questions, indent=2)
                st.download_button(
                    label="Download as JSON",
                    data=quiz_json,
                    file_name="quiz_questions.json",
                    mime="application/json"
                )
            
            with col2:
                # Text export
                quiz_text = "\n\n".join(
                    f"Q: {q['question']}\nOptions:\n" + 
                    "\n".join(f"- {opt}" for opt in q["options"]) + 
                    f"\nAnswer: {q['answer']}"
                    for q in all_questions
                )
                st.download_button(
                    label="Download as Text",
                    data=quiz_text,
                    file_name="quiz_questions.txt",
                    mime="text/plain"
                )

def load_cached_summary(file_hash):
    summary_path = os.path.join(CACHE_DIR, f"{file_hash}_summary.pkl")
    if os.path.exists(summary_path):
        with open(summary_path, "rb") as f:
            return pickle.load(f)
    return None

def save_cached_summary(file_hash, summary):
    summary_path = os.path.join(CACHE_DIR, f"{file_hash}_summary.pkl")
    with open(summary_path, "wb") as f:
        pickle.dump(summary, f)

def generate_summary_with_cache(text: str, file_hash: str, summarizer, tokenizer, 
                              max_length: int = 250, min_length: int = 100) -> str:
    """Generate or load cached summary"""
    cached_summary = load_cached_data(file_hash, "summary")
    if cached_summary:
        st.toast(f"Using cached summary for {file_hash[:8]}...", icon="💾")
        return cached_summary
    
    summary = generate_summary(text, summarizer, tokenizer, max_length, min_length)
    save_cached_data(file_hash, "summary", summary)
    return summary

def load_cached_embedding(file_hash):
    embedding_path = os.path.join(CACHE_DIR, f"{file_hash}_embedding.pkl")
    if os.path.exists(embedding_path):
        with open(embedding_path, "rb") as f:
            return pickle.load(f)
    return None

def save_cached_embedding(file_hash, embedding):
    embedding_path = os.path.join(CACHE_DIR, f"{file_hash}_embedding.pkl")
    with open(embedding_path, "wb") as f:
        pickle.dump(embedding, f)

def generate_embedding_with_cache(summary: str, file_hash: str, similarity_model) -> np.ndarray:
    """Generate or load cached embedding"""
    cached_embedding = load_cached_data(file_hash, "embedding")
    if cached_embedding is not None:
        st.toast(f"Using cached embedding for {file_hash[:8]}...", icon="💾")
        return cached_embedding
    
    embedding = similarity_model.encode(summary)
    save_cached_data(file_hash, "embedding", embedding)
    return embedding

# --- Enhanced Cache Management ---
def clear_cache():
    """Clear all cached files"""
    for filename in os.listdir(CACHE_DIR):
        file_path = os.path.join(CACHE_DIR, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            st.error(f"Error deleting {file_path}: {e}")

def process_file(file, summarizer, tokenizer, similarity_model, max_length, min_length):
    """Robust file processing with proper array handling"""
    try:
        # Extract file contents
        file_text, file_bytes = extract_text(file)
        if file_text is None or file_bytes is None:
            st.error(f"Could not extract content from {file.name}")
            return None, None, None
        
        # Generate file hash
        file_hash = get_file_hash(file_bytes)
        
        # Check cache and process
        cached_summary = load_cached_data(file_hash, "summary")
        cached_embedding = load_cached_data(file_hash, "embedding")
        
        if cached_summary is not None and cached_embedding is not None:
            st.toast(f"Using cached data for {file.name[:20]}...", icon="💾")
            return cached_summary, cached_embedding, file_hash
        
        # Generate summary if not cached
        if cached_summary is None:
            with st.spinner(f"Generating summary for {file.name[:20]}..."):
                summary = generate_summary(
                    file_text, 
                    summarizer, 
                    tokenizer,
                    max_length,
                    min_length
                )
                save_cached_data(file_hash, "summary", summary)
        else:
            summary = cached_summary
        
        # Generate embedding if not cached
        if cached_embedding is None:
            with st.spinner(f"Generating embedding for {file.name[:20]}..."):
                embedding = similarity_model.encode(summary)
                # Explicit conversion to numpy array if needed
                if not isinstance(embedding, np.ndarray):
                    embedding = np.array(embedding)
                save_cached_data(file_hash, "embedding", embedding)
        else:
            embedding = cached_embedding
        
        return summary, embedding, file_hash
        
    except Exception as e:
        st.error(f"Failed to process {file.name}: {str(e)}")
        return None, None, None

# --- Helper Functions ---
def extract_text(file) -> Tuple[Optional[str], Optional[bytes]]:
    """Extract text from file and return both text and original bytes"""
    try:
        file_bytes = file.getvalue() if hasattr(file, 'getvalue') else file.read()
        file.seek(0)  # Reset file pointer
        
        if file.name.endswith(".pdf"):
            reader = PdfReader(file)
            text = "\n".join([page.extract_text() or "" for page in reader.pages])
        elif file.name.endswith(".docx"):
            doc = Document(file)
            text = "\n".join([para.text for para in doc.paragraphs])
        elif file.name.endswith(".txt"):
            text = file_bytes.decode("utf-8")
        else:
            return None, None
            
        return text, file_bytes
    except Exception as e:
        st.error(f"Error processing {file.name}: {str(e)}")
        return None, None

def chunk_text(text: str, tokenizer, max_tokens: int = 1024) -> List[str]:
    """Split text into chunks that fit the model's token limit"""
    if not text.strip():
        return []
    
    sentences = [s.strip() + ". " for s in text.split(". ") if s.strip()]
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(tokenizer.encode(sentence))
        if current_length + sentence_length <= max_tokens:
            current_chunk.append(sentence)
            current_length += sentence_length
        else:
            if current_chunk:
                chunks.append("".join(current_chunk).strip())
            current_chunk = [sentence]
            current_length = sentence_length
    
    if current_chunk:
        chunks.append("".join(current_chunk).strip())
    
    return chunks


# --- Enhanced Summary Generation ---
def generate_summary(text: str, summarizer, tokenizer, 
                   max_length: int = 150, min_length: int = 40) -> str:
    """Enhanced summary generation with better chunk handling"""
    try:
        # Pre-process text
        text = " ".join(text.split())
        text = text.replace("..", ".").replace(". .", ".")
        
        # Dynamic length adjustment
        input_length = len(tokenizer.encode(text))
        if input_length < min_length:
            return text  # Return original if shorter than min_length
            
        # Improved chunking
        chunks = []
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        current_chunk = ""
        
        for para in paragraphs:
            para_tokens = len(tokenizer.encode(para))
            if para_tokens > 900:  # Conservative chunk size
                chunks.extend(chunk_text(para, tokenizer, max_tokens=900))
            elif len(tokenizer.encode(current_chunk + para)) <= 900:
                current_chunk += "\n\n" + para
            else:
                chunks.append(current_chunk.strip())
                current_chunk = para
                
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # Generate summary
        summary_parts = []
        for chunk in chunks:
            chunk_length = len(tokenizer.encode(chunk))
            chunk_max = min(max_length, max(int(chunk_length * 0.7), min_length))
            
            result = summarizer(
                chunk,
                max_length=chunk_max,
                min_length=max(10, min(min_length, chunk_max-5)),
                do_sample=False,
                truncation=True
            )
            summary_parts.append(result[0]['summary_text'])
        
        # Post-process
        final_summary = " ".join(summary_parts)
        return final_summary.replace(" .", ".").replace("  ", " ").strip()
        
    except Exception as e:
        st.error(f"Summary generation failed: {str(e)}")
        return ""


def create_similarity_visualization(similarity_matrix: np.ndarray, 
                                  filenames: List[str]) -> plt.Figure:
    """Create a heatmap visualization of the similarity matrix"""
    short_names = [name[:15] + "..." if len(name) > 15 else name for name in filenames]
    df = pd.DataFrame(similarity_matrix, columns=short_names, index=short_names)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        df,
        annot=True,
        cmap="YlOrRd",
        vmin=0,
        vmax=1,
        square=True,
        ax=ax,
        fmt=".2f",
        cbar_kws={"shrink": 0.75}
    )
    ax.set_title("Document Similarity Heatmap", pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig

# --- UI Components ---
def sidebar():
    """Render the sidebar content"""
    with st.sidebar:
        st.markdown("# 🎓 ScholarSphere")
        # st.markdown("### Academic Writing Assistant")
        st.markdown("AI-powered tools to enhance your workflow.")
        
        st.info("""
        **Features:**
        - Document summarization
        - Content similarity analysis
        - Visualize document relationships
        - Document to Quiz Generator
        - Batch processing
        """)
        
        st.markdown("---")
        st.markdown("### Settings")
        summarize_settings = st.expander("Summary Settings", expanded=False)
        with summarize_settings:
            max_length = st.slider("Maximum length", 100, 300, 150)  #Default value 150 tokens
            min_length = st.slider("Minimum length", 50, 100, 50)   #Default value 40 tokens
            if min_length > max_length:
                st.warning("⚠️ Minimum length must be ≤ Maximum length")
                min_length = min(min_length, max_length)  # Auto-correct
            st.markdown("Adjust the length settings to fit your needs.")
            # st.markdown("**Note:** Longer summaries may take more time to generate.")
        
        st.markdown("---")
        if st.button("🔄 Clear All Cached Data"):
            clear_cache()
            st.success("Cache cleared successfully!")
            st.rerun()
        st.markdown("""
        [Report Issue](https://github.com/SampurnGupta) | 
        [About](https://github.com/SampurnGupta)
        """)

        # Return the settings as a dictionary
        return {
            "max_length": max_length,
            "min_length": min_length
        }

def summary_tab(summarizer, tokenizer,max_length, min_length):
    """Render the summary generation tab"""
    st.subheader("Document Summarization")
    st.markdown("Upload research papers, articles, or reports to generate concise summaries.")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        uploaded_files = st.file_uploader(
            "Select documents",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            help="Supported formats: PDF, DOCX, TXT"
        )
    
    with col2:
        st.markdown("")
        st.markdown("")
        process_btn = st.button("Generate Summaries", type="primary")
    
    if uploaded_files and process_btn:
        with st.status("Processing documents...", expanded=True) as status:
            summaries = []
            full_texts = []
            
            for file in uploaded_files:
                st.write(f"📄 Processing {file.name}...")
                file_text, file_bytes = extract_text(file)
                
                if not file_text:
                    st.error(f"Could not extract text from {file.name}")
                    continue
                
                file_hash = get_file_hash(file_bytes)
                summary = generate_summary_with_cache(
                    file_text, 
                    file_hash, 
                    summarizer, 
                    tokenizer,
                    max_length,
                    min_length
                )
                
                if summary:
                    summaries.append((file.name, summary))
                    full_texts.append(file_text)
                    st.success(f"Summary generated for {file.name}")
                else:
                    st.warning(f"Empty summary for {file.name}")
                
                time.sleep(0.1)
            
            status.update(label="All documents processed!", state="complete")
        
        # Display individual summaries
        if summaries:
            st.subheader("Individual Summaries")
            for filename, summary in summaries:
                with st.expander(f"{filename}", expanded=False):
                    # st.markdown(f"**Summary:**\n\n{summary}")
                    st.markdown(f"\n\n{summary}")
            
            # Generate and display combined summary
            st.subheader("Combined Analysis")
            combined_text = "\n\n".join(full_texts)
            combined_summary = generate_summary(combined_text, summarizer, tokenizer)
            
            if combined_summary:
                with st.expander("📚 Integrated Summary of All Documents", expanded=True):
                    st.markdown(combined_summary)
                
                # Create downloadable package
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                    for filename, summary in summaries:
                        zip_file.writestr(
                            f"summaries/{filename}_summary.txt", 
                            summary
                        )
                    zip_file.writestr(
                        "combined_summary.txt", 
                        combined_summary
                    )
                
                st.download_button(
                    label="📥 Download All Summaries",
                    data=zip_buffer.getvalue(),
                    file_name="scholarsphere_summaries.zip",
                    mime="application/zip",
                    help="Includes all individual summaries and combined analysis"
                )

def similarity_tab(summarizer, tokenizer, similarity_model):
    """Render the similarity analysis tab"""
    st.subheader("Document Similarity Analysis")
    st.markdown("Compare multiple documents to identify content overlap and relationships.")
    
    uploaded_files = st.file_uploader(
        "Select documents to compare",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        key="similarity"
    )
    
    if uploaded_files and len(uploaded_files) > 1:
        if st.button("Analyze Similarity", type="primary"):
            with st.status("Analyzing documents...", expanded=True) as status:
                embeddings = []
                summaries = []
                filenames = []
                
                for file in uploaded_files:
                    st.write(f"🔍 Analyzing {file.name}...")
                    summary, embedding, _ = process_file(
                        file,
                        summarizer,
                        tokenizer,
                        similarity_model,
                        max_length=250,  # Default values for similarity analysis
                        min_length=100
                    )
                    
                    if summary is not None and embedding is not None:
                        embeddings.append(embedding)
                        summaries.append(summary)
                        filenames.append(file.name)
                        st.success(f"Processed {file.name}")
                    else:
                        st.warning(f"Failed to process {file.name}")
                    
                    time.sleep(0.1)
                
                status.update(label="Analysis complete!", state="complete")
            
            if len(embeddings) > 1:
                # Calculate similarity matrix
                similarity_matrix = cosine_similarity(embeddings)
                
                # Display results
                st.subheader("Similarity Results")
                
                tab1, tab2, tab3 = st.tabs(["Matrix", "Visualization", "Summary"])
                
                with tab1:
                    st.subheader("📊 Document-to-Document Similarity Matrix")
                    st.dataframe(
                        pd.DataFrame(
                            similarity_matrix,
                            columns=filenames,
                            index=filenames
                        ).style.format("{:.2f}").background_gradient(cmap="YlOrRd"),
                        use_container_width=True
                    )
                
                with tab2:
                    fig = create_similarity_visualization(similarity_matrix, filenames)
                    st.pyplot(fig)
                                
                with tab3:
                    st.markdown("### Document Summaries")
                    for filename, summary in zip(filenames, summaries):
                        with st.expander(filename, expanded=False):
                            st.markdown(summary)
                
                # Create downloadable package
                # Create download button
                def create_similarity_zip(similarity_matrix, filenames, summaries, fig):
                    zip_buffer = io.BytesIO()
                    
                    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                        # 1. Save similarity matrix as CSV
                        matrix_df = pd.DataFrame(
                            similarity_matrix,
                            columns=filenames,
                            index=filenames
                        )
                        zip_file.writestr(
                            "similarity_matrix.csv",
                            matrix_df.to_csv()
                        )
                        
                        # 2. Save visualization as PNG
                        img_buffer = io.BytesIO()
                        fig.savefig(img_buffer, format='png', bbox_inches='tight', dpi=300)
                        zip_file.writestr(
                            "similarity_heatmap.png",
                            img_buffer.getvalue()
                        )
                        
                        # 3. Save individual summaries
                        for filename, summary in zip(filenames, summaries):
                            zip_file.writestr(
                                f"summaries/{filename}_summary.txt",
                                summary
                            )
                        
                        # 4. Add a README file
                        readme = """SIMILARITY ANALYSIS REPORT
                        
            Contains:
            1. similarity_matrix.csv - Pairwise similarity scores
            2. similarity_heatmap.png - Visual representation
            3. summaries/ - Individual document summaries
                        """
                        zip_file.writestr("README.txt", readme)
                    
                    zip_buffer.seek(0)
                    return zip_buffer

                # Create and display download button
                zip_buffer = create_similarity_zip(similarity_matrix, filenames, summaries, fig)
                st.download_button(
                    label="📥 Download Full Analysis",
                    data=zip_buffer,
                    file_name="similarity_analysis_report.zip",
                    mime="application/zip",
                    help="Includes: similarity matrix, heatmap visualization, and all summaries"
                )


            else:
                st.warning("Need at least 2 valid documents for similarity analysis")

# --- Research Explorer Tab ---
from keybert import KeyBERT

def research_tab():
    st.subheader("🔎 Research Explorer")
    st.markdown("Upload a document to extract key research topics and fetch related papers.")
    file = st.file_uploader(
        "Select a document for topic extraction",
        type=["pdf","docx","txt"],
        key="research_uploader"
    )
    if not file:
        return

    text, _ = extract_text(file)
    if not text:
        st.error("Could not extract text.")
        return

    # 1) Extract more descriptive topics
    with st.spinner("Extracting key topics…"):
        kw_model = KeyBERT(model="all-MiniLM-L6-v2")
        kws = kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(2,3),    # allow 2–3 word phrases
            use_mmr=True,                   # max-marginal-relevance
            diversity=0.7,                  # more varied
            stop_words="english",
            top_n=8                         # get 8 candidates
        )
        topics = [phrase for phrase,score in kws]

    if not topics:
        st.warning("No topics found.")
        return

    st.markdown("**Suggested Research Topics:**")
    for t in topics:
        st.markdown(f"- {t}")

    # 2) Fetch papers from Semantic Scholar
    st.markdown("**Recommended Papers:**")
    for topic in topics:
        params = {
            "query": topic,
            "limit": 3,
            "fields": "title,authors,year,url"
        }
        resp = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params=params
        )
        if resp.ok:
            papers = resp.json().get("data", [])
            for p in papers:
                title = p.get("title", "No title")
                url   = p.get("url", "")
                year  = p.get("year", "?")
                auths = ", ".join(a["name"] for a in p.get("authors", [])[:3])
                st.markdown(f"- [{title}]({url}) ({year}) • {auths}")
        # else: silently ignore failures (no warning)
# --- Main App ---
def main():
    """Main application function"""
    settings = sidebar()
    
    st.title("ScholarSphere")
    st.caption("AI-powered Academic Assistant")
    
    # Move GPU toggle outside cached function
    use_gpu = st.sidebar.checkbox("Use GPU if available", value=False)
    
    # Pass GPU selection as parameter
    summarizer, tokenizer, similarity_model, quiz_components = load_models(use_gpu)
    
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Summarization", "Similarity Analysis", "Quizzer", "Research Explorer"]
    )
    
    with tab1:
        # Pass the settings to summary_tab
        summary_tab(summarizer, tokenizer, settings["max_length"], settings["min_length"])
    
    with tab2:
        similarity_tab(summarizer, tokenizer, similarity_model)

    with tab3:
        quiz_model, quiz_tokenizer, qa_pipeline = quiz_components
        if quiz_model and quiz_tokenizer:
            quiz_tab(quiz_components, similarity_model)
        else:
            st.info("Quiz generation is disabled (missing SentencePiece).")
    
    with tab4:
        research_tab()

if __name__ == "__main__":
    main()