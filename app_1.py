import streamlit as st
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import os
import docx2txt
import PyPDF2

# Initialize models
# summarizer = pipeline("summarization", model="facebook/distilbart-cnn-12-6", device=-1)  # Use CPU for model inference
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

# Streamlit app UI setup
st.title("File Summarization and Similarity Checker")
st.write("Upload your `.txt`, `.pdf`, or `.docx` files to generate summaries and compare similarity.")
uploaded_files = st.file_uploader("Upload text files (PDF, DOCX, TXT)", accept_multiple_files=True)

st.write("**Note:** The summarization model may take some time to process larger files.")

def extract_text(file):
    file_type = file.name.split('.')[-1].lower()
    text = ""

    if file_type == 'txt':
        file.seek(0)
        text = file.read().decode('utf-8', errors='ignore')

    elif file_type == 'pdf':
        file.seek(0)
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()

    elif file_type == 'docx':
        file.seek(0)
        temp_path = os.path.join("temp", file.name)
        os.makedirs("temp", exist_ok=True)
        with open(temp_path, "wb") as f:
            f.write(file.read())
        text = docx2txt.process(temp_path)
        os.remove(temp_path)

    else:
        st.warning(f"{file.name} has an unsupported file format.")
        return ""

    return text


# Function to check and adjust the text length to be within the model's limit
def truncate_text_to_tokens(text, tokenizer, max_tokens=1024):
    inputs = tokenizer(text, return_tensors='pt', truncation=False)
    token_count = len(inputs['input_ids'][0])
    
    if token_count > max_tokens:
        text = tokenizer.decode(inputs['input_ids'][0][:max_tokens], skip_special_tokens=True)
    return text


summaries = []
embeddings = []

# Summarization and similarity calculation
if uploaded_files:
    st.success(f"{len(uploaded_files)} file(s) uploaded successfully!")
    for file in uploaded_files:
        st.write(f"• {file.name}")

    for file in uploaded_files:
        # Read file content
        file_text = extract_text(file)
        if not file_text:
            st.error(f"Could not extract text from {file.name}.")
            continue

        # Tokenize the text to check its length and truncate if necessary
        file_text = truncate_text_to_tokens(file_text, summarizer.tokenizer)
        
        # Dynamically adjust max_length based on input length
        input_length = len(file_text.split())  # in words
        max_length = min(200, input_length // 2)  # Ensure max_length is no larger than half of input length

        # Summarize the file content
        summary = summarizer(file_text, max_length=max_length, min_length=50, do_sample=False)[0]['summary_text']

        # # Summarize the file content
        # trimmed_text = file_text[:1000]  # Adjust as needed
        # # Dynamically adjust max_length based on input length
        # input_length = len(file_text.split())
        # max_length = min(200, input_length // 2)  # Ensure max_length is no larger than half of input length
        # summary = summarizer(file_text, max_length=max_length, min_length=50, do_sample=False)[0]['summary_text']

        summaries.append(summary)
        with st.expander(f"Summary of {file.name}"):
            st.write(summary)
        
        # Calculate embeddings for similarity
        embeddings.append(model.encode(summary))

    # Calculate similarity between files
    if len(summaries) > 1:
        st.write("**Similarity Report:**")
        for i in range(len(summaries)):
            for j in range(i + 1, len(summaries)):
                similarity = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
                st.write(f"Similarity between {uploaded_files[i].name} and {uploaded_files[j].name}: {similarity:.2f}")