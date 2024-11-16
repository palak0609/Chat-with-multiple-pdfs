import streamlit as st
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
import openai
import pickle

# Load API keys
load_dotenv()
#openai_api_key = os.getenv("OPENAI_API_KEY")
openai_api_key = st.secrets["OPENAI_API_KEY"]

# Helper function to load/save cached data
def save_cache(data, file_name):
    with open(file_name, "wb") as f:
        pickle.dump(data, f)

def load_cache(file_name):
    if os.path.exists(file_name):
        with open(file_name, "rb") as f:
            return pickle.load(f)
    return None

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        with pdfplumber.open(pdf) as pdf_reader:
            for page in pdf_reader.pages:
                text += page.extract_text() or ""  # Ensure no NoneType errors if extract_text fails
    return text

# Function to split text into larger chunks
def get_text_chunks(text):
    chunk_size = 10000  # Larger chunks
    chunk_overlap = 1000  # Small overlap
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks

def process_text_chunks(text_chunks, embedding_model_name="all-MiniLM-L6-v2"):
    cached_embeddings = load_cache("cached_embeddings.pkl")
    if cached_embeddings:
        st.info("Using cached embeddings.")
        return cached_embeddings

    model = SentenceTransformer(embedding_model_name)
    embeddings = [model.encode(chunk) for chunk in text_chunks]

    # Save embeddings for reuse
    save_cache(embeddings, "cached_embeddings.pkl")
    return embeddings

def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in the provided context,
    say, "answer is not available in the context". Don't provide a wrong answer.

    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """
    model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def user_input(user_question):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vector_store = FAISS.load_local("faiss_index", embeddings)
    docs = vector_store.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply:", response["output_text"])

def main():
    st.set_page_config(page_title="Chat PDF", layout="wide")
    st.header("Chat with PDF using OpenAI💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        if os.path.exists("faiss_index"):
            user_input(user_question)
        else:
            st.warning("Please upload and process PDFs first.")

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True, type=["pdf"])
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    if raw_text:
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                        st.success("Done")
                    else:
                        st.error("No text extracted from the uploaded PDFs.")
            else:
                st.warning("Please upload at least one PDF file.")

if __name__ == "__main__":
    main()
