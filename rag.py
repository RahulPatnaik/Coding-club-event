import streamlit as st
from langchain_groq import ChatGroq
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import Document
from dotenv import load_dotenv
import os
import PyPDF2

# Load environment variables from .env file
load_dotenv()

# Use the Groq API key from the environment variable
api_key = os.getenv("GROQ_API_KEY")

# Initialize the language model with Groq
llm = ChatGroq(
    model="mixtral-8x7b-32768",
    api_key=api_key,
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# Load the embeddings model for FAISS retrieval
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Function to extract text from uploaded PDF files
def extract_text_from_pdf(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Streamlit interface
st.title("RAG Chatbot with Groq and Document Upload")
st.write("Upload a document to enable question answering based on the contents of the document.")

# File uploader for documents (PDFs, TXT files)
uploaded_file = st.file_uploader("Choose a document", type=["pdf", "txt"])

if uploaded_file is not None:
    # Extract text from the uploaded file
    if uploaded_file.type == "application/pdf":
        document_text = extract_text_from_pdf(uploaded_file)
    elif uploaded_file.type == "text/plain":
        document_text = uploaded_file.read().decode("utf-8")
    
    # Create the document with 'page_content' and 'metadata'
    document = Document(page_content=document_text)
    
    # Initialize FAISS store with the document
    faiss_store = FAISS.from_documents([document], embeddings)

    # Input for user's query
    user_question = st.text_input("Enter your question:")

    if user_question:
        # Step 1: Retrieve relevant document text
        retrieved_docs = faiss_store.similarity_search(user_question, k=3)
        context = " ".join([doc.page_content for doc in retrieved_docs])

        # Step 2: Prepare the prompt with the context as part of the system message
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"You are a helpful assistant. Here is some context to help answer the question:\n\n{context}"
                ),
                ("human", user_question),
            ]
        )

        # Step 3: Invoke the LLM with the prompt
        chain = prompt | llm
        aimsg = chain.invoke({"input": user_question})

        # Display the response
        st.write("**AI Response:**", aimsg.content)
