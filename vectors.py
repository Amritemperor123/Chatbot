import os
import numpy as np
import uuid
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from supabase import create_client
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Read text data from file
with open("data.txt", "r", encoding="utf-8") as file:
    text_data = file.readlines()

# Split text into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
documents = text_splitter.create_documents(text_data)

# Initialize Hugging Face Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Generate embeddings
doc_embeddings = model.encode([doc.page_content for doc in documents], convert_to_list=True)

# Connect to Supabase
# SUPABASE_URL = os.getenv("SUPABASE_URL_LC_CHATBOT")
# SUPABASE_KEY = os.getenv("SUPABASE_API_KEY")
supabase = create_client('https://mpwzvopiompnknvcyuim.supabase.co', 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im1wd3p2b3Bpb21wbmtudmN5dWltIiwicm9sZSI6ImFub24iLCJpYXQiOjE3Mzg4MjgxOTgsImV4cCI6MjA1NDQwNDE5OH0.dCmdUFgzssB1ZrR9gKckVun57Ss81Q-_blvcmbFDi0Q')

# Store embeddings in Supabase
for i, doc in enumerate(documents):
    supabase.table("documents").insert({
        "id": str(uuid.uuid4()),
        "content": doc.page_content,
        "embedding": doc_embeddings[i].tolist()  # Store as an array
    }).execute()

print("Embeddings successfully stored in Supabase!")