from langchain_huggingface import HuggingFaceEmbeddings
import os

os.environ["HF_HOME"] = 'E:/huggingface_cache'

embedding= HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

documents = [
    "Delhi is capital Of India",
    "Kolkata is capital of West Bengal",
    "Paris is capital of France"
]

vector = embedding.embed_documents(documents)

print(str(vector))