from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import pandas as pd
import os

load_dotenv()
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

index_name = "restaurant-reviews"

pc = Pinecone(api_key=PINECONE_API_KEY)
if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

def get_documents(upload_file) -> list[Document]:
   df = pd.read_csv(upload_file)      
   documents = []
   for i, row in df.iterrows():
        doc = Document(
        page_content=row["Title"] + " " + row["Review"],
        metadata={"rating": row["Rating"], "date": row["Date"]},
            id=str(i)
        )
        documents.append(doc)
   return documents

def create_vector_store(documents: list[Document]):
    vector_store = PineconeVectorStore.from_documents(
        index_name= index_name,
        documents= documents,
        embedding= embeddings,        
    )
    return vector_store

def get_retriever(vector_store: PineconeVectorStore):
   retriever = vector_store.as_retriever(
      search_type="similarity",
      search_kwargs={"k": 3}
   )
   return retriever