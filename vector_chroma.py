from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama.embeddings import OllamaEmbeddings
import pandas as pd
import os

embeddings = OllamaEmbeddings(model="mxbai-embed-large")
db_location = "./chrome_langchain_db"
add_documents = not os.path.exists(db_location)

def get_documents(upload_file) -> list[Document]:   
   if add_documents:
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
   if add_documents:
    vector_store = Chroma.from_documents(
        documents= documents,
        embedding= embeddings,
        persist_directory= db_location,
        collection_name="restaurant_reviews"
    )
    return vector_store

def get_retriever(vector_store: Chroma):
   retriever = vector_store.as_retriever(
      search_type="similarity",
      search_kwargs={"k": 3}
   )
   return retriever
   
   