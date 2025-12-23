import streamlit as st
from model import get_restaurant_review
#from vector_chroma import get_documents, create_vector_store, get_retriever
from vector_pinecone import *

st.set_page_config("Restaurant Review Agent", layout='centered')
st.title("Restaurant Review Agent")
upload_file = st.file_uploader("Upload your restaurant reviews CSV file", type=["csv"])
if upload_file is not None:
    st.success("File uploaded successfully!")    
    question = st.text_input("Ask a question about the restaurant :")
    if question and upload_file:     
        if st.button("Process"):
            with st.spinner("Getting your answer..."):
                documents = get_documents(upload_file)
                vector_store = create_vector_store(documents)
                retriever = get_retriever(vector_store)
                answer = get_restaurant_review(question, retriever)
                st.subheader("Answer:")       
                st.write(answer)