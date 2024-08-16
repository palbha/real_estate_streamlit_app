import streamlit as st
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import os


os.environ["OPENAI_API_KEY"] = "***"
os.environ["OPENAI_API_BASE"] = "https://openai.vocareum.com/v1"

# Initialize OpenAI LLM and embeddings
llm = OpenAI(openai_api_key=os.environ["OPENAI_API_KEY"],model_name="gpt-3.5-turbo-0125", temperature=0.5, max_tokens=2500)
embedding_model = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"], model="text-embedding-3-large")

# Function to create listings
def create_listings(city, country):
    prompt = f"Create a real estate listing for {city}, {country}."
    listing = llm(prompt)
    return listing

# Function to save listings to Chroma DB
def save_to_chroma(listings):
    docs = [{"content": listing} for listing in listings]
    db = Chroma.from_documents(docs, embedding_model, persist_directory='./chroma_db')
    return db

# Streamlit UI
st.title("Real Estate Listing Creator and Query Tool")

# Section to create listings
st.header("Create Listings")
city = st.text_input("City")
country = st.text_input("Country")
if st.button("Create Listings"):
    listing = create_listings(city, country)
    st.write("Generated Listing:")
    st.write(listing)
    
    if st.button("Save to Chroma DB"):
        db = save_to_chroma([listing])
        st.success("Listing saved to Chroma DB!")

# Section to query Chroma DB
st.header("Query Listings")
query = st.text_input("Enter your query")
if st.button("Search Listings"):
    db = Chroma(persist_directory='./chroma_db', embedding_function=embedding_model)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # You can use "map_reduce" for more complex queries
        retriever=db.as_retriever()
    )
    response = qa_chain.run(query)
    st.write("Query Result:")
    st.write(response)
