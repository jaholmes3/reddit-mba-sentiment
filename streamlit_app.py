#scrollTo=0e669793
# app.py
import streamlit as st
import google.generativeai as genai
import os
from pinecone import Pinecone, ServerlessSpec
import pandas as pd
import requests
import time
from textblob import TextBlob
from datetime import datetime
# Removed google.colab.userdata as it's not available in Streamlit

st.title("Reddit MBA Sentiment Analysis")
st.write("This application analyzes sentiment of Reddit posts from the MBA subreddit and provides answers to your questions using a LLM/RAG system based on the post content.")

# Securely handle API keys using Streamlit Secrets
# Create a .streamlit/secrets.toml file with your API keys:
# GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"
# PINECONE_API_KEY="YOUR_PINECONE_API_KEY"
# Alternatively, you can use environment variables in your deployment environment.
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
except KeyError:
    st.error("API keys not found. Please set GOOGLE_API_KEY and PINECONE_API_KEY in your Streamlit secrets.")
    st.stop()


genai.configure(api_key=GOOGLE_API_KEY)

# Initialize Pinecone connection
try:
    # For serverless, you only need the API key and region
    # For older free tier or paid plans, you might need environment and project ID
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index_name = "reddit-mba-sentiment" # Make sure this matches the name used in the notebook
    # Check if index exists before connecting
    existing_indexes = pc.list_indexes()
    if index_name not in [index_info.name for index_info in existing_indexes]:
         st.error(f"Error: Pinecone index '{index_name}' does not exist. Please create it first.")
         st.stop() # Stop the app if the index is not found
    else:
        index = pc.Index(index_name)
        st.success(f"Connected to Pinecone index '{index_name}'.")

except Exception as e:
    st.error(f"Error initializing Pinecone: {e}")
    st.stop()


# Initialize the Gemini generative model
generative_model = genai.GenerativeModel('gemini-1.5-flash-latest') # Changed model name
#generative_model = genai.GenerativeModel('gemini-1.5-pro-latest') # Changed model name

# Define a function to truncate text
def truncate_text(text, max_length=30000): # Adjust max_length as needed, staying below 36000
    if text is None:
        return ""
    return text[:max_length]

# Function to generate embedding for a query
def get_embedding(text):
    # Truncate text if necessary, similar to how we did for indexing
    truncated_text = truncate_text(text) # Use the truncate_text function defined earlier
    response = genai.embed_content(model='embedding-001', content=truncated_text)
    return response['embedding']

# Function to query the Pinecone index
def query_index(query_embedding, top_k=5):
    # Query the index with the query embedding
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True # Include metadata to get the original text and other details
    )
    return results

# Function to generate a response using RAG
def generate_rag_response(query_text, top_k=5):
    # Get the embedding for the query
    query_embedding = get_embedding(query_text)

    # Query the Pinecone index to retrieve relevant documents
    search_results = query_index(query_embedding, top_k=top_k)

    # Augment the prompt with the retrieved document content
    retrieved_content = ""
    for i, result in enumerate(search_results['matches']):
        retrieved_content += f"--- Document {i+1} (Score: {result['score']:.4f}) ---\n"
        # Include subreddit in the retrieved content
        if 'subreddit' in result['metadata']:
            retrieved_content += f"Subreddit: r/{result['metadata']['subreddit']}\n"
        retrieved_content += f"Title: {result['metadata']['title']}\n"
        retrieved_content += f"Selftext: {result['metadata']['selftext']}\n"
        # Include the permalink in the retrieved content
        if 'permalink' in result['metadata']:
            retrieved_content += f"Source: https://www.reddit.com{result['metadata']['permalink']}\n"
        retrieved_content += "\n" # Add an extra newline for separation

    # Create the augmented prompt
    # Instruct the model to use the provided documents and include sources and subreddits if relevant
    augmented_prompt = f"""Based on the following Reddit posts, please answer the question.
Please reference the provided documents and include the 'Source' URL and subreddit name for any information you use from a document.

{retrieved_content}

Question: {query_text}

Answer:
"""

    # Generate the response using the generative model
    try:
        response = generative_model.generate_content(augmented_prompt)
        return response.text
    except Exception as e:
        return f"Error generating response: {e}"


# --- Streamlit UI ---

# Text input for user query
user_query = st.text_input("Enter your query about MBA topics:")

# Button to submit the query
if st.button("Get Response"):
    # Check if the user input is not empty
    if user_query:
        # Call the RAG response generation function
        with st.spinner("Generating response..."): # Show a spinner while processing
            rag_response = generate_rag_response(user_query)
        # Display the response
        st.subheader("Response:")
        st.write(rag_response)
    else:
        st.warning("Please enter a query.")
