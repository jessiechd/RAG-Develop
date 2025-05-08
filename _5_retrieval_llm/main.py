import os
import json
import torch
import uuid
import numpy as np
from supabase import create_client, Client
from transformers import AutoTokenizer, AutoModel
import ast
import re
import vecs
from dotenv import load_dotenv
import openai
from scipy.spatial.distance import cosine
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import psycopg2
from pgvector.psycopg2 import register_vector
from pathlib import Path


nltk.download('all')
nltk.download('punkt')
nltk.download('stopwords')

env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=env_path)
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
DB_CONNECTION = os.getenv("DB_CONNECTION")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET")
openai.api_key = OPENAI_API_KEY

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

vx = vecs.create_client(DB_CONNECTION)
vec_text = vx.get_or_create_collection(name="vec_text", dimension=768)
vec_table = vx.get_or_create_collection(name="vec_table", dimension=768)

tokenizer = AutoTokenizer.from_pretrained("Alibaba-NLP/gte-multilingual-base", trust_remote_code=True)
model = AutoModel.from_pretrained("Alibaba-NLP/gte-multilingual-base", trust_remote_code=True).to(
    torch.device("cuda" if torch.cuda.is_available() else "cpu"))

def get_embedding(text):
    """Generates an embedding vector from input text."""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().tolist()

def query_supabase(user_query, user_id, session_id):
    """Retrieves both text and table chunks based on query, ensuring relevance balance and filtering by user_id and session_id."""
    query_embedding = get_embedding(user_query)
    embedding_str = ','.join([str(x) for x in query_embedding])

    conn = psycopg2.connect(DB_CONNECTION)
    register_vector(conn)
    cur = conn.cursor()

    cur.execute(f"""
        SELECT id, 1 - (vec <=> ARRAY[{embedding_str}]::vector) AS similarity
        FROM vecs.vec_text
        WHERE metadata->>'user_id' = %s AND metadata->>'session_id' = %s
        ORDER BY vec <=> ARRAY[{embedding_str}]::vector
        LIMIT 5
    """, (user_id, session_id))

    text_chunk_ids = cur.fetchall()
    text_results = []

    if text_chunk_ids:
        chunk_id_list = tuple([str(row[0]) for row in text_chunk_ids])
        cur.execute(f"""
            SELECT chunk_id, content, metadata
            FROM public.documents_chunk
            WHERE chunk_id IN %s AND metadata->>'user_id' = %s AND metadata->>'session_id' = %s;
        """, (chunk_id_list, user_id, session_id))
        text_chunks = {row[0]: row[1:] for row in cur.fetchall()}
        text_results = [(cid, "text", text_chunks[cid][0], sim)
                        for cid, sim in text_chunk_ids if cid in text_chunks]
        text_results.sort(key=lambda x: x[3], reverse=True)

    cur.execute(f"""
        SELECT id, 1 - (vec <=> ARRAY[{embedding_str}]::vector) AS similarity
        FROM vecs.vec_table
        WHERE metadata->>'user_id' = %s AND metadata->>'session_id' = %s
        ORDER BY vec <=> ARRAY[{embedding_str}]::vector
        LIMIT 5
    """, (user_id, session_id))

    table_chunk_ids = cur.fetchall()
    table_results = []

    if table_chunk_ids:
        chunk_id_list = tuple([str(row[0]) for row in table_chunk_ids])
        cur.execute(f"""
            SELECT chunk_id, description, metadata
            FROM public.tables_chunk
            WHERE chunk_id IN %s AND metadata->>'user_id' = %s AND metadata->>'session_id' = %s;
        """, (chunk_id_list, user_id, session_id))
        table_chunks = {row[0]: row[1:] for row in cur.fetchall()}
        table_results = [(cid, "table", table_chunks[cid][0], sim)
                         for cid, sim in table_chunk_ids if cid in table_chunks]
        table_results.sort(key=lambda x: x[3], reverse=True)

    conn.close()

    combined_results = text_results[:3] + table_results[:3]
    combined_results.sort(key=lambda x: x[3], reverse=True)
    return combined_results[:5]

def call_openai_llm(user_query, retrieved_chunks, chat_history=[]):
    """Send the query along with retrieved context and chat history to OpenAI API."""
    context_text = "\n\n".join([f"Chunk {i+1}: {chunk[2]}" for i, chunk in enumerate(retrieved_chunks)])
    print("\n[DEBUG] Context sent to LLM:")
    print(context_text[:500]) 
    messages = [
        {"role": "system", "content": "You are an intelligent assistant. Use the following retrieved information to answer the user's query."},
    ]
    messages.extend(chat_history)
    messages.append({"role": "user", "content": f"Context:\n{context_text}\n\nUser's Question: {user_query}"})
    client = openai.OpenAI(api_key=openai.api_key)
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=messages,
        temperature=0.7
    )
    answer = response.choices[0].message.content  
    chat_history.append({"role": "user", "content": user_query})
    chat_history.append({"role": "assistant", "content": answer})
    return answer, chat_history

def chat():
    """Handles continuous chat interaction with support for new chat and conversational context."""
    chat_history = [] 
    print("Welcome to the assistant! Type 'exit' to end the chat, 'new chat' to start over.")

    while True:
        user_query = input("User: ")

        if user_query.lower() in ["exit", "quit"]:
            print("Chat ended.")
            break
        
        if user_query.lower() == "new chat":
            chat_history = []  
            print("Starting a new chat...\n")
            continue  


        retrieved_chunks = query_supabase(user_query)

        answer, chat_history = call_openai_llm(user_query, retrieved_chunks, chat_history)
        
        print(f"Assistant: {answer}\n")



if __name__ == "__main__":
    chat()




