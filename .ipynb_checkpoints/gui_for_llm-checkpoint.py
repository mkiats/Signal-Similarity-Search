import os
import torch
import streamlit as st
import faiss as FF
import numpy as np
import torch.nn as nn
import matplotlib as plt
from scipy.io import wavfile as wavfile
from scipy.signal import stft as stft
from faiss import write_index, read_index
from langchain.embeddings import LlamaCppEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from helper import *

cosineindex_llm = read_index("./new_faiss_index/faiss_questionset_65B")
cosineindex_signal = read_index("./new_faiss_index/faiss_signalSet")
llama = LlamaCppEmbeddings(model_path=r"models/ggml-model-f16-q4_0.bin", n_ctx= 2048) # Input any preferred LLM model
loader = TextLoader('./faiss_documents/questionSet.txt')
documents = loader.load()
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size = 50, 
    chunk_overlap = 0,
    length_function = len)
docs = text_splitter.split_documents(documents)

# Start of helper functions
def summariseSignal(upload):
    st.write("TODO: Load Summary")

def mostSimilar(upload):
    st.write("TODO: Find most similar")

def leastSimilar(upload):
    st.write("TODO: Find least similar")

def process_query(cur_query, llm, cosineindex_llm, cosine_search_threshold=0.8, k_neighbours=3):
    
    cur_query_copy = cur_query
    cur_query = llm.embed_query(cur_query)
    cur_query = np.array([cur_query])
    cur_query /= np.linalg.norm(cur_query, axis=1, keepdims=True)
    dist, idx = cosineindex_llm.search(cur_query, k_neighbours)
    dist = dist.squeeze()
    idx = idx.squeeze()
    
    query_list=[] # Contains id of questions that are similar
    
    # Eliminate questions with cosine similarity lower than threshold
    for i in range(k_neighbours):
        if (dist[i] > cosine_search_threshold):
            query_list.append(idx[i])
    
    if len(query_list)==0:
        process_empty_list(cur_query_copy)
        
    return query_list
        
def process_empty_list(cur_query):
    print(f"Query cannot be found/executed...\n",
          "Do you wish to save this query for future development purposes?\n")
    add_to_development = int(input("1 for yes, 0 for no..."))
    if add_to_development==1:
        with open('failed_queries.txt', 'w') as f:
            f.write(cur_query, "\n")
            f.close()
        print("Query saved...\n")

# End of helper functions

# st.title("LLM + SimCLR")

# request = st.text_input("What do you wish to find out?", " ")
request = input("What do you wish to find out?\n")
# st.write(cosineindex_llm.ntotal)
print(cosineindex_llm.ntotal)
upload = st.file_uploader("Choose a file...")

if request == str(0):
    # st.write(request)
    # st.write("Returning wav file information")
    sr, df = wavfile.read(upload)
    print(sr, df)
    # st.write(sr)
    # st.write(df.shape)
else: 

    query_list = process_query(request, llm=llama, cosine_search_threshold=0.9, k_neighbours=3, cosineindex_llm=cosineindex_llm)
    # selected_question = st.multiselect("Choose the queries you are looking for...", [docs[i].page_content for i in query_list])
    print([docs[i].page_content for i in query_list])
    # st.write(docs[idx.squeeze()[0]].page_content)
    print(docs[idx.squeeze()[0]].page_content)
    summariseSignal(upload)


