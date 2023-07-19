import os
import torch
import shutil
import streamlit as st
import faiss as FF
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.io import wavfile as wavfile
from scipy.signal import stft as stft
from faiss import write_index, read_index
from langchain.embeddings import LlamaCppEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from helper_class import *

# Parameters to set
cosineIndex_llmPath = os.path.join("_faiss_index", "faiss_questionset_65B")
cosineIndex_signalPath = os.path.join("_faiss_index", "faiss_signalSet")
# llm_modelPath = os.path.join("_model", "alpaca-lora-65B.ggmlv3.q4_0.bin")
llm_modelPath = os.path.join("_model", "llama-2-13b.ggmlv3.q4_0.bin")
simclr_model_path = os.path.join("_model", "lr1e-05b32_30.pth")
signal_dataset_path = os.path.join("_trial_data", "train")

loader = TextLoader('./_faiss_documents/questionSet.txt')
documents = loader.load()
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size = 50, 
    chunk_overlap = 0,
    length_function = len)
docs = text_splitter.split_documents(documents)


# DO NOT TOUCH
cosineIndex_llm = read_index(cosineIndex_llmPath)
cosineIndex_signal = read_index(cosineIndex_signalPath)
llm_model = LlamaCppEmbeddings(model_path=llm_modelPath, n_ctx= 2048)
simclr_model = ResNet(128)
simclr_model.load_state_dict(torch.load(simclr_model_path, map_location="cpu"))
simclr_model.eval()






# Helper function for the signals __________________________________________________________________________________

def find_k_most_similar(uploaded_signal_path, k_neighbours=3):

    # get spectrogram from wavFile
    sr, wav = wavfile.read(uploaded_signal_path)
    f, t, Zxx = normalise(wav, sr)
    Zxx = tensorify(Zxx)

    # Get vector embedding of wavfile
    vector_embed = simclr_model(Zxx)

    # Normalise vector embedding
    vector_embed = vector_embed.detach().numpy()
    vector_embed /= np.linalg.norm(vector_embed, axis=1, keepdims=True)

    # Search vector database
    similarity, idx = cosineIndex_signal.search(vector_embed, k_neighbours)
    similarity = similarity.squeeze()
    idx = idx.squeeze()
    
    res=list(zip(idx, similarity)) # res = [(x1, y1), (x2, y2)] where x is the index, y is the cosine similarity
    return res

def predict_signal(uploaded_signal_path, k_neighbours=3):
    print("TODO")
    return

def summarise_signal(uploaded_signal_path, k_neighbours=3):
    sr, wav = wavfile.read(uploaded_signal_path)
    f, t, Zxx = normalise(wav, sr)
    return (sr, t, f, Zxx)
    
def plot_spec_signal(t, f, Zxx, output_dir, title="Spectrogram", cmap="magma"):
    plt.pcolormesh(t, f, np.abs(Zxx), cmap=cmap)
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.colorbar()
    plt.show()
    plt.savefig(output_dir)
    plt.close()
    return

# Helper functions for questions on dataset
def summarise_dataset(uploaded_signal_path, k_neighbours=3):
    # TODO: show distribution of train and validation set
    return 

def show_similar_signal_timestamp(uploaded_signal_path, k_neighbours=3, signal_similarity_threshold=0.9):
    results = find_k_most_similar(uploaded_signal_path, k_neighbours)
    signal_dataset = os.listdir(signal_dataset_path)
    
    for idx, sim in results:
        if sim > signal_similarity_threshold:
            file_name = os.path.basename(signal_dataset[idx])
            timestamp = int(file_name.split("_")[1])
            print(timestamp, sim)
    return

# Helper functions for questions on model
def show_model(uploaded_signal_path, k_neighbours=3):
    print("SimCLR model used")
    print("Model architecture is as such")
    print(simclr_model)
    return

def show_hyperpara_model(uploaded_signal_path, k_neighbours=3):
    NUM_EPOCHS = 15
    LR = 0.00005
    BATCH_SIZE = 32
    embedding_size = 128
    
    print(f"Learning rate is {LR}\n",
          f"Num of epochs is {NUM_EPOCHS}\n",
          f"Batch size is {BATCH_SIZE}\n",
          f"Final embedding size is {embedding_size}\n"
          )
    return

def show_loss_curve_model(uploaded_signal_path, k_neighbours=3):
    # TO BE REMOVED
    return (x_arr, y_arr)


def invalid_fn(uploaded_signal_path, k_neighbours=3):
    return -1



    

    
# Helper functions for LLM querying _______________________________________________________________________________________
query_dict = {
    0: 0,
    1: 0,
    2: 0,
    3: 1,
    4: 1,
    5: 2,
    6: 2,
    7: 3,
    8: 4,
    9: 4,
    10: 5,
    11: 6,
    "": -1
    }

fn_dict = {
    -1: invalid_fn,
    0: find_k_most_similar,
    1: summarise_signal,
    2: summarise_dataset,
    3: show_similar_signal_timestamp, #Previously show accuracy score
    4: show_model,
    5: show_hyperpara_model,
    6: show_loss_curve_model
}

# Generate similar queries that exceeds the threshold
def process_empty_list(cur_query):
    if not os.path.exists("failed_queries.txt"):
        file = open("failed_queries.txt", "w")
        file.close()
        
    print(f"Query cannot be found/executed...\n")
    with open('failed_queries.txt', 'a') as F:
        F.write("\n")
        F.write(cur_query)
        F.close()
    print("Query saved...\n")
    return
        
        
def process_general_query(cur_query, cosine_search_threshold=0.6, k_neighbours=3):
# returns zip list of cosineSim and Id of queries
    
    cur_query_copy = cur_query
    cur_query = llm_model.embed_query(cur_query)
    cur_query = np.array([cur_query])
    cur_query /= np.linalg.norm(cur_query, axis=1, keepdims=True)
    dist, idx = cosineIndex_llm.search(cur_query, k_neighbours)
    dist = dist.squeeze()
    idx = idx.squeeze()
    
    query_list=list(zip(idx, dist)) # Contains id of questions that are similar
    print(query_list)
    filtered_query = []
    filtered_id = []
    # Eliminate questions with cosine similarity lower than threshold
    for i in range(len(query_list)):
        (query_id, query_sim) = query_list[i]
        if (query_sim<cosine_search_threshold):
            continue

        if query_dict[query_id] not in filtered_query:
            filtered_query.append(query_dict[query_id]) # query_dict[i] -> fn_dict key
            filtered_id.append(query_list[i])
        else:
            continue

    if len(query_list)==0:
        process_empty_list(cur_query_copy)

    return filtered_id



def process_selected_query(fn_id, uploaded_signal_path):
    output = fn_dict[fn_id](uploaded_signal_path=uploaded_signal_path, k_neighbours=5)
    return output
        


def streamlit_find_k_most_similar():
    
    return