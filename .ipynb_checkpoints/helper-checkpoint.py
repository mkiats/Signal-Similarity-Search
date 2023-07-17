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


# Basic block for ResNet
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(identity)
        out = self.relu(out)

        return out

    
    
# ResNet model
class ResNet(nn.Module):
    def __init__(self, embedding_size):
        super(ResNet, self).__init__()
        self.embedding_size = embedding_size

        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, self.embedding_size)

        # Initialize the weights
        self._initialize_weights()

    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(ResBlock(in_channels, out_channels, stride=stride))
        for _ in range(1, num_blocks):
            layers.append(ResBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out

    
    
def tensorify(Zxx):
    input = torch.tensor(Zxx)
    # Add an extra dimension to represent the channels
    input = input.unsqueeze(0)
    # Add an extra dimension to represent the batch size
    input = input.unsqueeze(0)
    return input



def normalise(data, sr):
    f, t, Zxx = stft(data, sr, nfft=1024)

    # Get bandwidth 0 to 4000
    f = np.array(f)[np.array(f) <= 4000]
    Zxx = Zxx[:len(f),:]

    Zxx = np.abs(Zxx)

    # normalisation
    Zxx = Zxx + 1
    Zxx = np.log(Zxx)
    Zxx = Zxx/np.median(Zxx)

    return f, t, Zxx





# Initialising signal vector DB
def initialise_signal_vector_db(signal_dir, write_index_dir, model):
    print("Adding vector db for signal")
    # Creates new index if index yet to exist
    if not os.path.exists(os.path.join(write_index_dir)):
        vector_len = 128
        cosineindex_signal = FF.IndexFlatIP(vector_len) # EDIT: Require new directory if embedding function changes
        write_index(cosineindex_signal, write_index_dir)
    
    # Initialises the faiss index
    faiss_index = read_index(write_index_dir)
    
    # Adds signals into the vector db
    files=os.listdir(signal_dir)
    for i in range(len(files)):
        print(files[i])
        file_path = os.path.join(signal_dir, files[i])

        # get spectrogram from wavFile
        sr, wav = wavfile.read(file_path)
        f, t, Zxx = normalise(wav, sr)
        Zxx = tensorify(Zxx)

        # Get vector embedding of wavfile
        vector_embed = model(Zxx)

        # Normalise vector embedding
        vector_embed = vector_embed.detach().numpy()
        vector_embed /= np.linalg.norm(vector_embed, axis=1, keepdims=True)

        # Add to vector database
        faiss_index.add(vector_embed)
    
    # Saves the index
    write_index(faiss_index, write_index_dir)
    print("\ndone")
    
    
    
    
    
# Initialise question vector DB
def initialise_llm_vector_db(text_loading_dir, write_index_dir, embed_fn):
    print("Adding vector db for llm")
    # Creates new index if index yet to exist
    if not os.path.exists(os.path.join(write_index_dir)):
        vector_len = len(embed_fn.embed_query(" "))
        cosineindex_llm = FF.IndexFlatIP(vector_len) # EDIT: Require new directory if embedding function changes
        write_index(cosineindex_llm, write_index_dir)
    
    # Initialises the faiss index
    faiss_index = read_index(write_index_dir)
    
    # Loads the documents
    loader = TextLoader(text_loading_dir)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size = 50, 
        chunk_overlap = 0,
        length_function = len)
    docs = text_splitter.split_documents(documents)
    
    # Adds documents into vector db
    for i in range(len(docs)):
        print(docs[i].page_content)
        query_embed = embed_fn.embed_query(docs[i].page_content)
        query_embed = np.array([query_embed])
        query_embed /= np.linalg.norm(query_embed, axis=1, keepdims=True)
        faiss_index.add(query_embed)
    
    # Saves the index
    write_index(faiss_index, write_index_dir)
    print("llm vector db done")

    
    
    
    
# Helper functions for LLM querying
# Generate similar queries that exceeds the threshold
def process_empty_list(cur_query):
    if not os.path.exists("failed_queries.txt"):
        file = open("failed_queries.txt", "w")
        file.close()
        
    print(f"Query cannot be found/executed...\n",
          "Do you wish to save this query for future development purposes?\n")
    add_to_development = int(input("1 for yes, 0 for no..."))
    if add_to_development==1:
        with open('failed_queries.txt', 'a') as F:
            F.write(cur_query)
            F.write("\n")
            F.close()
        print("Query saved...\n")
        
        
def process_general_query(cur_query, llm, cosineindex_llm, cosine_search_threshold=0.6, k_neighbours=3):
# returns zip list of cosineSim and Id of queries
    
    cur_query_copy = cur_query
    cur_query = llm.embed_query(cur_query)
    cur_query = np.array([cur_query])
    cur_query /= np.linalg.norm(cur_query, axis=1, keepdims=True)
    dist, idx = cosineindex_llm.search(cur_query, k_neighbours)
    dist = dist.squeeze()
    idx = idx.squeeze()
    print(dist)
    query_list=list(zip(dist, idx)) # Contains id of questions that are similar
    # Eliminate questions with cosine similarity lower than threshold
    for i in reversed(range(len(query_list))):
        (query_sim, query_id) = query_list[i]
        if (query_sim<cosine_search_threshold):
            query_list.pop(i)
            
    if len(query_list)==0:
        process_empty_list(cur_query_copy)
    return query_list

fn_dict = {
        0: find_k_most_similar,
        1: predict_signal,
        2: summarise_signal,
        3: summarise_dataset,   
        4: show_accuracy_score_dataset,
        5: show_model,
        6: show_hyperpara_model,
        7: show_loss_curve_model
    }

def process_selected_query(query_id, cosineindex_llm, cosineindex_signal, llama, simCLR):
    uploaded_signal = os.path.join("data/train", "train_ALIS.wav")
    print(fn_dict[query_id])
    output = fn_dict[query_id](cosineindex_signal=cosineindex_signal,
                      cosineindex_llm=cosineindex_llm,
                      uploaded_signal=uploaded_signal,
                      llm_model=llama,
                      simclr_model=simCLR, 
                      k_neighbours=3)
    return output
        
              

        
# Helper function for the signals
def find_k_most_similar(cosineindex_signal, cosineindex_llm, uploaded_signal, llm_model, simclr_model, k_neighbours):

    # get spectrogram from wavFile
    sr, wav = wavfile.read(uploaded_signal)
    f, t, Zxx = normalise(wav, sr)
    Zxx = tensorify(Zxx)

    # Get vector embedding of wavfile
    vector_embed = simclr_model(Zxx)

    # Normalise vector embedding
    vector_embed = vector_embed.detach().numpy()
    vector_embed /= np.linalg.norm(vector_embed, axis=1, keepdims=True)

    # Search vector database
    similarity, idx = cosineindex_signal.search(vector_embed, k_neighbours)
    similarity = similarity.squeeze()
    idx = idx.squeeze()
    
    res=zip(idx, similarity) # res = [(x1, y1), (x2, y2)] where x is the index, y is the cosine similarity
    return list(res)

def predict_signal(cosineindex_signal, cosineindex_llm, uploaded_signal, llm_model, simclr_model, k_neighbours):
    print("TODO")
    return

def summarise_signal(cosineindex_signal, cosineindex_llm, uploaded_signal, llm_model, simclr_model, k_neighbours):
    sr, wav = wavfile.read(uploaded_signal)
    f, t, Zxx = normalise(wav, sr)
    print(f"Sample Rate is {sr}")
    plot_spec_signal(t, f, Zxx)
    return
    
def plot_spec_signal(t, f, Zxx, title="Spectrogram", cmap="magma"):
    plt.pcolormesh(t, f, np.abs(Zxx), cmap=cmap)
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.colorbar()
    plt.show()
    # TODO: plot spectrogram via Streamlit
    return

# Helper functions for questions on dataset
def summarise_dataset(cosineindex_signal, cosineindex_llm, uploaded_signal, llm_model, simclr_model, k_neighbours):
    # TODO: show distribution of train and validation set
    return 

def show_accuracy_score_dataset(cosineindex_signal, cosineindex_llm, uploaded_signal, llm_model, simclr_model, k_neighbours):
    # TODO: plot graph: Accuracy score against modulationn type
    return

# Helper functions for questions on model
def show_model(cosineindex_signal, cosineindex_llm, uploaded_signal, llm_model, simclr_model, k_neighbours):
    print("SimCLR model used")
    print("Model architecture is as such")
    print(simclr_model)
    return

def show_hyperpara_model(cosineindex_signal, cosineindex_llm, uploaded_signal, llm_model, simclr_model, k_neighbours):
    NUM_EPOCHS = 15
    LR = 0.00005
    BATCH_SIZE = 32
    embedding_size = 128
    
    print(f"Learning rate is {LR}\n",
          f"Num of epochs is {NUM_EPOCHS}\n",
          f"Batch size is {BATCH_SIZE}",
          f"Final embedding size is {embedding_size}"
          )
    return

def show_loss_curve_model(cosineindex_signal, cosineindex_llm, uploaded_signal, llm_model, simclr_model, k_neighbours):
    # TODO plot loss curve
    return
