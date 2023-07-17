import os
import torch
import streamlit as st
import faiss as FF
import numpy as np
import torch.nn as nn
from scipy.io import wavfile as wavfile
from scipy.signal import stft as stft
from faiss import write_index, read_index
from langchain.embeddings import LlamaCppEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter


# Basic block for ResNet _________________________________________________________________________________________________
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
def initialise_signal_vector_db(signal_dataset_dir, write_index_dir, model):
    print("Adding vector db for signal")
    # Creates new index if index yet to exist
    if not os.path.exists(os.path.join(write_index_dir)):
        vector_len = 128
        faiss_index = FF.IndexFlatIP(vector_len) # EDIT: Require new directory if embedding function changes
        write_index(faiss_index, write_index_dir)
    
    # Initialises the faiss index
    faiss_index = read_index(write_index_dir)
    
    # Adds signals into the vector db
    files=os.listdir(signal_dataset_dir)
    for i in range(len(files)):
        print(files[i])
        file_path = os.path.join(signal_dataset_dir, files[i])

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
    print("Signal vector db done\n")
    
    
    
    
    
# Initialise question vector DB
def initialise_llm_vector_db(question_dataset_dir, write_index_dir, model):
    print("Adding vector db for llm")
    # Creates new index if index yet to exist
    if not os.path.exists(os.path.join(write_index_dir)):
        vector_len = len(model.embed_query(" "))
        faiss_index = FF.IndexFlatIP(vector_len) # EDIT: Require new directory if embedding function changes
        write_index(faiss_index, write_index_dir)
    
    # Initialises the faiss index
    faiss_index = read_index(write_index_dir)
    
    # Loads the documents
    loader = TextLoader(question_dataset_dir)
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
        query_embed = model.embed_query(docs[i].page_content)
        query_embed = np.array([query_embed])
        query_embed /= np.linalg.norm(query_embed, axis=1, keepdims=True)
        faiss_index.add(query_embed)
    
    # Saves the index
    write_index(faiss_index, write_index_dir)
    print("llm vector db done\n")