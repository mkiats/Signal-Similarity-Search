import os
import numpy as np
import streamlit as st
import faiss
from faiss import write_index, read_index
from scipy.io import wavfile as wavfile
from process_signal import normalise, tensorify
from model import ResNet
import torch
from sentence_transformers import SentenceTransformer
import pandas as pd

EMBEDDING_SIZE = 128


# Initialise question vector DB
@st.cache_resource
def initialise_llm(questionSet_dir, index_dir):
    # model = SentenceTransformer('all-MiniLM-L6-v2')
    model = SentenceTransformer("all-mpnet-base-v2")

    # add questions so that they can be referenced later
    questions = []
    question_mapping = {}
    idx = 0
    with open(questionSet_dir, "r") as file:
        for line in file:
            words = line.split()
            questionId = int(words[-1])
            text = " ".join(words[:-1])
            questions.append(text)
            question_mapping[idx] = questionId
            idx += 1

    if os.path.exists(index_dir):
        # Loads index if already exists
        print("LLM question index found!")
        faiss_index = read_index(index_dir)
    else:
        print("Creating index for LLM question set...")
        # Creates new index if index yet to exist
        embeddings = []
        for question in questions:
            embedding = model.encode(question)
            embeddings.append(embedding)
        faiss_index = faiss.IndexFlatIP(len(embeddings[0]))
        faiss_index.add(np.array(embeddings))
        write_index(faiss_index, index_dir)
    print("LLM vector db done\n")
    return (model, faiss_index, questions, question_mapping)


# Initialising signal vector DB
@st.cache_resource
def initialise_simclr(
    model_weights_dir, signal_dataset_dir, pickle_dir, write_index_dir
):
    model = ResNet(128)
    model.load_state_dict(torch.load(model_weights_dir, map_location="cpu"))
    model.eval()

    signals = [
        os.path.join(signal_dataset_dir, file)
        for file in os.listdir(signal_dataset_dir)
        if file.endswith(".wav")
    ]

    # Save the labels for each wav file in a dictionary
    pickle = pd.read_pickle(pickle_dir)
    labels = {}
    for filename in signals:
        filename = os.path.basename(filename)
        key = os.path.splitext(filename)[0]
        labels[filename] = pickle["GTLabelnamesAll"][key]

    # Creates new index if index yet to exist
    faiss_index = faiss.IndexFlatIP(EMBEDDING_SIZE)
    if os.path.exists(os.path.join(write_index_dir)):
        # Read index if already exists
        print("Signal index found!")
        faiss_index = read_index(write_index_dir)
    else:
        # Adds signals into the vector db
        print("Creating index for signals...")
        for signal_path in signals:
            # get spectrogram from wavFile
            sr, wav = wavfile.read(signal_path)
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
    return (model, faiss_index, signals, labels)
