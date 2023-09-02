import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd

from scipy.io import wavfile as wavfile
from scipy.signal import stft

import streamlit as st
from scipy.signal import resample

from datetime import datetime


@st.cache_data
def tensorify(Zxx):
    input = torch.tensor(Zxx)
    # Add an extra dimension to represent the channels
    input = input.unsqueeze(0)
    # Add an extra dimension to represent the batch size
    input = input.unsqueeze(0)
    return input


@st.cache_data
def normalise(data, sr):
    data = data.astype(np.float32)
    data = resample(data, int(len(data) / sr * 8000))
    f, t, Zxx = stft(data, 8000)
    Zxx = np.abs(Zxx)

    # normalisation
    Zxx = Zxx + 1
    Zxx = np.log(Zxx)
    Zxx = Zxx / np.median(Zxx)

    return f, t, Zxx


@st.cache_data
def plot_spec(t, f, Zxx, title="Spectrogram", cmap="inferno"):
    if not os.path.exists("_figures"):
        os.makedirs("_figures")
    save_path = os.path.join("_figures", "tmp.png")
    plt.pcolormesh(t, f, np.abs(Zxx), cmap=cmap)
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.savefig(save_path)
    plt.clf()
    return Image.open(save_path)


@st.cache_data
def plot_distribution(labels, timeframe):
    signal_count = get_signal_count(labels, timeframe)
    plt.bar(list(signal_count.keys()), list(signal_count.values()))
    plt.xlabel("Signal Type")
    plt.ylabel("Count")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig("./_figures/data_distribution.jpg")
    return Image.open("./_figures/data_distribution.jpg")


@st.cache_data
def plot_time_distribution(labels, selected_signal_type):
    timestamps = []
    for filename, signal_types in labels.items():
        if selected_signal_type in signal_types:
            timestamps.append(get_datetime(filename))

    timestamps = pd.to_datetime(timestamps)
    df = pd.DataFrame({"timestamp": timestamps})

    time_interval = "5T"

    df["time_bin"] = df["timestamp"].dt.floor(time_interval).dt.time

    # Count the occurrences in each bin
    data_frequency = df["time_bin"].value_counts().sort_index()
    data_frequency.index = data_frequency.index.astype(str)

    # Plot the frequency of occurrence over time
    plt.figure(figsize=(10, 6))
    plt.plot(
        data_frequency.index,
        data_frequency.values,
        marker="o",
        linestyle="-",
        color="b",
    )
    plt.xlabel("Time")
    plt.ylabel("Frequency of Occurrence")
    plt.title("Frequency of Data Occurrence Over Time")
    plt.xticks(rotation=45)
    y_ticks = np.arange(0, data_frequency.values.max() + 1, 1)
    plt.yticks(y_ticks)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("_figures/" + selected_signal_type + ".png")
    return Image.open("_figures/" + selected_signal_type + ".png")


@st.cache_data
def get_signal_count(labels, timeframe=None):
    signal_count = {}
    for filename, ls in labels.items():
        time = get_datetime(filename).time()
        if timeframe == None or timeframe[0] < time < timeframe[1]:
            for label in ls:
                if label not in signal_count:
                    signal_count[label] = 0
                signal_count[label] += 1
    return signal_count


@st.cache_data
def k_most_similar(_simclr_model, _simclr_index, wav_file, signals, k_neighbours=3):
    # get spectrogram from wavFile
    sr, wav = wavfile.read(wav_file)
    f, t, Zxx = normalise(wav, sr)
    Zxx = tensorify(Zxx)

    # Get vector embedding of wavfile
    vector_embed = _simclr_model(Zxx)

    # Normalise vector embedding
    vector_embed = vector_embed.detach().numpy()
    vector_embed /= np.linalg.norm(vector_embed, axis=1, keepdims=True)

    # Search vector database
    similarity, idx = _simclr_index.search(vector_embed, k_neighbours + 1)
    similarity = similarity.squeeze()
    idx = idx.squeeze()

    files = [signals[i] for i in idx.tolist()]

    return (files[1:], similarity.tolist()[1:])


@st.cache_data
def get_datetime(file_name):
    file_name = os.path.basename(file_name)
    timestamp = int(file_name.split("_")[1])
    return datetime.fromtimestamp(timestamp)
