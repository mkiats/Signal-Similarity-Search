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
from helper_fn import *
from helper_class import *


st.title("LLM + SimCLR")

# request = input("What do you wish to find out?\n")
# st.write(cosineindex_llm.ntotal)
print("Question set has ", cosineIndex_llm.ntotal)

request = st.text_input("What do you wish to find out?", "")
while (request == ""):
    continue
if request != "":
    filtered_list = process_general_query(request, cosine_search_threshold=0.6, k_neighbours=5)
    # If function exists
    if len(filtered_list)!=0:
        maxCount = 0
        for count, item in enumerate(filtered_list):
            st.write(f"{count}. {docs[item[0]].page_content}")
            maxCount=count
        selected_query = st.text_input("Choose which query to execute? Input -1 if none of the queries matches ur request...", "")

        while (selected_query == "" or int(selected_query) >= maxCount):
            st.write("Invalid Input, pls input a valid query ID... ")

        # User has selected a query
        while (selected_query != ""):
            if int(selected_query) <0 :
                process_empty_list(request)
            else:    
                # Upload file widget
                upload = st.file_uploader("Choose a file...")
                # Catch empty uploads
                if (upload == None):
                    st.write("No file detected...")
                # Process upload
                # if (upload != None):
                else:
                    query_dict_key = filtered_list[int(selected_query)][0]
                    image_file_count = 0
                    reinitialise_path("_testImage")
                    reinitialise_path("_trainImage")
                    # Query for find_k_most_similar
                    if query_dict[query_dict_key] == find_k_most_similar:
                        # Information of uploaded signal for spectrogram plotting
                        upload_sr, upload_t, uploaded_f, upload_Zxx = summarise_signal(upload)
                        uploaded_signal_image_path = os.path.join("_testImage", "image.png")
                        plot_spec_signal(upload_t, uploaded_f, upload_Zxx, uploaded_signal_image_path)
                        # Information of similar signal
                        output = process_selected_query(query_dict_key, upload)
                        signals = os.listdir(signal_dataset_path)
                        for i,j in output:
                            similar_signal_path = os.path.join("_trainImage", signals[i])
                            sim_signal_sr, sim_signal_t, sim_signal_f, sim_signal_Zxx = summarise_signal(similar_signal_image_path)
                            similar_signal_image_path = os.path.join(similar_signal_path, f"{image_file_count}.png")
                            plot_spec_signal(sim_signal_t, sim_signal_f, sim_signal_Zxx, similar_signal_image_path)
                            image_file_count+=1

                        st.set_page_config(page_title="Image-Comparison Example", layout="centered")
                        # render image-comparison
                        image_comparison(
                            img1="_testImage/image.png",
                            img2="_trainImage/0.png",
)
                

# streamlit run gui_for_llm.py