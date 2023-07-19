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
request = st.text_input("What do you wish to find out?", "") # COMMENT
# request = input("type your query...")
while (request == ""):
    continue
if request != "":
    filtered_list = process_general_query(request, cosine_search_threshold=0.6, k_neighbours=5)
    print(f"Filtered list is {filtered_list}")
    if len(filtered_list)==0:
        process_empty_list(request)
    # If function exists
    if len(filtered_list)!=0:
        maxCount = 0
        for count, item in enumerate(filtered_list):
            # st.write(f"{count}. {docs[item[0]].page_content}" ) # COMMENT
            print(f"{count}. {docs[item[0]].page_content}")
            maxCount=count
        # selected_query = st.text_input("Choose which query to execute? Input -1 if none of the queries matches ur request...", "") # COMMENT
        selected_query = input("Choose which query to execute? Input -1 if none of the queries matches ur request...")

        
        # Catch invalid queries for <selected_query>
        if (selected_query == "" or int(selected_query) > maxCount):
            # st.write("Invalid Input, pls input a valid query ID... ") # COMMENT
            print("Invalid Input, pls input a valid query ID... ")


        # User has selected a query
        while (selected_query != ""):
            selected_query = int(selected_query)
            fn_dict_key = query_dict[filtered_list[selected_query][0]]
            print(f"FnDictKey = {fn_dict_key}, selectedQuery = {selected_query}")
            if int(selected_query) <0 :
                process_empty_list(request)
            else:    
                # Upload file widget
                # upload = st.file_uploader("Choose a file...") # COMMENT
                upload = "_trial_data/train/train_ALIS.wav"
                # Catch empty uploads
                if (upload == None):
                    st.write("No file detected...")
                # Process upload
                # if (upload != None):
                else:
                    reinitialise_path("_testImage")
                    reinitialise_path("_trainImage")
                    # Query for find_k_most_similar
                    if fn_dict[fn_dict_key] == find_k_most_similar:
                        print("find k most similar")
                        # Information of uploaded signal for spectrogram plotting
                        upload_sr, upload_t, uploaded_f, upload_Zxx = summarise_signal(upload)
                        uploaded_signal_image_path = os.path.join("_testImage", "image.png")
                        plot_spec_signal(upload_t, uploaded_f, upload_Zxx, uploaded_signal_image_path)
                        # Information of similar signal
                        image_file_count = 0
                        output = process_selected_query(fn_dict_key, upload)
                        signals = os.listdir(signal_dataset_path)
                        print(output)
                        for i,j in output:
                            print(signals[i])
                            similar_signal_path = os.path.join(signal_dataset_path, signals[i])
                            sim_signal_sr, sim_signal_t, sim_signal_f, sim_signal_Zxx = summarise_signal(similar_signal_path)
                            similar_signal_image_path = os.path.join("_trainImage", f"{image_file_count}.png")
                            plot_spec_signal(sim_signal_t, sim_signal_f, sim_signal_Zxx, similar_signal_image_path)
                            image_file_count+=1
                        


                    elif fn_dict[fn_dict_key] == summarise_signal:
                        print("sumarise signal")
                        upload_sr, upload_t, uploaded_f, upload_Zxx = process_selected_query(fn_dict_key, upload)
                        uploaded_signal_image_path = os.path.join("_testImage", "image.png")
                        plot_spec_signal(upload_t, uploaded_f, upload_Zxx, uploaded_signal_image_path)
                        break
                        
                    elif fn_dict[fn_dict_key] == summarise_dataset:
                        print("summarise dataset")
                        break
                    elif fn_dict[fn_dict_key] == show_similar_signal_timestamp:
                        print("show_similar_signal_timestamp")
                        process_selected_query(fn_dict_key, upload)
                        break
                    elif fn_dict[fn_dict_key] == show_model:
                        print("show_model")
                        process_selected_query(fn_dict_key, upload)
                        break
                    elif fn_dict[fn_dict_key] == show_hyperpara_model:
                        print("show_hyperpara_model")
                        process_selected_query(fn_dict_key, upload)
                        break
                    elif fn_dict[fn_dict_key] == show_loss_curve_model:
                        print("show_loss_curve_model")
                        x, y = process_selected_query(fn_dict_key, upload)
                        print(x, "|", y)
                        break
                    elif fn_dict[fn_dict_key] == invalid_fn:
                        print("invalid_fn")
                        break
                    else:
                        break
            break



                
# streamlit run gui_for_llm.py