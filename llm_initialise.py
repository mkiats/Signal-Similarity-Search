# Loading imports 
import os
import shutil
from langchain.embeddings import LlamaCppEmbeddings
from helper_class import *
print("\n\n\nImports loaded")



# Parameters to set
# llm_modelPath = os.path.join("_model", "alpaca-lora-65B.ggmlv3.q4_0.bin")
llm_modelPath = os.path.join("_model", "ggml-model-f16-q4_0.bin")
simclr_model_path = os.path.join("_model", "lr1e-05b32_30.pth")


# DO NOT TOUCH
llm_model = LlamaCppEmbeddings(model_path=llm_modelPath, n_ctx= 2048)
simclr_model = ResNet(128)
simclr_model.load_state_dict(torch.load(simclr_model_path, map_location="cpu"))
simclr_model.eval()
print("\nModels loaded\n")

# Initialises new environment
if os.path.exists(os.path.join("_faiss_index")):
    toBeRemoved = '_faiss_index'
    shutil.rmtree(toBeRemoved)

# Loading index and model
if not os.path.exists(os.path.join("_faiss_index")):
    os.makedirs(os.path.join("_faiss_index"))
print("_faiss_index reinitialised\n")

# Initialises new environment
if os.path.exists(os.path.join("_image")):
    toBeRemoved = '_image'
    shutil.rmtree(toBeRemoved)

# Loading index and model
if not os.path.exists(os.path.join("_image")):
    os.makedirs(os.path.join("_image"))
print("_image reinitialised\n")

# Make signal faiss db
initialise_signal_vector_db("data/train", "./_faiss_index/faiss_signalSet", simclr_model)
# Make llm faiss db
initialise_llm_vector_db('./_faiss_documents/questionSet.txt',  "./_faiss_index/faiss_questionset_65B", llm_model)
# initialise_llm_vector_db('./_faiss_documents/questionSet.txt',  "./new_faiss_index/faiss_questionset__7B", llm__7b)

print("\n\n\n Files initialised\n\n\n")