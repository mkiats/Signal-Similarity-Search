from langchain.embeddings import LlamaCppEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader

embeddings = LlamaCppEmbeddings(model_path=r"C:\Users\mengk\Projects-Jupyter\DeepLearningWavFiles\models\nous-hermes-13b.ggmlv3.q4_0.bin", n_ctx= 2048)

from langchain.document_loaders import TextLoader
loader = TextLoader('./faiss_documents/questionSet.txt')
documents = loader.load()
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size = 50, 
    chunk_overlap = 0,
    length_function = len)
docs = text_splitter.split_documents(documents)

db = FAISS.from_documents(docs, embeddings)
db.save_local("./faiss_index/faiss_questionSet")

loader1 = TextLoader('./faiss_documents/modulationSet.txt')
documents1 = loader1.load()
text_splitter1 = CharacterTextSplitter(
    separator="\n",
    chunk_size = 50, 
    chunk_overlap = 0,
    length_function = len)
docs1 = text_splitter.split_documents(documents1)

db1 = FAISS.from_documents(docs1, embeddings)
db1.save_local("./faiss_index/faiss_modulationSet")