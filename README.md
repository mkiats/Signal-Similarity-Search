# SimCLR + LLM

SimCLR (A Simple Framework for Contrastive Learning of Visual Representations) is a self-supervised learning framework that has gained popularity in the field of deep learning for computer vision. It was [introduced by researchers at Google Research](https://github.com/google-research/simclr) to create semantic embeddings of images to be used in downstream tasks.

This work attempts to adapt SimCLR's contrastive learning model to the signal processing domain. To demonstrate the effectiveness of these embeddings in aiding the identification of unknown signals, we designed a signal query GUI using StreamLit. This GUI allows users to compare the embeddings of their signals against a database of known signals, facilitating quick and accurate signal identification.

Additionally, we incorporated [Sentence BERT](https://arxiv.org/abs/1908.10084) to generate semantic embeddings of user queries. These sentence embeddings are compared to our database of possible queries to determine what the user is asking for. This essentially enables dynamic user querying, improving users' experiences.

![SignalLLM](https://github.com/tyanhan/signal_llm/assets/68331979/09755cec-fe79-488f-9743-ee259473b345)

---

## Quick Start

1. Run `pip install requirements.txt`

2. Put your signal data in the form of `.wav` files under the `_data` directory. Store your labels in a pickle file named `label.pickle` and put it under the `_data` directory.

3. Place your question set under the `_faiss_documents` folder with file name `questionSet.txt`. Each question should be written on a new line with its question ID appended to the back of the question.

4. Write your saved model weights into `_model` folder with file name `model.pth`.

5. Run `streamlit run main.py`

:exclamation:**Delete the indexes under `_faiss_index` after modifying the question set and/or signal database to refresh the application**

## Developer's Guide

`main.py` is the entry point into the application. Upon running `streamlit run main.py`, the code initialises the modified ResNet model for creating signal embeddings and the Sentence Bert model for creating query embeddings.

### In `main.py`

`initialise_llm` takes in the question set's directory and the question index's directory. If the question index already exists, it simply stores the questions into an array and the question to question ID mapping in a dictionary for future reference. Otherwise, it creates a new index for the question set. **The model used to generate query embeddings can be modified in the `initialise_llm` function.**

`initialise_simclr` takes in the model weights' directory, signal dataset directory, pickle directory and signal index directory. If the signal index already exists, the function simply stores the signal file paths in a list and the labels in a dictionary for future reference. Otherwise it also creates a new index for the signal database.

After initialising both models, the GUI is ready to accept user inputs. The `get_questionIds` function converts users' queries into a question ID based on most similar question search in the question set. The `process_question` function then takes in the question ID and invokes the function which the user requests.

### Other files/folders

`model.py` consists of the signal model used for creating signal embeddings.

`process_query.py` consists of functions related to query processing such as converting a question into a question ID. The `process_query` function here determines which function to execute based on the given question ID

`process_signal.py` consists of functions related to signal processing such as signal normalisation, spectrogram plotting etc.

`_figures` consists of images created to be rendered on the GUI

`_information` consists of information about the different signal types

`failed_queries.txt` logs all users' queries that were not sufficiently similar to any of the questions in the question set
