import os
import numpy as np
import streamlit as st

from scipy.io import wavfile as wavfile
from datetime import timedelta

from process_signal import (
    normalise,
    k_most_similar,
    plot_spec,
    plot_distribution,
    plot_time_distribution,
    get_signal_count,
    get_datetime,
)


@st.cache_data
def process_empty_list(cur_query):
    if not os.path.exists("failed_queries.txt"):
        file = open("failed_queries.txt", "w")
        file.close()

    with open("failed_queries.txt", "a") as F:
        F.write(cur_query)
        F.write("\n")
        F.close()
    print("Query saved...\n")
    return -1


def get_questionIds(
    query,
    _llm_model,
    _llm_index,
    questions,
    question_mapping,
    threshold=0.3,
    k_neighbours=3,
):
    embedding = _llm_model.encode(query)
    print(len(embedding))
    dist, idx = _llm_index.search(np.array([embedding]), k_neighbours)
    dist = dist.squeeze().tolist()
    idx = idx.squeeze().tolist()

    print("Cosine Similarity:", dist)
    print("Most similar questions:", [questions[i] for i in idx])
    print("Question IDs:", [question_mapping[i] for i in idx])

    # Filter questions above threshold cosine similarity score
    dist = [d for d in dist if d > threshold]
    idx = idx[: len(dist)]

    questionIds = []
    filtered_questions = []
    for i in range(len(idx)):
        if question_mapping[idx[i]] not in questionIds:
            questionIds.append(question_mapping[idx[i]])
            filtered_questions.append(questions[idx[i]])

    return (filtered_questions, questionIds)


def process_question(
    questionId, placeholder, simclr_model, simclr_index, signals, labels
):
    # Find k most similar signals
    if questionId == 0:
        upload = placeholder.file_uploader(
            "Upload your wav file here...", type=[".wav"]
        )
        if upload == None:
            placeholder.write("No file detected...")
        else:
            placeholder.write(f"Your Signal:")
            sr, data = wavfile.read(upload)
            f, t, Zxx = normalise(data, sr)
            placeholder.image(plot_spec(t, f, Zxx))

            files, similarity = k_most_similar(
                simclr_model, simclr_index, upload, signals
            )

            placeholder.write("We found some similar signals in our database:")

            votes = {}
            for i in range(len(files)):
                filename = os.path.basename(files[i])
                sr, data = wavfile.read(files[i])
                f, t, Zxx = normalise(data, sr)
                dt = get_datetime(filename)
                placeholder.write(f"Similarity score: {round(similarity[i], 2)}")
                placeholder.write(f"Seen on: {dt}")
                placeholder.image(plot_spec(t, f, Zxx, labels[filename]))

                for label in labels[filename]:
                    if label not in votes:
                        votes[label] = 0
                    votes[label] += 1

            predicted_label = max(votes, key=votes.get)
            placeholder.write(
                f"The signal is most likely a {predicted_label} signal.\n"
            )

    # Plot spectrogram
    elif questionId == 1:
        upload = placeholder.file_uploader(
            "Upload your wav file here...", type=[".wav"]
        )
        if upload == None:
            placeholder.write("No file detected...")
        else:
            placeholder.write(f"Your Signal:")
            sr, data = wavfile.read(upload)
            f, t, Zxx = normalise(data, sr)
            placeholder.image(plot_spec(t, f, Zxx))

    # Plot data distribution
    elif questionId == 2:
        timestamps = [get_datetime(filename) for filename in labels.keys()]
        start_time = min(timestamps).time()
        end_time = max(timestamps).time()

        selected_time = st.slider(
            "Select Time",
            min_value=start_time,
            max_value=end_time,
            value=(start_time, end_time),
            step=timedelta(minutes=1),
        )
        placeholder.image(plot_distribution(labels, selected_time))

    # Details of signal type
    elif questionId == 3:
        signal_types = [file.split(".")[0] for file in os.listdir("_information")]
        tmp_placeholder = st.empty()
        selected_signal_type = tmp_placeholder.selectbox(
            "What signal type would you like to find out more about?",
            [""] + signal_types + ["My signal type is not listed!"],
        )
        if selected_signal_type != "":
            # if none of the options are valid, return error message
            if selected_signal_type == "My signal type is not listed!":
                placeholder.write(
                    "Unfortunately, we do not have information about your signal type."
                )
            else:
                placeholder.write(selected_signal_type)
                read_and_print_file(
                    placeholder, "_information/" + selected_signal_type + ".txt"
                )

    # plot time distribution of particular signal
    elif questionId == 4:
        signal_types = list(get_signal_count(labels).keys())
        tmp_placeholder = st.empty()
        selected_signal_type = tmp_placeholder.selectbox(
            "What signal type would you like to find out more about?",
            [""] + signal_types,
        )
        if selected_signal_type != "":
            placeholder.write(selected_signal_type)
            placeholder.image(plot_time_distribution(labels, selected_signal_type))
    else:
        placeholder.write(questionId)


def read_and_print_file(placeholder, file_path):
    try:
        with open(file_path, "r") as file:
            file_contents = file.read()
            if file_contents.strip():  # Check if the file is not empty
                placeholder.write(file_contents + "\n")
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"Error occurred: {e}")
