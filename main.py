import streamlit as st

from initialise import initialise_llm, initialise_simclr
from process_query import get_questionIds, process_empty_list, process_question


llm_model, llm_index, questions, question_mapping = initialise_llm(
    "./_faiss_documents/questionSet.txt", "./_faiss_index/faiss_questionSet"
)
simclr_model, simclr_index, signals, labels = initialise_simclr(
    "_model/model.pth",
    "./_data",
    "./_data/labels.pickle",
    "./_faiss_index/faiss_signalSet",
)

st.title("Signal Database")
placeholder = st.container()

request = placeholder.text_input("What do you wish to find out?", "")

if request != "":
    # get all potential queries that the user is asking
    filtered_questions, questionIds = get_questionIds(
        request, llm_model, llm_index, questions, question_mapping
    )

    # no valid queries
    if len(questionIds) == 0:
        process_empty_list(request)
        placeholder.write(
            "Unfortunately, I do not have the capability to process your query. If you need assistance with a different topic, feel free to ask, and I'll do my best to provide helpful answers."
        )
    else:
        # else process question
        questionId = questionIds[0]
        process_question(
            questionId,
            placeholder,
            simclr_model,
            simclr_index,
            signals,
            labels,
        )
