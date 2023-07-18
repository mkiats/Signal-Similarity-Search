FROM python:3.11-slim

WORKDIR \Users\mengk\Projects-Jupyter\simCLR_Streamlit

COPY / ./

RUN apt-get update && apt-get install -y build-essential gcc mono-mcs

RUN rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "gui_for_llm.py", "--server.port=8501", "--server.address=0.0.0.0"]