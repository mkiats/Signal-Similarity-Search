FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN apt-get update && apt-get install -y build-essential gcc 
RUN rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
EXPOSE 8501
ENTRYPOINT ["streamlit", "run", "gui_for_llm.py", "--server.port", "8501"]