FROM python:3.10

WORKDIR /app
COPY . .

RUN apt-get update && apt-get install -y portaudio19-dev
RUN pip install streamlit torch transformers sentence-transformers faiss-cpu \
    PyPDF2 pyttsx3 SpeechRecognition

EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
