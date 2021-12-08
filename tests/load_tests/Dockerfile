FROM python:3.9-slim
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
COPY requirements.txt requirements.txt 
RUN pip install -r requirements.txt 
COPY locustfile.py locustfile.py
EXPOSE 8089
CMD ["locust", "-f", "locustfile.py", "--host", "http://127.0.0.1:8000"]