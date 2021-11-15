FROM python:3.9-buster

COPY requirements.txt requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt && apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

WORKDIR /droneCrowdCounting
COPY . /droneCrowdCounting
EXPOSE 80
CMD ["python", "src/api.py"]