import pytest
from fastapi.testclient import TestClient
from http import HTTPStatus
from api import app
import os
import cv2
from PIL import Image as pil
import numpy as np
import uuid
import io
import json

## ATTENZIONE: PER OGNI TEST CREATO, MODIFICARE LA CARTELLA DI ROOT DALLE CONFIGURAZIONI DI TEST!
### API TESTING ###

SIZE = (1920, 1080, 3)
MIN_VIDEO_FRAMES = 50
MAX_VIDEO_FRAMES = 51
FPS = 30
TEMP_DIR = 'tmp_pytest'

def random_image():
    """
    Generate a random image file and saves it
    @return: Path of the image file
    """
    bytes_pred = io.BytesIO()
    array = (np.random.rand(*SIZE) * 256).astype('uint8')
    pil.fromarray(array).save(bytes_pred, format='JPEG')
    bytes_pred.seek(0)
    return bytes_pred


def random_video():
    """
    Generate a random video file and saves it
    @return: Path of the video file
    """
    frames = np.random.randint(MIN_VIDEO_FRAMES, MAX_VIDEO_FRAMES)
    tmp = os.path.join(TEMP_DIR, uuid.uuid4().hex)
    os.makedirs(tmp, exist_ok=True)
    file_path = os.path.join(tmp, 'random_video.mp4')
    # generate a random video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(file_path, fourcc, FPS, SIZE[0:2])
    for _ in range(frames):
        video_writer.write((np.random.rand(*SIZE) * 256).astype('uint8'))
    video_writer.release()
    return file_path

class TestApi:
    # Initialize client
    client = TestClient(app)

    def test_apiroot(self):
        response = self.client.get("/")
        assert response.status_code == HTTPStatus.OK
        assert response.json()['data']['message'] == "Welcome to Drone-CrowdCounting! Please, read the `/docs`!"

    def test_apiimage(self):
        img = random_image()
        url = "/predictions/images"
        files = [
            ('file', ('img.jpg', img, 'image/jpeg'))
        ]
        response = self.client.post(url, headers={}, data={}, files=files)
        assert response.status_code == HTTPStatus.OK



    def test_apivideo(self):
        video = random_video()
        url = "/predictions/videos"
        files = [
            ('file', ('video.mp4', video, 'video/mp4'))
        ]

        response = self.client.post(url, headers={}, data={}, files=files)
        assert response.status_code == HTTPStatus.OK


    def test_apivideo_isnull(self):
        video = random_video()
        url = "/predictions/videos"
        files = [
            ('file', ('video.mp4', video, 'video/mp4'))
        ]

        response = self.client.post(url, headers={}, data={}, files=[])
        assert response.json()['detail'][0]['type'] == "value_error.missing"


    def test_apiimg_isnull(self):
        img = random_image()
        url = "/predictions/videos"
        files = [
            ('file', ('img.jpg', img, 'image/jpeg'))
        ]
        response = self.client.post(url, headers={}, data={}, files=[])
        assert response.json()['detail'][0]['type'] == "value_error.missing"


    def test_apiimg_count_body(self):
        img = random_image()
        url = "/predictions/images"+"?count=True"+"&heatmap=False"
        files = [
            ('file', ('img.jpg', img, 'image/jpeg'))
        ]
        response = self.client.post(url, headers={}, data={}, files=files)
        assert response.json()["count"] >="0.0"

    def test_apiimg_type(self):
        img = random_image()
        url = "/predictions/images"+"?count=False"+"&heatmap=True"
        files = [
            ('file', ('img.jpg', img, 'image/jpeg'))
        ]
        response = self.client.post(url, headers={}, data={}, files=files)
        assert response.headers["content-type"] == "image/png"

    def test_apiimg_noinput_one(self):
        img = random_image()
        url = "/predictions/images"+"?count=False"+"&heatmap=False"
        files = [
            ('file', ('img.jpg', img, 'image/jpeg'))
        ]
        response = self.client.post(url, headers={}, data={}, files=files)
        text= json.loads(response.text)
        assert text['detail']=="Why predict something and not wanting any result?"

    def test_apiimg_noinput_two(self):
        img = random_image()
        url = "/predictions/images"+"?count=False"+"&heatmap=False"
        files = [
            ('file', ('img.jpg', img, 'image/jpeg'))
        ]
        response = self.client.post(url, headers={}, data={}, files=files)
        assert response.status_code == int("404")
