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

SIZE = (1080, 1920, 3)
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
    video_writer = cv2.VideoWriter(file_path, fourcc, FPS, (SIZE[1], SIZE[0]))
    for _ in range(frames):
        frame = (np.random.rand(*SIZE) * 256).astype('uint8')
        video_writer.write(frame)
    video_writer.release()
    return file_path


def is_number(number: str) -> bool:
    try:
        float(number)
    except ValueError:
        return False
    return True


@pytest.fixture
def random_image_payload():
    img = random_image()
    files = [
        ('file', ('img.jpg', img, 'image/jpeg'))
    ]
    return files


@pytest.fixture
def random_video_payload():
    video = random_video()
    with open(video, 'rb') as video_stream:
        video_bytes = video_stream.read()
    files = [
        ('file', ('video.mp4', video_bytes, 'video/mp4'))
    ]
    return files


class TestApi:
    # Initialize client
    client = TestClient(app)

    def test_apiroot(self):
        response = self.client.get("/")
        assert response.status_code == HTTPStatus.OK
        assert response.json()['data']['message'] == "Welcome to Drone-CrowdCounting! Please, read the `/docs`!"

    def test_apiimage(self, random_image_payload):
        url = "/predictions/images"
        response = self.client.post(url, headers={}, data={}, files=random_image_payload)
        assert response.status_code == HTTPStatus.OK

    def test_apivideo(self, random_video_payload):
        url = "/predictions/videos"
        response = self.client.post(url, headers={}, data={}, files=random_video_payload)
        assert response.status_code == HTTPStatus.OK

    def test_apivideo_isnull(self):
        url = "/predictions/videos"
        response = self.client.post(url, headers={}, data={}, files=[])
        assert response.json()['detail'][0]['type'] == "value_error.missing"

    def test_apiimg_isnull(self):
        url = "/predictions/videos"
        response = self.client.post(url, headers={}, data={}, files=[])
        assert response.json()['detail'][0]['type'] == "value_error.missing"

    @pytest.mark.parametrize(
        "count, heatmap",
        [
            (True, True),
            (True, False),
            (False, False),
        ],
    )
    def test_apiimg_params(self, count, heatmap, random_image_payload):
        url = "/predictions/images" + "?count=" + str(count) + "&heatmap=" + str(heatmap)

        response = self.client.post(url, headers={}, data={}, files=random_image_payload)
        if count and heatmap:
            assert response.headers["count"] >= "0.0"
        elif count and not heatmap:
            assert response.json()["count"] >= "0.0"
        elif not count and not heatmap:
            text = json.loads(response.text)
            assert text['detail'] == "Why predict something and not wanting any result?"

    @pytest.mark.parametrize(
        "count, heatmap",
        [
            (True, True),
            (True, False),
            (False, False),
        ],
    )
    def test_apivideo_params(self, count, heatmap, random_video_payload):
        url = "/predictions/videos" + "?count=" + str(count) + "&heatmap=" + str(heatmap)

        response = self.client.post(url, headers={}, data={}, files=random_video_payload)
        if count and heatmap:  # (True, True)
            assert response.headers['n_frames'] >= "0.0"
            for frame in range(int(float(response.headers['n_frames']))):
                assert is_number(response.headers[str(int(frame))])
        elif count and not heatmap:  # (True, False)
            assert response.headers['n_frames'] >= "0.0"
            for frame in range(int(float(response.headers['n_frames']))):
                assert is_number(response.json()[int(frame)]['count'])
        elif not count and not heatmap:  # (False, False)
            text = json.loads(response.text)
            assert text['detail'] == "Why predict something and not wanting any result?"

    @pytest.mark.parametrize(
        "count, heatmap",
        [
            (False, True),
            (True, True)
        ],
    )
    def test_apiimg_type(self, count, heatmap, random_image_payload):
        url = "/predictions/images" + "?count=" + str(count) + "&heatmap=" + str(heatmap)
        response = self.client.post(url, headers={}, data={}, files=random_image_payload)
        assert response.headers["content-type"] == "image/png"

    @pytest.mark.parametrize(
        "count, heatmap",
        [
            (False, True),
            (True, True)
        ],
    )
    def test_apivideo_type(self, count, heatmap, random_video_payload):
        url = "/predictions/videos" + "?count=" + str(count) + "&heatmap=" + str(heatmap)
        response = self.client.post(url, headers={}, data={}, files=random_video_payload)
        assert response.headers["content-type"] == "video/mp4"

    def test_apiimg_noinput_status_code(self, random_image_payload):
        url = "/predictions/images" + "?count=False" + "&heatmap=False"
        response = self.client.post(url, headers={}, data={}, files=random_image_payload)
        assert response.status_code == int("404")

    def test_apivideo_noinput_status_code(self, random_image_payload):
        url = "/predictions/videos" + "?count=False" + "&heatmap=False"
        response = self.client.post(url, headers={}, data={}, files=random_image_payload)
        assert response.status_code == int("404")
