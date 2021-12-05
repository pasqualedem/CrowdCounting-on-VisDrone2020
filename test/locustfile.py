import io
import os
import uuid
import numpy as np
import cv2
import shutil

from locust import HttpUser, task, tag, between, events
from PIL import Image as pil

SIZE = (1920, 1080, 3)
MIN_VIDEO_FRAMES = 50
MAX_VIDEO_FRAMES = 51
FPS = 30

MIN_RUNS = 1
MAX_RUNS = 10
EXP_SCALE = 2.5  # Scale parameter of exponential distribution to calculate number of runs for each user

TEMP_DIR = 'tmp_locust'


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """
    Create directory to make virtual user work
    """
    os.makedirs(TEMP_DIR, exist_ok=True)


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """
    Delete directory created at startup
    """
    shutil.rmtree(TEMP_DIR)


class DroneUser(HttpUser):
    """
    User of Drone-CrowdCounting APIs. Sends a random number of requests governed by an exponential distribution.
    An interval between 5 and 10 seconds passes through each request.
    """
    wait_time = between(5, 10)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.runs = int(np.random.exponential(EXP_SCALE) + 1)
        self.cur_runs = 0

    def check_stop(self):
        """
        Checks if user has finished its tasks, and then stops it
        """
        self.cur_runs += 1
        if self.cur_runs == self.runs:
            self.stop()

    @task(1)
    def root(self):
        self.client.get("/")

    @task(1)
    def docs(self):
        self.client.get("/docs")

    @task(5)
    @tag("prediction", 'image')
    def img_prediction(self):
        url = "/predictions/images" + self.get_random_params()
        img_bytes = random_image()
        files = [
            ('file', ('img.jpg', img_bytes, 'image/jpeg'))
        ]
        self.client.post(url, headers={}, data={}, files=files)
        self.check_stop()

    @task(4)
    @tag("prediction", 'video')
    def video_prediction(self):
        url = "/predictions/videos" + self.get_random_params()
        video_path = random_video()
        with open(video_path, 'rb') as video_bytes:
            files = [
                ('file', ('video.mp4', video_bytes, 'video/mp4'))
            ]
            self.client.post(url, headers={}, data={}, files=files)
        os.remove(video_path)
        self.check_stop()

    @classmethod
    def get_random_params(cls):
        count = 'count=' + str(bool(np.random.randint(2))).lower()
        heatmap = 'heatmap=' + str(bool(np.random.randint(2))).lower()
        return '?' + count + '&' + heatmap


def random_image():
    """
    Generate a random video file and saves it
    @return: Path of the video file
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