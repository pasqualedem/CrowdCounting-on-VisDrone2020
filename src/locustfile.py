import io
import numpy as np
from locust import HttpUser, task, between
from PIL import Image as pil

SIZE = (1920, 1080, 3)


class DroneUser(HttpUser):
    wait_time = between(1, 5)

    @task
    def hello(self):
        self.client.get("/")

    @task
    def docs(self):
        self.client.get("/docs")


class ImageUser(DroneUser):
    weight = 5
    @task
    def get_img_prediction(self):
        url = "/predictions/images?count=true&heatmap=true"
        bytes_pred = io.BytesIO()
        array = (np.random.rand(*SIZE) * 256).astype('uint8')
        pil.fromarray(array).save(bytes_pred, format='JPEG')
        bytes_pred.seek(0)
        files = [
            ('file', ('img.jpg', bytes_pred, 'image/jpeg'))
        ]
        self.client.post(url, headers={}, data={}, files=files)


# class VideoUser(DroneUser):
#     weight = 1
#     @task
#     def get_video_prediction(self):
#         pass
