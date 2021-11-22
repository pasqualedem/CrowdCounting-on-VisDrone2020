import json

import numpy as np
import matplotlib.pyplot as plt
import io
import os.path
import cv2
import uvicorn
import torch
import shutil
import uuid

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from starlette.background import BackgroundTasks
from http import HTTPStatus
from run import run_net, load_CC_run
from PIL import Image
from dataset.visdrone import cfg_data
from pathlib import Path

description = """Drone-CrowdCounting API allows you to deal with crowd air view pictures shot with drones

## Users

You will be able to:
* **Generate heatmap providing photo or video** 
* **Compute how many people are present in the photo or video frames** 

## Team

[Mauro Camporeale](https://github.com/MauroCamporeale) <br>
[Sergio Caputo](https://github.com/sergiocaputo) <br> 
[Pasquale De Marinis](https://github.com/pasqualedem) <br>
[Alessia Laforgia](https://github.com/AlessiaLa) <br>

"""

app = FastAPI(
    title="Drone-CrowdCounting 👥🚁",
    description=description,
    version="1.0.0",
)
model = None
img_queue = []
count_queue = []


def img_queue_callback(input, prediction, name):
    """

    """
    img_queue.append(prediction)


def count_queue_callback(input, prediction, name):
    """

    """
    count_queue.append(str(np.round(torch.sum(prediction.squeeze()).item() / cfg_data.LOG_PARA)))


@app.on_event("startup")
def _load_model():
    """
    Loads the model given in the config
    """
    model = load_CC_run()
    model.eval()

    print('Model correctly loaded on: ' + str(next(model.parameters()).device))


@app.on_event("shutdown")
def _del_tmp():
    shutil.rmtree('tmp')


@app.get("/", tags=["General"])  # path operation decorator
def _index(request: Request):
    """Root endpoint."""

    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {"message": "Welcome to Drone-CrowdCounting! Please, read the `/docs`!"},
    }
    return response


@app.post(
    "/predictions/images",
    summary="Given a picture in input returns either a heatmap or people count or both",
    description="Given a picture, you can choose to make the system generate a heatmap and/or people count using query parameters. <br><br> \
        ?count=true&heatmap=true => both generated <br> \
        ?count=true&heatmap=false => only count returned <br> \
        ?count=false&heatmap=true => only heatmap returned  ",
    responses={
        200: {
            "description": "Returns the predicted number of people in the image and/or the heatmap",
            "content": {
                "application/json": {
                    "example": {
                        "image_name": "00002.jpg",
                        "people_number": "189.0"

                    }
                },
                "image/png": {}
            }
        },
        404: {
            "description": "Error returned: API call without any parameter set on true ",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Why predict something and not wanting any result?"
                    }
                }

            }
        }
    },
    tags=["Prediction"])
async def predictFromImages(file: UploadFile = File(...), count: bool = True, heatmap: bool = True):
    img = get_array_img(file)
    name = file.filename
    if count and not heatmap:
        run_net(img, [count_queue_callback], model)
        return {'img_name': str(name), 'count': count_queue.pop(0)}

    if heatmap and not count:
        run_net(img, [img_queue_callback], model)
        prediction = img_queue.pop(0)
        return StreamingResponse(get_bytes_img(prediction), media_type="image/png")

    if heatmap and count:
        run_net(img, [img_queue_callback, count_queue_callback], model)
        results = {'img_name': str(name), 'count': count_queue.pop(0)}
        prediction = img_queue.pop(0)
        return StreamingResponse(get_bytes_img(prediction), headers=results, media_type="image/png")

    if not count and not heatmap:
        raise HTTPException(status_code=404, detail="Why predict something and not wanting any result?")


@app.post(
    "/predictions/videos",
    summary="Given a video in input returns either a heatmap or people count or both per frame",
    description="Given a video, you can choose to make the system generate a heatmap and/or people count per frame using query parameters. <br><br> \
        ?count=true&heatmap=true => both generated <br> \
        ?count=true&heatmap=false => only count returned <br> \
        ?count=false&heatmap=true => only heatmap returned  ",
    responses={
        200: {
            "description": "Returns the predicted number of people in the video and/or the heatmap per frame",
            "content": {
                "application/json": {
                    "example": {
                        "detail": [
                            {"video_frame": "0", "people_number": "317.0"},
                            {"video_frame": "1", "people_number": "303.0"},
                            {"video_frame": "2", "people_number": "306.0"},
                            {"video_frame": "3", "people_number": "306.0"},
                            {"video_frame": "4", "people_number": "308.0"},
                            {"video_frame": "5", "people_number": "311.0"},
                            {"video_frame": "6", "people_number": "324.0"},
                            {"video_frame": "7", "people_number": "316.0"},
                            {"video_frame": "8", "people_number": "314.0"},
                            {"video_frame": "9", "people_number": "305.0"},
                            {"video_frame": "10", "people_number": "310.0"},
                            {"video_frame": "11", "people_number": "307.0"},
                            {"video_frame": "12", "people_number": "314.0"},
                            {"video_frame": "13", "people_number": "304.0"},
                            {"video_frame": "14", "people_number": "304.0"}
                        ]
                    }
                },
                "video/mp4": {}
            }
        },
        404: {
            "description": "Error returned: API call without any parameter set on true ",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Why predict something and not wanting any result?"
                    }
                }
            }
        }
    },
    tags=["Prediction"])
async def predictFromVideos(background_tasks: BackgroundTasks, file: UploadFile = File(...), count: bool = True, heatmap: bool = True):
    tmp = os.path.join('tmp', uuid.uuid4().hex)
    tmp_heats = os.path.join(tmp, 'heatmaps')
    os.makedirs('tmp', exist_ok=True)
    os.makedirs(tmp, exist_ok=True)
    os.makedirs(tmp_heats, exist_ok=True)
    tmp_filename = os.path.join(tmp, file.filename)
    background_tasks.add_task(delete_files, tmp)

    with open(f'{tmp_filename}', 'wb') as buffer:
        shutil.copyfileobj(file.file, buffer)

    def tmp_save_callback(input, prediction, name):
        path = os.path.join(tmp_heats, Path(name).stem) + '.png'
        plt.imsave(path, prediction.squeeze(), cmap='jet')

    if count and not heatmap:
        run_net(tmp_filename, [count_queue_callback], model)
        response = [{"video_frame": str(i), "people_number": count_queue.pop(0)} for i in range(len(count_queue))]

    if heatmap and not count:
        run_net(tmp_filename, [tmp_save_callback], model)
        heat_path, heat_filename = make_video(tmp, file.filename, tmp_filename, tmp_heats)
        response = FileResponse(heat_path, media_type="video/mp4", filename=heat_filename)

    if heatmap and count:
        run_net(tmp_filename, [count_queue_callback, tmp_save_callback], model)
        heat_path, heat_filename = make_video(tmp, file.filename, tmp_filename, tmp_heats)
        counts = {str(i): count_queue.pop(0) for i in range(len(count_queue))}
        count_filename = 'count_results.json'
        count_file = os.path.join(tmp, count_filename)
        with open(count_file, mode='w') as fp:
            fp.write(json.dumps(counts, indent=2))

        response = FileResponse(heat_path, headers=counts, media_type="video/mp4", filename=heat_filename)

    if not count and not heatmap:
        raise HTTPException(status_code=404, detail="Why predict something and not wanting any result?")

    return response


def delete_files(path: str) -> None:
    shutil.rmtree(path)


def make_video(tmp, filename, tmp_filename, tmp_heats):
    file_name = Path(tmp_filename).stem + '_heatmap{}'.format(os.path.splitext(filename)[1])
    file_path = os.path.join(tmp, file_name)
    cap = cv2.VideoCapture(tmp_filename)

    # generate file list for heatmaps
    frames = os.listdir(tmp_heats)
    frames.sort(key=lambda fname: int(fname.split('.')[0]))
    frames = [os.path.join(tmp_heats, frame) for frame in frames]
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # generate a video from the heatmaps obtained from each frame
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(file_path, fourcc, fps, (width, height))
    for frame in frames:
        video_writer.write(cv2.imread(frame))
    # cv2.destroyAllWindows()
    video_writer.release()
    return file_path, file_name


def get_array_img(file: UploadFile = File(...)):
    contents = file.file.read()
    return np.array(Image.open(io.BytesIO(contents)))


def get_bytes_img(array):
    bytes_pred = io.BytesIO()
    plt.imsave(bytes_pred, array.squeeze(), cmap='jet', format='png')
    bytes_pred.seek(0)
    return bytes_pred


if __name__ == '__main__':
    uvicorn.run("api:app", host="0.0.0.0", reload=True, reload_dirs=['src'])
