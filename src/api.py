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
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from starlette.background import BackgroundTasks
from http import HTTPStatus
from run import run_net, load_CC_run
from PIL import Image
from dataset.visdrone import cfg_data
from pathlib import Path
from utils import info_print

from monitoring import initialize_instrumentator

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


origins = [
    "http://localhost:4200",
    "http://localhost:3000",
    "http://drone-crowdcounting.azurewebsites.net/"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    expose_headers=["*"],
    allow_headers=origins
)

model = None
img_queue = []
count_queue = []


def img_queue_callback(input, prediction, name):
    """
    Adds prediction heatmap to img_queue queue
    """
    prediction = prediction.to('cpu')
    img_queue.append(prediction)


def count_queue_callback(input, prediction, name):
    """
    Adds prediction count to count_queue queue
    """
    count_queue.append(str(np.round(torch.sum(prediction.squeeze()).item() / cfg_data.LOG_PARA)))


@app.on_event("startup")
def _load_model():
    """
    Loads the model given in the config
    """
    global model
    model = load_CC_run()
    model.eval()

    info_print('Model correctly loaded on: ' + str(next(model.parameters()).device))


@app.on_event("startup")
async def _load_instrumentator():
    """
    Loads the instrumentator
    """
    instrumentator = initialize_instrumentator()
    instrumentator.instrument(app).expose(app, include_in_schema=False, should_gzip=True)

    info_print('Instrumentator correctly initialized')


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
                        "img_name": "00002.jpg",
                        "count": "189.0"

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
        result = {'img_name': str(name), 'count': count_queue.pop(0)}
        return JSONResponse(result, headers=result)

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
                            {"video_frame": "0", "count": "317.0"},
                            {"video_frame": "1", "count": "303.0"},
                            {"video_frame": "2", "count": "306.0"},
                            {"video_frame": "3", "count": "306.0"},
                            {"video_frame": "4", "count": "308.0"},
                            {"video_frame": "5", "count": "311.0"},
                            {"video_frame": "6", "count": "324.0"},
                            {"video_frame": "7", "count": "316.0"},
                            {"video_frame": "8", "count": "314.0"},
                            {"video_frame": "9", "count": "305.0"},
                            {"video_frame": "10", "count": "310.0"},
                            {"video_frame": "11", "count": "307.0"},
                            {"video_frame": "12", "count": "314.0"},
                            {"video_frame": "13", "count": "304.0"},
                            {"video_frame": "14", "count": "304.0"}
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
    frames_folder = save_video(tmp_filename, tmp)

    def tmp_save_callback(input, prediction, name):
        path = os.path.join(tmp_heats, Path(name).stem) + '.png'
        plt.imsave(path, prediction.squeeze(), cmap='jet')

    if count and not heatmap:
        run_net(frames_folder, [count_queue_callback], model)
        body = [{"video_frame": str(i), "count": count_queue.pop(0)} for i in range(len(count_queue))]
        response = JSONResponse(body, headers={'n_frames': str(len(body))})

    if heatmap and not count:
        run_net(frames_folder, [tmp_save_callback], model)
        heat_path, heat_filename = make_video(tmp, file.filename, tmp_filename, tmp_heats)
        response = FileResponse(heat_path, media_type="video/mp4", filename=heat_filename)

    if heatmap and count:
        run_net(frames_folder, [count_queue_callback, tmp_save_callback], model)
        heat_path, heat_filename = make_video(tmp, file.filename, tmp_filename, tmp_heats)
        counts = {str(i): count_queue.pop(0) for i in range(len(count_queue))}
        counts['n_frames'] = str(len(counts))
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


def save_video(file, folder):
    folder = os.path.join(folder, 'video_folder')
    os.makedirs(folder, exist_ok=True)
    video = cv2.VideoCapture(file)
    digit_len = len(str(int(video.get(cv2.CAP_PROP_FRAME_COUNT))))
    frame_count = 0
    while True:
        ret, data = video.read()
        if not ret:
            break
        data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        Image.fromarray(data).save(
            os.path.join(folder, str(frame_count).zfill(digit_len) + '.jpg'))
        frame_count += 1
    return folder


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
    fourcc = cv2.VideoWriter_fourcc(*'H264')
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
