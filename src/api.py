import base64
import os.path
from pathlib import Path
import cv2

import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
import shutil
import json
from run import run_net, load_CC_run
from run import run_net
import glob
import ffmpeg
from zipfile import ZipFile

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


@app.on_event("startup")
def _load_model():
    """
    Loads the model given in the config
    """
    model = load_CC_run()
    model.eval()
    print('Model correctly loaded')


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
                }
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
    })
async def predictFromImages(file: UploadFile = File(...), count: bool = True, heatmap: bool = True):
    with open(f'{file.filename}', 'wb') as buffer:
        shutil.copyfileobj(file.file, buffer)

    tmp = 'tmp/predictions'
    if not os.path.exists(tmp):
        os.makedirs(tmp)
    else:
        for f in os.listdir(tmp):
            # print(f)
            os.remove(os.path.join(tmp, f))

    if count and not heatmap:
        run_net(file.filename, ['count_callback'], model)
        os.remove(file.filename)
        with open('count_results.json') as f:
            data = json.load(f)
        os.remove('count_results.json')
        results = {"image_name": data[0]['img_name'], "people_number": data[0]['count']}
        return results

    if heatmap and not count:
        run_net(file.filename, ['save_callback'], model)
        path = os.path.join(tmp, Path(file.filename).stem + '.png')
        os.remove(file.filename)
        return FileResponse(path, media_type="image/png")

    if heatmap and count:
        run_net(file.filename, ['count_callback', 'save_callback'], model)
        path = os.path.join(tmp, Path(file.filename).stem + '.png')
        os.remove(file.filename)
        with open('count_results.json') as f:
            data = json.load(f)
        results = {"image_name": data[0]['img_name'], "people_number": data[0]['count']}
        os.remove('count_results.json')
        return FileResponse(path, headers=results, media_type="image/png")

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
                            {"image_name": "0", "people_number": "317.0"},
                            {"image_name": "1", "people_number": "303.0"},
                            {"image_name": "2", "people_number": "306.0"},
                            {"image_name": "3", "people_number": "306.0"},
                            {"image_name": "4", "people_number": "308.0"},
                            {"image_name": "5", "people_number": "311.0"},
                            {"image_name": "6", "people_number": "324.0"},
                            {"image_name": "7", "people_number": "316.0"},
                            {"image_name": "8", "people_number": "314.0"},
                            {"image_name": "9", "people_number": "305.0"},
                            {"image_name": "10", "people_number": "310.0"},
                            {"image_name": "11", "people_number": "307.0"},
                            {"image_name": "12", "people_number": "314.0"},
                            {"image_name": "13", "people_number": "304.0"},
                            {"image_name": "14", "people_number": "304.0"}
                        ]
                    }
                }
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
    })
async def predictFromVideos(file: UploadFile = File(...), count: bool = True, heatmap: bool = True):
    with open(f'{file.filename}', 'wb') as buffer:
        shutil.copyfileobj(file.file, buffer)

    tmp = 'tmp/predictions'
    if not os.path.exists(tmp):
        os.makedirs(tmp)
    else:
        for f in os.listdir(tmp):
            # print(f)
            os.remove(os.path.join(tmp, f))

    if count and not heatmap:
        run_net(file.filename, ['count_callback'], model)
        os.remove(file.filename)
        with open('count_results.json') as f:
            data = json.load(f)
        os.remove('count_results.json')
        results = []
        for i in range(len(data)):
            results.append({"video_frame": data[i]['img_name'], "people_number": data[i]['count']})
        return results

    if heatmap and not count:
        run_net(file.filename, ['save_callback'], model)
        file_name = Path(file.filename).stem + '_heatmap{}'.format(os.path.splitext(file.filename)[1])
        file_path = os.path.join(tmp, file_name)
        list_pred = os.path.join(tmp, 'list.txt')
        cap = cv2.VideoCapture(file.filename)
        os.remove(file.filename)

        # generate file list for heatmaps
        f = open(list_pred, "w")
        for i in range(len(os.listdir(tmp)) - 1):
            f.write('file {}.png \n'.format(i))
        f.close()

        fps = cap.get(cv2.CAP_PROP_FPS)

        # generate a video from the heatmaps obtained from each frame
        (
            ffmpeg
                .input(list_pred, r=fps, f='concat', safe='0')
                .output(file_path)
                .run()
        )
        return FileResponse(file_path, media_type="video/mp4", filename=file_name)

    if heatmap and count:
        run_net(file.filename, ['count_callback', 'save_callback'], model)
        file_name = Path(file.filename).stem + '_heatmap{}'.format(os.path.splitext(file.filename)[1])
        file_path = os.path.join(tmp, file_name)
        list_pred = os.path.join(tmp, 'list.txt')
        cap = cv2.VideoCapture(file.filename)
        os.remove(file.filename)

        # generate file list for heatmaps
        f = open(list_pred, "w")
        for i in range(len(os.listdir(tmp)) - 1):
            f.write('file {}.png \n'.format(i))
        f.close()

        fps = cap.get(cv2.CAP_PROP_FPS)

        # generate a video from the heatmaps obtained from each frame
        (
            ffmpeg
                .input(list_pred, r=fps, f='concat', safe='0')
                .output(file_path)
                .run()
        )

        count_file = '{}_people_count.json'.format(Path(file_path).stem)
        os.rename('count_results.json', count_file)

        zipfilename = Path(file_path).stem + '_results'
        zipfilepath = os.path.join(tmp, zipfilename)

        zipfile = ZipFile('{}.zip'.format(zipfilepath), 'w')
        root = os.getcwd()
        os.chdir(tmp)
        zipfile.write(file_name)
        os.chdir(root)
        zipfile.write(count_file)
        zipfile.close()
        zipfilename = zipfilename + '.zip'
        zipfilepath = zipfilepath + '.zip'
        os.remove(count_file)

        return FileResponse(zipfilepath, media_type="application/x-zip-compressed", filename=zipfilename)

    if not count and not heatmap:
        raise HTTPException(status_code=404, detail="Why predict something and not wanting any result?")


if __name__ == '__main__':
    uvicorn.run("api:app", host="0.0.0.0", reload=True, reload_dirs=['src'])
