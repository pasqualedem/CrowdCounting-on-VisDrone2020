import base64
import os.path

import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import shutil
import json
from run import run_net

app = FastAPI()

@app.post("/predictions/images", responses={200: {"description": "returns the predicted number of people in the image and/or the heatmap"}})
async def predictFromImages(file: UploadFile = File(...), count: bool = True, heatmap: bool = True):

    with open(f'{file.filename}','wb') as buffer:
        shutil.copyfileobj(file.file, buffer)

    path = "prediction.png"

    if count and not heatmap:
        run_net(file.filename, ['count_callback'])
        os.remove(file.filename)
        with open('count_results.json') as f:
            data = json.load(f)
        return {"image_name": data['img_name'], "people_number": data['count']}

    if heatmap and not count:
        run_net(file.filename, ['save_callback'])
        os.remove(file.filename)
        return FileResponse(path, media_type="image/png")

    if heatmap and count:
        run_net(file.filename, ['count_callback', 'save_callback'])
        os.remove(file.filename)
        with open('count_results.json') as f:
            data = json.load(f)
        results = {"image_name": data['img_name'], "people_number": data['count']}
        return FileResponse(path, headers=results, media_type="image/png")

    if not count and not heatmap:
        return {"error": "why predict something and not wanting any result?"}


@app.post("/predictions/videos", responses={200: {"description": "returns the predicted number of people in the video and/or the heatmap"}})
async def predictFromVideos(file: UploadFile = File(...), count: bool = True, heatmap: bool = True):

    with open(f'{file.filename}','wb') as buffer:
        shutil.copyfileobj(file.file, buffer)

    path = "prediction.png"

    if count and not heatmap:
        run_net(file.filename, ['count_callback'])
        os.remove(file.filename)
        with open('count_results.json') as f:
            data = json.load(f)
        return {"image_name": data['img_name'], "people_number": data['count']}

    if heatmap and not count:
        run_net(file.filename, ['save_callback'])
        os.remove(file.filename)
        return FileResponse(path, media_type="image/png")

    if heatmap and count:
        run_net(file.filename, ['count_callback','save_callback'])
        os.remove(file.filename)
        with open('count_results.json') as f:
            data = json.load(f)
        results = {"image_name": data['img_name'], "people_number": data['count']}
        return FileResponse(path, headers=results, media_type="image/png")

    if not count and not heatmap:
        return {"error": "why predict something and not wanting any result?"}


if __name__ == '__main__':
    uvicorn.run("api:app")