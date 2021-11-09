import os.path

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, StreamingResponse
import cv2
import io
import shutil
import json
from src.run import run_net

app = FastAPI()

@app.post("/predictions/num", responses={200: {"description": "returns the predicted number of people in the image", "content": {"json": {"example": {"image_name": "image.jpg", "people_number": "120"}}}}})
async def predictPeopleNumber(file: UploadFile = File(...)):
    with open(f'{file.filename}','wb') as buffer:
        shutil.copyfileobj(file.file, buffer)

    run_net(file.filename,['count_callback'])

    with open('count_results.json') as f:
        data = json.load(f)

    os.remove(file.filename)

    return {"image_name": data['img_name'], "peolpe_number": data['count']}

@app.post("/predictions/heatmap", responses={200: {"description": "returns the people heatmap", "content": {"image/png": {"example": "No example available."}}}})
async def predictHeatmap(file: UploadFile = File(...), download: bool = False):
    with open(f'{file.filename}','wb') as buffer:
        shutil.copyfileobj(file.file, buffer)

    run_net(file.filename,['save_callback'])

    os.remove(file.filename)

    path = "prediction.png"
    if os.path.exists(path):
        if download:
            return FileResponse(path, media_type="image/png", filename="heatmap.png")
        else:
            return FileResponse(path, media_type="image/png")
    return {"error":"File not found!" + str(path) }



    return {"image_name": data['img_name'], "peolpe_number": data['count']}

@app.post("/predictions", responses={200: {"description": "returns the people heatmap and the predicted people count in the headers", "content": {"image/png": {"example": "No example available"}}}})
async def predictAll(file: UploadFile = File(...)):
    with open(f'{file.filename}','wb') as buffer:
        shutil.copyfileobj(file.file, buffer)

    run_net(file.filename,['count_callback','save_callback'])

    os.remove(file.filename)

    with open('../count_results.json') as f:
        data = json.load(f)

    results = {"image_name": data['img_name'], "peolpe_number": data['count']}

    path = "prediction.png"
    if os.path.exists(path):
        return FileResponse(path, headers=results, media_type="image/png")
    return {"error": "File not found!" + str(path)}



