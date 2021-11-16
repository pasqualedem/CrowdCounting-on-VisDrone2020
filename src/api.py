import base64
import os.path

import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
import shutil
import json
from run import run_net



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
    title = "Drone-CrowdCounting 👥🚁",
    description = description,
    version = "1.0.0",
)

        
@app.post(
    "/predictions/images", 
    summary = "Given a picture in input returns either a heatmap or people count or both",
    description = "Given a picture, you can choose to make the system generate a heatmap and/or people count using query parameters. <br><br> \
        ?count=true&heatmap=true => both generated <br> \
        ?count=true&heatmap=false => only count returned <br> \
        ?count=false&heatmap=true => only heatmap returned  ",
     responses= {
         200: {
            "description": "Returns the predicted number of people in the image and/or the heatmap",
            "content" : {
                "application/json" : {
                    "example" : {  
                        "image_name": "00002.jpg",
                         "people_number": "189.0"
                        
                    }
                 }
            }
        },
        404: {
            "description": "Error returned: API call without any parameter set on true ",
            "content" : {
                "application/json" : {
                    "example" : {  
                        "detail": "Why predict something and not wanting any result?"
                    }
                 }

            }
        }
    })
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
        raise HTTPException(status_code=404, detail="Why predict something and not wanting any result?")



@app.post(
    "/predictions/videos",
    summary = "Given a video in input returns either a heatmap or people count or both per frame",
    description = "Given a video, you can choose to make the system generate a heatmap and/or people count per frame using query parameters. <br><br> \
        ?count=true&heatmap=true => both generated <br> \
        ?count=true&heatmap=false => only count returned <br> \
        ?count=false&heatmap=true => only heatmap returned  ",
     responses= {
         200: {
            "description": "Returns the predicted number of people in the video and/or the heatmap per frame",
            "content" : {
                "application/json" : {
                    "example" : {  
                        "# to be defined "
                        
                    }
                 }
            }
        },
        404: {
            "description": "Error returned: API call without any parameter set on true ",
            "content" : {
                "application/json" : {
                    "example" : {  
                        "detail": "Why predict something and not wanting any result?"
                    }
                 }

            }
        }
    })
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
        raise HTTPException(status_code=404, detail="Why predict something and not wanting any result?")


if __name__ == '__main__':
    uvicorn.run("api:app", host="0.0.0.0")