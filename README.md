# Drone-CrowdCounting [![codecov](https://codecov.io/gh/se4ai2122-cs-uniba/Drone-CrowdCounting/branch/main/graph/badge.svg?token=WFM4DH44WF)](https://codecov.io/gh/se4ai2122-cs-uniba/Drone-CrowdCounting) ![Pylint Report](https://github.com/se4ai2122-cs-uniba/Drone-CrowdCounting/actions/workflows/linting.yml/badge.svg)

*Counting system to estimate the number of people and density map in a drone-captured image*

# Usage
## API Endpoints
The API is accessible at the following endpoints:
- `/` which gives a welcome message
- `/docs` which provides a documentation of the API
- `/predictions/images` used to receive prediction for a given image
- `/predictions/videos` used to receive prediction for a given video

## Request of prediction
The request is made by passing:
- the image or video to be processed
- the parameters to select the output type

An example of request with `curl`:
```bash
curl -X 'POST' \
  'http://www.drone-crowdcounting.com/predictions/images?count=true&heatmap=true' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@<filename.format>;type=image/jpeg'
```

## Container
### Pull and run docker container
Our container is hosted on [dockerhub](https://hub.docker.com/r/sergiocaputoo/visdrone):

`docker pull sergiocaputoo/visdrone`

It is preferable to run the container with a GPU, but if you don't have one, you can run it with a CPU without problems.
#### Run on CPU
```docker run -it --name visdrone_cpu -p 8000:8000 sergiocaputoo/visdrone```
#### Run on GPU
```docker run -it --name visdrone_cuda -p 8000:8000 --gpus all sergiocaputoo/visdrone```

### Build and run docker container
#### Run on CPU
```/docker_scripts/api```
#### Run on GPU
```/docker_scripts/api_gpu```

## Run locally without docker
### Requirements
- python 3.9
- [ffmpeg](https://www.ffmpeg.org/)

If you just want to run the api:
```bash
pip install -r requirements.txt
```

### Run api
```bash
python src/api.py
```
Those commands will run the api, which will accept requests on port 8000.


## Monitoring
The backend is monitored by prometheus; a metric dashboard is exposed on grafana using prometheus as datasource.
Also an alert manager is defined by prometheus that alerts on slack if the backend goes down.

To run the monitoring suite:
```/docker_scripts/monitoring```

## Locust
```bash
locust -f tests/load_tests/locustfile.py --host http://localhost:8000
```

## PyTest
To run pytest without gpu:

```bash
$env:PYTHONPATH = "src"
pytest -m "not gpu" --cov src tests/
```

To run pytest with gpu:
```bash
$env:PYTHONPATH = "src"
pytest --cov src tests/
```

## Great Expectations
```bash
cd tests/great_expectations/checkpoints
great_expectations --v3-api checkpoint run complete_data.yml
done
```
## Frontend
Our simple material design frontend is developed with angular, and hosted on microsoft azure: 
[frontend link](http://drone-crowdcounting.com/)

![frontend](https://user-images.githubusercontent.com/38686676/148965188-d3564ec7-5a5d-4b6d-84fb-b28b08376baf.png)

The source code can be found on github:
[frontend repository](https://github.com/MauroCamporeale/dronecrowd-wa)
