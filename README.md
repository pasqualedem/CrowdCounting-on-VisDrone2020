# Drone-CrowdCounting [![codecov](https://codecov.io/gh/se4ai2122-cs-uniba/Drone-CrowdCounting/branch/main/graph/badge.svg?token=WFM4DH44WF)](https://codecov.io/gh/se4ai2122-cs-uniba/Drone-CrowdCounting) ![Pylint Report](https://github.com/se4ai2122-cs-uniba/Drone-CrowdCounting/actions/workflows/linting.yml/badge.svg)

*Counting people in drone images.*

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
  -F 'file=@00029.jpg;type=image/jpeg'
```

## Container
### Pull docker container
Our container is hosted on [dockerhub](https://hub.docker.com/r/sergiocaputoo/visdrone):

`docker pull sergiocaputoo/visdrone`

### Run container
It is preferable to run the container with a GPU, but if you don't have one, you can run it with a CPU without problems.
#### Run on CPU
```docker run -it --name visdrone_cpu -p 8000:8000 visdrone```
#### Run on GPU
```docker run -it --name visdrone_cuda -p 8000:8000 --gpus all visdrone```

## Run locally
### Requirements
If you just want to run the api:
```bash
pip install -r requirements.txt
```

### Run api
```bash
python src/api.py
```
Those commands will run the api, which will accept requests on port 8000.


## Prometheus
```bash
docker run -d -p 9090:9090 --add-host host.docker.internal:host-gateway \
    -v "$PWD/prometheus.yml":/etc/prometheus/prometheus.yml \
    --name=prometheus prom/prometheus
```

## Grafana
```bash
docker run -d -p 3000:3000 --add-host host.docker.internal:host-gateway \
    --name=grafana grafana/grafana-enterprise
```

## Locust
```bash
locust -f tests/locust.py --host http://localhost:8000
```

## PyTest
To run pytest without gpu:

```bash
PYTHONPATH=src pytest -m "not gpu" --cov src tests/
```

To run pytest with gpu:
```bash
PYTHONPATH=src pytest --cov src tests/
```

## Great Expectations
```bash
cd tests/great_expectations/checkpoints
great_expectations --v3-api checkpoint run complete_data.yml
done
```
