from fastapi.testclient import TestClient
from http import HTTPStatus
from api import app

### API TESTING ###

# Initialize client
client = TestClient(app)


def test_apiroot():
    response = client.get("/")
    print(response)
    print(response.json())
    assert response.status_code == HTTPStatus.OK
    assert response.json()['data']['message'] == "Welcome to Drone-CrowdCounting! Please, read the `/docs`!"


