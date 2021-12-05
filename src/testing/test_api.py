import pytest
from fastapi import FastAPI
import requests
import json
import os
import uvicorn
from fastapi.testclient import TestClient
from http import HTTPStatus
from src.api import app

### API TESTING ###

# Initialize client
client = TestClient(app)

# @app.on_event("startup")
# def redirect():
#    return RedirectResponse(f"http://localhost:8000")

def test_apiroot():
    response = client.get("http://localhost:8000/")
    print(response)
    print(response.json())
    assert response.status_code == HTTPStatus.OK
    assert response.json()['data']['message'] == "Welcome to Drone-CrowdCounting! Please, read the `/docs`!"


if __name__ == '__main__':
    print(os.getcwd())
    os.chdir('C:/Users/Alessia/Documents/GitHub/Drone-CrowdCounting')
    test_apiroot()

