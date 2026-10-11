import pytest
from fastapi.testclient import TestClient

from api import app


@pytest.fixture(scope="session")
def client():
    # `with` runs the FastAPI lifespan, so models are trained/loaded before tests.
    with TestClient(app) as c:
        yield c
