import pytest
from fastapi.testclient import TestClient
from serv2 import app, load_documents, preprocess_documents, initialize_embeddings, initialize_vector_store
import os
import asyncio

@pytest.fixture(scope="session", autouse=True)
async def initialize_service():
    """Initialize the service before running tests"""
    # Create a test document
    test_content = "Test medical content about diabetes.\nDiabetes is a metabolic disease that causes high blood sugar."
    with open("test_doc.txt", "w") as f:
        f.write(test_content)
    
    try:
        # Initialize the service components
        initialize_embeddings()
        documents = load_documents(["test_doc.txt"])
        chunks = preprocess_documents(documents)
        initialize_vector_store(documents, chunks)
        
        # Set the service as ready
        app.state.is_ready = True
        
        yield
    finally:
        # Clean up
        if os.path.exists("test_doc.txt"):
            os.remove("test_doc.txt")

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client

def test_load_documents():
    test_content = "Test medical content\nThis is a test document."
    with open("test_doc.txt", "w") as f:
        f.write(test_content)
    
    try:
        documents = load_documents(["test_doc.txt"])
        assert len(documents) == 1
        assert documents[0] == test_content
    finally:
        os.remove("test_doc.txt")

def test_preprocess_documents():
    test_docs = [
        "This is a long document " * 100,
        "This is a short document"
    ]
    chunks = preprocess_documents(test_docs)
    assert len(chunks) > 2
    assert isinstance(chunks, list)
    assert all(isinstance(chunk, str) for chunk in chunks)

def test_empty_question(client):
    # Test completely empty string
    response = client.post("/answer", json={"question": ""})
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()

    # Test string with only whitespace
    response = client.post("/answer", json={"question": "   "})
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()

    # Test with None (should be caught by Pydantic validation)
    response = client.post("/answer", json={"question": None})
    assert response.status_code == 422
def test_valid_question(client):
    response = client.post("/answer", json={"question": "What is diabetes?"})
    assert response.status_code == 200
    assert "answer" in response.json()

def test_invalid_request_format(client):
    response = client.post("/answer", json={"invalid_key": "question"})
    assert response.status_code == 422  # Validation error

def test_status_endpoint(client):
    response = client.get("/status")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"

if __name__ == "__main__":
    pytest.main(["-v"])