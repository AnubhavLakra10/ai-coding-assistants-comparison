import requests

BASE_URL = "http://127.0.0.1:8000"
# API utility functions for interacting with the backend service
def submit_task(task: str, assistant: str):
    payload = {
        "task": task,
        "assistant": assistant
    }
    try:
        response = requests.post(f"{BASE_URL}/api/generate-code", json=payload)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        return {"error": str(e)}
