"""
QDrant utility functions
"""

import requests


def is_qdrant_running(url: str) -> bool:
    """Check if QDrant is running at the given URL"""
    try:
        response = requests.get(url, timeout=2)
        return response.status_code == 200
    except requests.RequestException:
        return False


def wait_for_qdrant(url: str = "http://localhost:6333", timeout: int = 30) -> bool:
    """Wait for QDrant to be ready"""
    print("Waiting for QDrant to be ready...")

    for i in range(timeout):
        if is_qdrant_running(url):
            print("✓ QDrant is running and ready!")
            return True

        import time
        time.sleep(1)
        if i % 5 == 0 and i > 0:
            print(f"Still waiting... ({i}s)")

    print("✗ QDrant failed to start within timeout")
    return False


