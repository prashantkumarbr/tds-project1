"""Small helper to POST a task to a student endpoint for testing."""

import requests


def send_task(endpoint: str = 'http://localhost:6000/handle_tasks'):
    payload = {
        "email": "student@example.com",
        "secret": "97689C53F7241A88",
        "task": "captcha-solver-test31",
        "round": 2,
        "nonce": "ab12-12345",
        "brief": "Create a captcha solver that handles ?url=https://.../image.png. Default to attached sample.",
        "checks": [
            "Repo has MIT license",
            "README.md is professional",
            "Page displays captcha URL passed at ?url=...",
            "Page displays solved captcha text within 15 seconds",
        ],
        "evaluation_url": "http://localhost:9000/notify",
    # A minimal 1x1 transparent PNG as data URI
    "attachments": [{ "name": "sample.png", "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII=" }]
    }

    r = requests.post(endpoint, json=payload)
    try:
        print('Response:', r.status_code, r.json())
    except Exception:
        print('Non-JSON response:', r.status_code, r.text)


if __name__ == '__main__':
    send_task()
