from fastapi import FastAPI
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import requests
import os
import base64
import time
import json
from openai import OpenAI

import logging

# ------------------ Setup ------------------
load_dotenv()
app = FastAPI()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

GITHUB_USER = os.getenv("GITHUB_USER")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
STUDENT_SECRET = os.getenv("STUDENT_SECRET")
AIPIPE_TOKEN = os.getenv("AIPIPE_TOKEN")

client = OpenAI(
    base_url="https://aipipe.org/openai/v1",
    api_key=AIPIPE_TOKEN

)


# ------------------ Helpers ------------------

def validate_secret(secret: str) -> bool:
    return secret == STUDENT_SECRET


async def generate_app_files(brief: str, attachments: list[dict]) -> list[dict]:
    """
    Use GPT to generate app files dynamically based on the task brief.
    """
    logging.info("Generating app files using LLM...")

    # Prepare attachments text summary
    attachment_list = "\n".join([f"- {a['name']}" for a in attachments]) or "None"

    # Prompt for GPT
    prompt = f"""
You are a professional code generator. 
Create a minimal self-contained HTML+JS web app that satisfies this brief:

---
{brief}
---

If ?url= parameter is mentioned, handle it via JavaScript (fetch and display).
If there are attachments, load them locally as default fallback content.

The app must:
- Run fully in the browser (no backend calls)
- Be suitable for GitHub Pages hosting
- Include HTML, CSS, and JavaScript (in one file)
- Use plain JS (and Bootstrap if UI is expected)
- Render output clearly in the page.

Respond ONLY with the complete HTML document content (no explanations).
"""

    try:
        # New API style
        llm_response = await client.responses.create(
            model="openai/gpt-4.1-nano",  # use AI Pipe compatible model name
            messages=[
                {"role": "system", "content": "You generate clean, working code for web apps."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
        )
        html_content = llm_response.choices[0].message.content
        logging.info("LLM generated HTML content successfully.")
    except Exception as e:
        logging.error(f"LLM generation failed: {e}")
        html_content = f"""<!DOCTYPE html>
            <html><body><h1>Fallback App</h1><p>Brief: {brief}</p></body></html>"""

    # Create README.md with proper sections
    readme_content = f"""# LLM Generated App

This app was automatically generated based on the following brief:

{brief}

## Summary
This app fulfills the brief using a minimal web interface. It runs on GitHub Pages with client-side JS.

## Files
- index.html
{attachment_list}

## Usage
Open the deployed GitHub Pages URL in a browser.
If the app supports `?url=` parameter, pass your URL like:
`?url=https://example.com/data.json`

## Code Explanation
All app logic is inside `index.html` (inline JS).

## License
MIT License
"""

    # Create the base files list
    files = [
        {"name": "index.html", "content": html_content},
        {"name": "README.md", "content": readme_content},
    ]

    # Add attachments as separate files (decoded)
    for att in attachments:
        try:
            header, encoded = att['url'].split(",", 1)
            content = base64.b64decode(encoded).decode("utf-8", errors="ignore")
            files.append({"name": att['name'], "content": content})
        except Exception as e:
            logging.warning(f"Failed to decode attachment {att['name']}: {e}")

    return files


def create_github_repo(repo_name: str) -> dict:
    logging.info(f"Creating GitHub repo: {repo_name}")
    payload = {"name": repo_name, "private": False, "license_template": "mit"}
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}
    resp = requests.post("https://api.github.com/user/repos", headers=headers, json=payload)
    if resp.status_code != 201:
        raise Exception(f"Repo creation failed: {resp.status_code}, {resp.text}")
    return resp.json()


def push_file(repo_name: str, file: dict):
    logging.info(f"Pushing file: {file['name']}")
    content = file['content']
    if isinstance(content, bytes):
        content = base64.b64encode(content).decode('utf-8')
    else:
        content = base64.b64encode(content.encode('utf-8')).decode('utf-8')
    payload = {"message": f"Add {file['name']}", "content": content}
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}
    url = f"https://api.github.com/repos/{GITHUB_USER}/{repo_name}/contents/{file['name']}"
    resp = requests.put(url, headers=headers, json=payload)
    if resp.status_code not in (200, 201):
        raise Exception(f"Push failed for {file['name']}: {resp.status_code}, {resp.text}")
    return resp.json()


def get_latest_commit_sha(repo_name: str, branch="main"):
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}
    resp = requests.get(f"https://api.github.com/repos/{GITHUB_USER}/{repo_name}/commits/{branch}", headers=headers)
    if resp.status_code != 200:
        raise Exception(f"Failed to get commit: {resp.status_code}, {resp.text}")
    return resp.json()['sha']


def enable_github_pages(repo_name: str):
    logging.info("Enabling GitHub Pages...")
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}
    payload = {"source": {"branch": "main", "path": "/"}}
    resp = requests.post(f"https://api.github.com/repos/{GITHUB_USER}/{repo_name}/pages", headers=headers, json=payload)
    if resp.status_code not in (201, 204):
        raise Exception(f"Pages enable failed: {resp.status_code}, {resp.text}")
    pages_url = f"https://{GITHUB_USER}.github.io/{repo_name}/"
    for _ in range(30):  # wait up to ~90 seconds
        try:
            r = requests.get(pages_url)
            if r.status_code == 200:
                return pages_url
        except:
            pass
        time.sleep(3)
    raise Exception("GitHub Pages URL not responding 200")


def notify_evaluation(data: dict, repo_url: str, commit_sha: str, pages_url: str):
    logging.info("Notifying evaluation API...")
    payload = {
        "email": data['email'],
        "task": data['task'],
        "round": data['round'],
        "nonce": data['nonce'],
        "repo_url": repo_url,
        "commit_sha": commit_sha,
        "pages_url": pages_url,
    }
    headers = {"Content-Type": "application/json"}
    delay = 1
    for _ in range(6):
        try:
            r = requests.post(data['evaluation_url'], headers=headers, json=payload)
            if r.status_code == 200:
                logging.info("Evaluation API notified successfully.")
                return True
        except Exception as e:
            logging.warning(f"Retrying notify after error: {e}")
        time.sleep(delay)
        delay *= 2
    raise Exception("Failed to notify evaluation API after retries.")


async def handle_round(data: dict):
    repo_name = f"{data['task']}_{data['nonce']}"
    files = await generate_app_files(data['brief'], data.get('attachments', []))
    create_github_repo(repo_name)
    for f in files:
        push_file(repo_name, f)
    pages_url = enable_github_pages(repo_name)
    commit_sha = get_latest_commit_sha(repo_name)
    notify_evaluation(data, f"https://github.com/{GITHUB_USER}/{repo_name}", commit_sha, pages_url)
    return {
        "repo_url": f"https://github.com/{GITHUB_USER}/{repo_name}",
        "pages_url": pages_url,
        "commit_sha": commit_sha,
    }

# ------------------ FastAPI Endpoint ------------------

@app.post("/handle_tasks")
async def handle_tasks(data: dict):
    if not validate_secret(data.get("secret", "")):
        return JSONResponse({"status": "error", "message": "Invalid secret"}, status_code=403)
    if data.get("round") not in (1, 2):
        return JSONResponse({"status": "error", "message": "Invalid round"}, status_code=400)

    try:
        result = await handle_round(data)
        return JSONResponse({"status": f"round {data['round']} completed", **result})
    except Exception as e:
        logging.error(f"Error during round handling: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=6000)
