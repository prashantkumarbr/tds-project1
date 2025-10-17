# app.py
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import requests
import os
import base64
import time
import json
import logging
import re
import asyncio
import aiohttp
from openai import OpenAI  # keep as you had; ensure package is available

load_dotenv()
app = FastAPI()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

GITHUB_USER = os.getenv("GITHUB_USER")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
STUDENT_SECRET = os.getenv("STUDENT_SECRET") or ""
AIPIPE_TOKEN = os.getenv("AIPIPE_TOKEN")

if not GITHUB_USER or not GITHUB_TOKEN:
    logging.warning("GITHUB_USER or GITHUB_TOKEN not set; repo operations will fail until set.")

client = OpenAI(
    base_url="https://aipipe.org/openrouter/v1",
    api_key=AIPIPE_TOKEN
)


# ------------------ Helpers ------------------

async def validate_secret(secret: str) -> bool:
    return secret == STUDENT_SECRET


def required_fields_present(data: dict):
    required = ["email", "secret", "task", "round", "nonce", "brief", "evaluation_url"]
    missing = [f for f in required if f not in data]
    return missing


async def generate_app_files(brief: str, attachments: list[dict]) -> list[dict]:
    """
    Use GPT to generate app files dynamically based on the task brief.
    Returns a list of dicts: {name: str, content: bytes or str}
    """
    logging.info("Generating app files using LLM...")

    attachment_list = "\n".join([f"- {a.get('name','(unknown)')}" for a in attachments]) or "None"

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
    html_content = None
    try:
        # call LLM (ai-pipe style)
        llm_response = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        html_content = llm_response.choices[0].message.content
        logging.info("LLM generated HTML content successfully.")
    except Exception as e:
        logging.error(f"LLM generation failed: {e}")
        html_content = f"""<!DOCTYPE html>
            <html><head><meta charset="utf-8"><title>Fallback App</title></head><body>
            <h1>Fallback App</h1><p>Brief: {brief}</p></body></html>"""

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

    files = [
        {"name": "index.html", "content": html_content},  # string fine
        {"name": "README.md", "content": readme_content},
    ]

    # Add attachments; keep bytes for binary files
    for att in attachments:
        try:
            name = att.get("name", "attachment.bin")
            data_uri = att.get("url", "")
            if "," in data_uri:
                header, encoded = data_uri.split(",", 1)
                # header like: data:image/png;base64
                is_base64 = header.endswith(";base64") or "base64" in header
                if is_base64:
                    raw = base64.b64decode(encoded)
                else:
                    # assume percent-encoded data
                    raw = encoded.encode("utf-8")
                files.append({"name": name, "content": raw})
            else:
                logging.warning(f"Attachment {name} missing data URI comma.")
        except Exception as e:
            logging.warning(f"Failed to decode attachment {att.get('name')}: {e}")

    return files


def scan_and_redact(contents: bytes, sensitive_values: list):
    """
    Replace occurrences of any sensitive_values (strings) in contents (bytes or str)
    with b'[REDACTED]'. Return bytes.
    """
    if isinstance(contents, str):
        b = contents.encode("utf-8", errors="ignore")
    else:
        b = contents
    for s in sensitive_values:
        if not s:
            continue
        try:
            b = b.replace(s.encode('utf-8'), b"[REDACTED]")
        except Exception:
            pass
    return b


async def create_github_repo(repo_name: str) -> dict:
    logging.info(f"Creating GitHub repo: {repo_name}")
    payload = {"name": repo_name, "private": False, "auto_init": False}
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}
    resp = requests.post("https://api.github.com/user/repos", headers=headers, json=payload)
    if resp.status_code not in (201,):
        raise Exception(f"Repo creation failed: {resp.status_code}, {resp.text}")
    return resp.json()


async def repo_exists(repo_name: str) -> bool:
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}
    r = requests.get(f"https://api.github.com/repos/{GITHUB_USER}/{repo_name}", headers=headers)
    return r.status_code == 200


async def get_repo_info(repo_name: str) -> dict:
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}
    r = requests.get(f"https://api.github.com/repos/{GITHUB_USER}/{repo_name}", headers=headers)
    if r.status_code != 200:
        raise Exception(f"Failed to get repo info: {r.status_code}, {r.text}")
    return r.json()


async def push_file(repo_name: str, file: dict, branch: str = None):
    """
    Push a file via GitHub contents API. file['content'] can be bytes or str.
    If file exists, update using sha. Return response json.
    """
    name = file["name"]
    content = file["content"]
    if isinstance(content, str):
        raw_bytes = content.encode("utf-8")
    else:
        raw_bytes = content
    b64 = base64.b64encode(raw_bytes).decode("utf-8")

    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}
    # Check whether file exists to get current sha
    url = f"https://api.github.com/repos/{GITHUB_USER}/{repo_name}/contents/{name}"
    params = {}
    if branch:
        params["ref"] = branch
    r = requests.get(url, headers=headers, params=params)
    payload = {"message": f"Add/Update {name}", "content": b64}
    if r.status_code == 200:
        # update
        sha = r.json().get("sha")
        if sha:
            payload["sha"] = sha
    resp = requests.put(url, headers=headers, json=payload)
    if resp.status_code not in (200, 201):
        raise Exception(f"Push failed for {name}: {resp.status_code}, {resp.text}")
    return resp.json()


async def get_latest_commit_sha(repo_name: str, branch="main"):
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}
    resp = requests.get(f"https://api.github.com/repos/{GITHUB_USER}/{repo_name}/commits/{branch}", headers=headers)
    if resp.status_code != 200:
        raise Exception(f"Failed to get commit: {resp.status_code}, {resp.text}")
    return resp.json().get('sha')


async def enable_github_pages(repo_name: str, branch: str):
    logging.info("Enabling GitHub Pages...")

    url = f"https://api.github.com/repos/{GITHUB_USER}/{repo_name}/pages"
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
    }
    payload = {"source": {"branch": branch, "path": "/"}}

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=payload) as resp:
            text = await resp.text()
            if resp.status not in (201, 202):
                raise Exception(f"Pages enable failed: {resp.status}, {text}")

    # Construct Pages URL
    pages_url = f"https://{GITHUB_USER}.github.io/{repo_name}/"

    # Poll until live (up to 2 minutes)
    for _ in range(40):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(pages_url, timeout=5) as check:
                    if check.status == 200:
                        logging.info(f"GitHub Pages enabled at {pages_url}")
                        return pages_url
        except Exception:
            pass
        await asyncio.sleep(3)

    raise Exception("GitHub Pages URL not responding 200")



async def notify_evaluation(data: dict, repo_url: str, commit_sha: str, pages_url: str):
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
    for attempt in range(6):
        try:
            r = requests.post(data['evaluation_url'], headers=headers, json=payload, timeout=10)
            if r.status_code == 200:
                logging.info("Evaluation API notified successfully.")
                return True
            else:
                logging.warning(f"Evaluation notify returned {r.status_code}: {r.text}")
        except Exception as e:
            logging.warning(f"Retrying notify after error: {e}")
        await asyncio.sleep(delay)
        delay *= 2
    raise Exception("Failed to notify evaluation API after retries.")


async def add_license_file() -> dict:
    # Minimal MIT license template; you might want to expand authors/year
    mit_text = """MIT License

Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy
... (standard MIT license text) ...
"""
    return {"name": "LICENSE", "content": mit_text}


async def handle_round(data: dict):
    """
    Core worker: generate files, create or update repo, push files, enable pages, notify evaluation.
    This function is intended to run in background (async).
    """
    logging.info(f"Starting work for task={data.get('task')} round={data.get('round')} nonce={data.get('nonce')}")
    repo_name = f"{data['task']}_{data['nonce']}".replace(" ", "_")
    # Ensure repo name sanity
    repo_name = re.sub(r"[^A-Za-z0-9_.-]", "_", repo_name)[:100]

    attachments = data.get('attachments', [])
    files = await generate_app_files(data['brief'], attachments)
    # Add LICENSE
    license_file = await add_license_file()
    files.append(license_file)

    # Security: redact any env-secret occurrences inside generated files
    sensitive_values = [GITHUB_TOKEN, AIPIPE_TOKEN, STUDENT_SECRET]
    sanitized_files = []
    for f in files:
        name = f['name']
        content = f['content']
        redacted = scan_and_redact(content, sensitive_values)
        sanitized_files.append({"name": name, "content": redacted})

    # Create or use existing repo depending on round
    try:
        if data['round'] == 1:
            repo_info = await create_github_repo(repo_name)
        else:
            # round 2: use existing repo if present; otherwise create
            if await repo_exists(repo_name):
                repo_info = await get_repo_info(repo_name)
                logging.info(f"Using existing repo for round 2: {repo_name}")
            else:
                repo_info = await create_github_repo(repo_name)
        default_branch = repo_info.get("default_branch") or "main"
    except Exception as e:
        logging.error(f"Repo create/get failed: {e}")
        raise

    # Push files
    for f in sanitized_files:
        try:
            await push_file(repo_name, f, branch=default_branch)
            logging.info(f"Pushed file {f['name']}")
        except Exception as e:
            logging.error(f"Failed pushing {f['name']}: {e}")
            raise

    # Ensure README exists (we already added), and license already added
    # Enable GitHub Pages
    try:
        pages_url = await enable_github_pages(repo_name, default_branch)
    except Exception as e:
        logging.error(f"Failed enabling Pages: {e}")
        raise

    # Get latest commit sha on the branch
    try:
        commit_sha = await get_latest_commit_sha(repo_name, branch=default_branch)
    except Exception as e:
        logging.error(f"Failed to get commit sha: {e}")
        raise

    # Notify evaluation API (with exponential backoff inside)
    try:
        await notify_evaluation(data, f"https://github.com/{GITHUB_USER}/{repo_name}", commit_sha, pages_url)
    except Exception as e:
        logging.error(f"Failed to notify evaluation server: {e}")
        raise

    logging.info(f"Completed round {data['round']} for repo {repo_name}")
    return {
        "repo_url": f"https://github.com/{GITHUB_USER}/{repo_name}",
        "pages_url": pages_url,
        "commit_sha": commit_sha,
    }


# ------------------ FastAPI Endpoint ------------------

@app.post("/handle_tasks")
async def handle_tasks(request: Request):
    """
    Accepts the JSON request described in the project spec.
    Returns HTTP 200 immediately (ack) and processes the task in background.
    """
    data = await request.json()
    missing = required_fields_present(data)
    if missing:
        return JSONResponse({"status": "error", "message": f"Missing fields: {missing}"}, status_code=400)

    # Verify secret (must be exact)
    if not await validate_secret(data.get("secret", "")):
        return JSONResponse({"status": "error", "message": "Invalid secret"}, status_code=403)

    if data.get("round") not in (1, 2):
        return JSONResponse({"status": "error", "message": "Invalid round (must be 1 or 2)"}, status_code=400)

    # Send immediate ack 200 with minimal info
    try:
        # schedule background worker
        asyncio.create_task(
            background_worker(data)
        )
        return JSONResponse({"status": "accepted", "message": "Task received and will be processed."})
    except Exception as e:
        logging.error(f"Failed to schedule background worker: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


async def background_worker(data: dict):
    try:
        result = await handle_round(data)
        logging.info(f"Background task completed: {result}")
    except Exception as e:
        logging.error(f"Background task failed: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=6000)
