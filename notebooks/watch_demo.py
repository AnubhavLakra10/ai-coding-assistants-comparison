
# import hashlib
# import time
# from datetime import datetime
# from pathlib import Path

# import requests

# DEMO_PATH = Path("scripts/demo.py")
# API_URL = "http://127.0.0.1:8000/api/log-feedback"
# ASSISTANT_FLAG_FILE = Path(".assistant_flag")  # Tracks assistant source
# LAST_SNIPPET_CACHE = ""


# def get_file_hash():
#     return hashlib.md5(DEMO_PATH.read_bytes()).hexdigest()


# def extract_task_and_code(content):
#     lines = content.strip().splitlines()
#     task_line = next((line for line in lines if line.strip().startswith("#")), "Unknown Task")
#     task = task_line.lstrip("#").strip()
#     code_lines = [line for line in lines if not line.strip().startswith("#")]
#     code = "\n".join(code_lines).strip()
#     return task, code


# # __file__.parent.parent == project root (assuming watch_demo.py lives under notebooks/)
# ASSISTANT_FLAG_FILE = Path(__file__).parent.parent / ".assistant_flag"
# # ...
# def get_current_assistant():
#     if ASSISTANT_FLAG_FILE.exists():
#         return ASSISTANT_FLAG_FILE.read_text(encoding="utf-8").strip()
#     return "Unknown"


# def watch_demo():
#     global LAST_SNIPPET_CACHE
#     print("[Watcher] Watching demo.py for changes...")
#     last_hash = get_file_hash()

#     while True:
#         time.sleep(2)
#         current_hash = get_file_hash()

#         if current_hash != last_hash:
#             print("[Watcher] 🔍 Change detected. Logging feedback...")

#             # Record start time (just before processing)
#             start_dt = datetime.utcnow()

#             content = DEMO_PATH.read_text(encoding="utf-8", errors="replace")
#             task, code = extract_task_and_code(content)

#             # If the code snippet didn’t actually change, skip logging
#             if code == LAST_SNIPPET_CACHE:
#                 last_hash = current_hash
#                 continue

#             LAST_SNIPPET_CACHE = code
#             assistant = get_current_assistant()

#             # Record end time (after processing)
#             end_dt = datetime.utcnow()
#             duration_secs = (end_dt - start_dt).total_seconds()

#             payload = {
#                 "task": task,
#                 "assistant": assistant,
#                 "generated_code": code,
#                 "start_time": start_dt.isoformat(),
#                 "end_time": end_dt.isoformat(),
#                 "duration": duration_secs,          # New field
#                 "accuracy_rating": None             # Placeholder for manual rating
#             }

#             try:
#                 response = requests.post(API_URL, json=payload, timeout=5)
#                 if response.status_code == 200:
#                     print(f"[Watcher] ✅ Logged with assistant: {assistant} (duration: {duration_secs:.3f}s)")
#                 else:
#                     print(f"[Watcher] ❌ Failed: {response.text}")
#             except requests.exceptions.RequestException as err:
#                 print(f"[Watcher] ❌ Error: {err}")

#             last_hash = current_hash


# if __name__ == "__main__":
#     watch_demo()

# notebooks/watch_demo.py

import hashlib
import time
from datetime import datetime
from pathlib import Path

import requests

# ─── CONFIGURATION ─────────────────────────────────────────────────────────────

# Resolve project root from this file’s location:
PROJECT_ROOT = Path(__file__).parent.parent

# Path to the demo script we’re “watching”
DEMO_PATH = PROJECT_ROOT / "scripts" / "demo.py"

# The backend endpoint for logging feedback
API_URL = "http://127.0.0.1:8000/api/log-feedback"

# Path to the file which flags which assistant is active (e.g. "Copilot" or "Windsurf")
ASSISTANT_FLAG_FILE = PROJECT_ROOT / ".assistant_flag"

# Cache last‐seen code to avoid duplicate logs
LAST_SNIPPET_CACHE = ""


def get_file_hash() -> str:
    """
    Compute the MD5 hash of the contents of scripts/demo.py.
    This lets us detect “real” changes.
    """
    return hashlib.md5(DEMO_PATH.read_bytes()).hexdigest()


def extract_task_and_code(content: str):
    """
    Given the full text of demo.py, return:
      - task: the first line starting with '#'
      - code: everything else (all lines not starting with '#')
    """
    lines = content.strip().splitlines()
    # Find the first line beginning with '#', strip the '#' and whitespace
    task_line = next((line for line in lines if line.strip().startswith("#")), "Unknown Task")
    task = task_line.lstrip("#").strip()

    # Everything else (non-comment lines) is treated as the generated code
    code_lines = [line for line in lines if not line.strip().startswith("#")]
    code = "\n".join(code_lines).strip()

    return task, code


def get_current_assistant() -> str:
    """
    Read .assistant_flag at project root to see which assistant is active.
    If the file does not exist, return "Unknown".
    """
    if ASSISTANT_FLAG_FILE.exists():
        return ASSISTANT_FLAG_FILE.read_text(encoding="utf-8").strip()
    return "Unknown"


def watch_demo():
    """
    Polls scripts/demo.py every 2 seconds. When it detects a new MD5 hash,
    it extracts (task, code), measures latency, and POSTs a JSON payload to
    /api/log-feedback with:
      - task
      - assistant
      - generated_code
      - start_time (ISO)
      - end_time   (ISO)
      - duration   (float seconds)
      - accuracy_rating (null for now)
    """
    # Confirm demo.py exists
    if not DEMO_PATH.exists():
        print(f"[Watcher] ERROR: Cannot find {DEMO_PATH}. Please ensure scripts/demo.py exists.")
        return

    print(f"[Watcher] Watching {DEMO_PATH} for changes...")
    last_hash = get_file_hash()
    global LAST_SNIPPET_CACHE

    while True:
        time.sleep(2)
        current_hash = get_file_hash()

        if current_hash != last_hash:
            print("[Watcher] 🔍 Change detected. Logging feedback...")

            # Record start_time (UTC) immediately before reading file
            start_dt = datetime.utcnow()
            start_iso = start_dt.isoformat()

            # Read demo.py contents (robust to encoding issues)
            content = DEMO_PATH.read_text(encoding="utf-8", errors="replace")
            task, code = extract_task_and_code(content)

            # If code truly hasn’t changed since last time, skip logging
            if code == LAST_SNIPPET_CACHE:
                last_hash = current_hash
                continue

            LAST_SNIPPET_CACHE = code
            assistant = get_current_assistant()

            # Record end_time (UTC) after extracting task/code
            end_dt = datetime.utcnow()
            end_iso = end_dt.isoformat()

            # Compute duration in seconds (float)
            duration_secs = (end_dt - start_dt).total_seconds()

            # Build payload matching the backend’s schema for FeedbackEntry
            payload = {
                "task": task,
                "assistant": assistant,
                "generated_code": code,
                "start_time": start_iso,
                "end_time":   end_iso,
                "duration":   duration_secs,
                # accuracy_rating left as null (will be populated later via frontend)
                "accuracy_rating": None
            }

            try:
                response = requests.post(API_URL, json=payload, timeout=5)
                if response.status_code == 200:
                    print(f"[Watcher] ✅ Logged as {assistant} (duration: {duration_secs:.6f}s)")
                else:
                    print(f"[Watcher] ❌ Failed to log: {response.status_code} {response.text}")
            except requests.exceptions.RequestException as err:
                print(f"[Watcher] ❌ Error sending request: {err}")

            last_hash = current_hash


if __name__ == "__main__":
    watch_demo()
