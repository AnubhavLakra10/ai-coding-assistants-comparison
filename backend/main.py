"""
Mock backend for AI Coding Assistant Comparison.
Simulates code generation for multiple assistants and logs results.
"""

import csv
import datetime
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()


#
# ─── PATHS & CSV HEADER MANAGEMENT ────────────────────────────────────────────
#

# We will store history.csv under ../data/history.csv relative to this file.
DATA_PATH = Path(__file__).parent.parent / "data" / "history.csv"


def ensure_csv_header():
    """
    Create history.csv (and its containing folder) if missing,
    or add new columns if the header is outdated.
    Required columns (in order):
      0: timestamp,
      1: task,
      2: assistant,
      3: generated_code,
      4: start_time,
      5: end_time,
      6: duration,
      7: accuracy_rating
    """
    # Ensure parent folder exists
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    # If file does not exist, create it with the full header
    if not DATA_PATH.exists():
        with open(DATA_PATH, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "task",
                "assistant",
                "generated_code",
                "start_time",
                "end_time",
                "duration",
                "accuracy_rating"
            ])
        return

    # If file exists, check if header is missing any new columns
    with open(DATA_PATH, mode="r+", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
        if not rows:
            # Empty file: write full header
            f.seek(0)
            f.truncate()
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "task",
                "assistant",
                "generated_code",
                "start_time",
                "end_time",
                "duration",
                "accuracy_rating"
            ])
            return

        existing_header = rows[0]
        required_header = [
            "timestamp",
            "task",
            "assistant",
            "generated_code",
            "start_time",
            "end_time",
            "duration",
            "accuracy_rating"
        ]

        if existing_header == required_header:
            # All good; nothing to do
            return

        # Otherwise, rewrite the header and re-append old rows
        # (Assume old rows had fewer columns; we'll pad them)
        new_rows = [required_header]
        for old_row in rows[1:]:
            # Pad or slice each existing row to length 8
            padded = old_row[:6]  # first 6 columns might exist
            # If the old row had fewer than 6, pad with empty strings
            while len(padded) < 6:
                padded.append("")
            # Append default "0.0" for duration and "" for rating
            padded.append("0.0")
            padded.append("")
            new_rows.append(padded)

        # Overwrite the CSV
        f.seek(0)
        f.truncate()
        writer = csv.writer(f)
        writer.writerows(new_rows)


# Ensure header is correct on startup
ensure_csv_header()


#
# ─── REQUEST / RESPONSE MODELS ─────────────────────────────────────────────────
#

class CodeRequest(BaseModel):
    """Defines the request schema for code generation."""
    task: str
    assistant: str


class FeedbackEntry(BaseModel):
    """Defines the schema for directly logging feedback."""
    task: str
    assistant: str
    generated_code: str
    start_time: Optional[str] = None     # ISO string
    end_time: Optional[str] = None       # ISO string
    duration: Optional[float] = None     # seconds elapsed
    accuracy_rating: Optional[int] = None  # 0–5, may be null


class RatingUpdate(BaseModel):
    """Schema for updating an existing row’s accuracy rating."""
    feedback_id: str         # this is the original timestamp string
    accuracy_rating: int     # 0–5


#
# ─── ENDPOINTS ─────────────────────────────────────────────────────────────────
#

@app.get("/")
async def root():
    """Basic root endpoint to verify the API is running."""
    return {"message": "AI Coding Assistant Comparison App is live!"}


@app.get("/api/assistants")
async def get_assistants():
    """Returns a mock list of AI assistants."""
    return {
        "assistants": [
            "MockGPT",
            "MockWindsurf",
            "MockCursor"
        ]
    }


@app.post("/api/generate-code")
async def generate_code(request: CodeRequest):
    """
    Simulates code generation by a selected assistant
    and logs the result to history.csv.
    We record:
      - timestamp = now (ISO)
      - task, assistant, generated_code
      - start_time = now ISO
      - end_time = now ISO (instantaneous for mock)
      - duration = 0.0
      - accuracy_rating = "" (empty)
    """
    # Build a fake code snippet based on the assistant
    if request.assistant.lower() == "mockwindsurf":
        code = (
            "def reverse_words(text):\n"
            "    # Split the input text into a list of individual words\n"
            "    words = text.split()\n"
            "    # Reverse the order of the words in the list\n"
            "    reversed_words = words[::-1]\n"
            "    # Join the reversed list of words back into a single string,\n"
            "    # using spaces as the separator\n"
            "    return ' '.join(reversed_words)"
        )
    else:
        code = (
            f"// Code for task: '{request.task}' by {request.assistant}\n"
            f"print('Hello from {request.assistant}')"
        )

    now_iso = datetime.datetime.now().isoformat(timespec="seconds")

    # Append one row with 8 columns:
    # [timestamp, task, assistant, generated_code, start_time, end_time, duration, accuracy_rating]
    try:
        with open(DATA_PATH, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([
                now_iso,
                request.task,
                request.assistant,
                code.replace("\n", "\\n"),
                now_iso,         # start_time
                now_iso,         # end_time
                "0.0",           # duration
                ""               # accuracy_rating (empty)
            ])
    except OSError as error:
        return {"error": f"Failed to write to log file: {str(error)}"}

    return {
        "task": request.task,
        "assistant": request.assistant,
        "generated_code": code
    }


@app.post("/api/log-feedback")
async def log_feedback(entry: FeedbackEntry):
    """
    Allows direct logging of assistant outputs to history.csv
    for manual runs from Copilot, Windsurf, Cursor, etc.
    Expects:
      - task, assistant, generated_code
      - Optional: start_time, end_time (ISO strings)
      - Optional: duration (float) and accuracy_rating (0–5)
    """
    timestamp = datetime.datetime.now().isoformat(timespec="seconds")

    # If duration is missing, default to 0.0; if accuracy_rating is missing, default to empty
    duration_str = f"{entry.duration:.6f}" if entry.duration is not None else "0.0"
    rating_str = "" if entry.accuracy_rating is None else str(entry.accuracy_rating)

    try:
        with open(DATA_PATH, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([
                timestamp,
                entry.task,
                entry.assistant,
                entry.generated_code.replace("\n", "\\n"),
                entry.start_time or "",
                entry.end_time or "",
                duration_str,
                rating_str
            ])
    except OSError as error:
        return {"error": f"Logging failed: {str(error)}"}

    return {"message": "Entry logged successfully"}


@app.patch("/api/update-rating")
async def update_rating(update: RatingUpdate):
    """
    Patches an existing row’s accuracy_rating in history.csv.
    We identify the row by matching its first column (timestamp) to feedback_id.
    """
    if not DATA_PATH.exists():
        raise HTTPException(status_code=404, detail="Log file not found")

    all_rows = []
    updated = False

    # Read entire CSV
    with open(DATA_PATH, mode="r", newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        header = next(reader, None)
        if header is None:
            raise HTTPException(status_code=500, detail="CSV header missing")
        all_rows.append(header)

        for row in reader:
            # Row format: [timestamp, task, assistant, code, start_time, end_time, duration, accuracy_rating]
            if row and row[0] == update.feedback_id:
                # Overwrite the 8th column (index 7) with the new rating
                row[7] = str(update.accuracy_rating)
                updated = True
            all_rows.append(row)

    if not updated:
        raise HTTPException(status_code=404, detail="Entry not found")

    # Write everything back
    with open(DATA_PATH, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerows(all_rows)

    return {"message": "Rating updated successfully"}
