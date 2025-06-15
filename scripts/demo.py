import hashlib
import time
from pathlib import Path

import requests

# Configuration
DEMO_PATH = Path("scripts/demo.py")
API_URL = "http://127.0.0.1:8000/api/log-feedback"
ASSISTANT_FLAG_FILE = Path(".assistant_flag")  # Must be in project root

LAST_SNIPPET_CACHE = ""

def get_file_hash():
    """Compute hash of demo.py"""
    return hashlib.md5(DEMO_PATH.read_bytes()).hexdigest()

def extract_task_and_code(content):
    """Extract first comment as task and remaining as code."""
    lines = content.strip().splitlines()
    task_line = next((line for line in lines if line.strip().startswith("#")), "Updated demo.py logic")
    task = task_line.lstrip("#").strip()
    code_lines = [line for line in lines if not line.strip().startswith("#")]
    code = "\n".join(code_lines).strip()
    return task, code

def get_current_assistant():
    if ASSISTANT_FLAG_FILE.exists():
        return ASSISTANT_FLAG_FILE.read_text(encoding="utf-8").strip()
    return "Unknown"



def watch_demo():
    """Polls demo.py every 2s, detects changes, logs to backend."""
    print("[Watcher] Watching demo.py for changes...")
    last_hash = get_file_hash()

    global LAST_SNIPPET_CACHE

    while True:
        time.sleep(2)
        current_hash = get_file_hash()

        if current_hash != last_hash:
            print("[Watcher] 🔍 Change detected. Reading new content...")
            # # Replace this
            # content = DEMO_PATH.read_text(encoding="utf-8")

            # With a safer fallback using errors='replace'
            content = DEMO_PATH.read_text(encoding="utf-8-sig", errors="replace")


            task, code = extract_task_and_code(content)

            if code == LAST_SNIPPET_CACHE:
                last_hash = current_hash
                continue

            LAST_SNIPPET_CACHE = code
            assistant = get_current_assistant()

            payload = {
                "task": task,
                "assistant": assistant,
                "generated_code": code
            }

            try:
                response = requests.post(API_URL, json=payload, timeout=5)
                if response.status_code == 200:
                    print(f"[Watcher] ✅ Logged to CSV as: {assistant}")
                else:
                    print(f"[Watcher] ❌ Failed to log: {response.status_code} {response.text}")
            except requests.exceptions.RequestException as err:
                print(f"[Watcher] ❌ Error sending request: {err}")

            last_hash = current_hash

if __name__ == "__main__":
    watch_demo()
























def reverse_words_if_not_palindrome(text: str) -> str:
    """
    Reverse each word unless it's a palindrome (case-insensitive).

    Args:
        text (str): The input string with words to be reversed.

    Returns:
        str: The modified string with words reversed unless they are palindromes.
    """
    def is_palindrome(word: str) -> bool:
        """
        Check if a word is a palindrome, case-insensitive.

        Args:
            word (str): The word to be checked.

        Returns:
            bool: True if the word is a palindrome, False otherwise.
        """
        alphanumeric_word = ''.join(filter(str.isalnum, word.lower()))
        return alphanumeric_word == alphanumeric_word[::-1]

    return ' '.join(
        word if is_palindrome(word) else word[::-1]
        for word in text.split()
    )




# -------------------------------------------
# Section 1: Reverse a string
# -------------------------------------------
def quick_sort(arr: list) -> list:
    """
    Sorts a list of numbers using the QuickSort algorithm with a for loop.

    Args:
        arr (list): The list of numbers to be sorted.

    Returns:
        list: The sorted list.
    """
    if len(arr) <= 1:
        return arr
    pivot = arr[0]
    left = [x for x in arr[1:] if x < pivot]
    right = [x for x in arr[1:] if x >= pivot]
    return quick_sort(left) + [pivot] + quick_sort(right)


# ----------------------------------------
# Section 2: Compute Factorial
# ----------------------------------------
def factorial(n):
    if n < 0:
        return None
    if n == 0 or n == 1:
        return 1
    return n * factorial(n - 1)


# ----------------------------------------
# Section 3: Check for Prime
# ----------------------------------------
def is_prime(num):
    if num < 2:
        return False
    for i in range(2, num):
        if num % i == 0:
            return False
    return True


# ----------------------------------------
# Section 4: Fetch JSON from a URL
# ----------------------------------------
import requests


def fetch_json(url):
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching JSON: {e}")
        return None


# ----------------------------------------
# Section 5: Sort a List of Numbers
# ----------------------------------------
def sort_numbers(nums):
    if len(nums) <= 1:
        return nums
    pivot = nums[0]
    less = [n for n in nums[1:] if n <= pivot]
    greater = [n for n in nums[1:] if n > pivot]
    return sort_numbers(less) + [pivot] + sort_numbers(greater)


# ----------------------------------------
# Section 6: Count Word Frequencies
# ----------------------------------------
def word_frequencies(text):
    words = text.split()
    frequencies = {}
    for word in words:
        word = word.lower().strip('.,!?";:')
        if word in frequencies:
            frequencies[word] += 1
        else:
            frequencies[word] = 1
    return frequencies



# ----------------------------------------
# Section 7: Fibonacci Sequence Generator
# ----------------------------------------
def fibonacci(n):
    if n <= 0:
        return []
    seq = [0]
    if n == 1:
        return seq
    seq.append(1)
    while len(seq) < n:
        seq.append(seq[-1] + seq[-2])
    return seq


# ----------------------------------------
# Section 8: Reverse Words in a Sentence
# ----------------------------------------
def create_palindrome_iterative(s):
    