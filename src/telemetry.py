import json
import os
import threading
import time

_lock = threading.Lock()

LOG_PATH = os.getenv("TAGGER_LOG_PATH", "/tmp/tagger.ndjson")


def log_line(event: dict):
    """
    Thread-safe NDJSON logger for structured telemetry events.
    
    Appends a single JSON line to the log file with a timestamp.
    Each event is a dictionary that will be serialized to JSON.
    """
    event = dict(event)
    event["ts"] = int(time.time() * 1000)
    line = json.dumps(event, ensure_ascii=False)
    
    with _lock:
        # Ensure directory exists
        log_dir = os.path.dirname(LOG_PATH)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
