#!/usr/bin/env python3
"""
Quick sampler for human review of tagger results.

Randomly samples ensemble results to help identify edge cases
and verify that scoring logic is working as expected.
"""

import json
import random
import sys


def sample_cases(path, n=20):
    """Load and randomly sample ensemble results from telemetry logs."""
    ensemble_events = []
    
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if '"ensemble_result"' in line:
                try:
                    ensemble_events.append(json.loads(line.strip()))
                except Exception:
                    pass
    
    if not ensemble_events:
        print("No ensemble_result events found in logs.")
        return
    
    picks = random.sample(ensemble_events, min(n, len(ensemble_events)))
    
    print(f"=== Random Sample of {len(picks)} Ensemble Results ===\n")
    
    for i, ev in enumerate(picks, 1):
        lid = ev.get("lecture_id")
        backfill = ev.get("backfill_triggered", False)
        agreement_count = ev.get("agreement_count", 0)
        total = ev.get("total_returned", 0)
        
        print(f"[{i}] Lecture {lid}")
        print(f"    Total tags: {total} | Agreement bonus: {agreement_count} | Backfill: {backfill}")
        print(f"    Top 3 suggestions:")
        
        for t in ev.get("top3", []):
            label_id = t.get("label_id", "?")
            combined = t.get("combined", 0)
            r = t.get("r", 0)
            p = t.get("p", 0)
            pth = t.get("pth", 1.0)
            
            print(f"      - {label_id}: combined={combined:.2f} | "
                  f"reasoning={r:.2f} | prototype={p:.2f} (threshold={pth:.2f})")
        
        print()


if __name__ == "__main__":
    log_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/tagger.ndjson"
    sample_count = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    
    print(f"Sampling from: {log_path}")
    print(f"Sample size: {sample_count}\n")
    
    sample_cases(log_path, sample_count)
