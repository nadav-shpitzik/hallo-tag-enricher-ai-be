#!/usr/bin/env python3
"""
Offline analyzer for tagger telemetry logs.

Computes:
- Coverage: % of lectures with at least one final tag
- Backfill rate: How often the fallback mechanism triggers
- Average tags per lecture
- Agreement rate: How often reasoning and prototype agree
- Per-category non-empty rates from reasoning step
- Possible misses: Cases where reasoning missed but prototype scored high
"""

import json
import sys
import statistics as stats
from collections import Counter, defaultdict


def load_lines(path):
    """Generator to load NDJSON lines from a file."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except Exception as e:
                    print(f"Warning: Failed to parse line: {e}", file=sys.stderr)


def main(path):
    """Analyze tagger telemetry logs and print summary statistics."""
    per_lecture = defaultdict(lambda: {"cats": defaultdict(list), "final": None})
    backfill_count = 0
    agreements = []
    totals = []

    # Process all events
    for ev in load_lines(path):
        k = ev.get("kind")
        
        if k == "category_reasoning_result":
            lid = ev.get("lecture_id")
            cat = ev.get("category")
            per_lecture[lid]["cats"][cat] = ev.get("chosen_ids", [])
        
        elif k == "ensemble_result":
            lid = ev.get("lecture_id")
            per_lecture[lid]["final"] = ev
            backfill_count += 1 if ev.get("backfill_triggered") else 0
            agreements.append(ev.get("agreement_count", 0))
            totals.append(ev.get("total_returned", 0))

    # Calculate coverage: lecture has at least one final tag
    lectures = list(per_lecture.keys())
    if not lectures:
        print("No lectures found in logs.")
        return
    
    covered = sum(
        1 for lid in lectures 
        if (per_lecture[lid]["final"] and per_lecture[lid]["final"].get("total_returned", 0) > 0)
    )
    coverage = covered / max(1, len(lectures))

    # Per-category non-empty rate (from reasoning step)
    cat_non_empty = Counter()
    cat_total = Counter()
    for lid, data in per_lecture.items():
        for cat, ids in data["cats"].items():
            cat_total[cat] += 1
            if ids:
                cat_non_empty[cat] += 1

    # Print summary
    print("=== Tagger Telemetry Analysis ===")
    print(f"Lectures processed: {len(lectures)}")
    print(f"Coverage (final has ≥1 tag): {coverage:.2%} ({covered}/{len(lectures)})")
    print(f"Backfill rate: {backfill_count/max(1,len(lectures)):.2%} ({backfill_count}/{len(lectures)})")
    
    if totals:
        print(f"Avg final tags per lecture: {stats.mean(totals):.2f} "
              f"(median={stats.median(totals):.0f}, max={max(totals)})")
    
    if agreements:
        print(f"Avg agreement bonus count per lecture: {stats.mean(agreements):.2f}")

    print("\n=== Per-Category Non-Empty Rates (Reasoning Step) ===")
    for cat in sorted(cat_total.keys()):
        rate = cat_non_empty[cat] / max(1, cat_total[cat])
        print(f"- {cat}: {rate:.2%} ({cat_non_empty[cat]}/{cat_total[cat]})")

    # Flag likely misses: no Topic chosen by reasoning, but prototype had high score in top3
    print("\n=== Possible Misses (Topic Empty, Prototype Suggested) ===")
    miss_count = 0
    for lid, data in per_lecture.items():
        cats = data["cats"]
        final = data["final"] or {}
        top3 = final.get("top3") or []
        
        has_topic_reasoning = bool(cats.get("Topic") or cats.get("נושא"))
        
        if (not has_topic_reasoning) and any(t.get("p", 0) >= (t.get("pth", 1.0)) for t in top3):
            miss_count += 1
            if miss_count <= 10:  # Show first 10
                print(f"- Lecture {lid}: Topic empty in reasoning, prototype suggested strong tag")
    
    if miss_count > 10:
        print(f"... and {miss_count - 10} more")
    elif miss_count == 0:
        print("(none detected)")


if __name__ == "__main__":
    log_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/tagger.ndjson"
    print(f"Analyzing: {log_path}\n")
    main(log_path)
