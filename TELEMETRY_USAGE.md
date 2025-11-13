# Telemetry Usage Guide

This document explains how to use the telemetry system to observe and verify the performance of the per-category reasoning and smart ensemble combiner.

## Overview

The telemetry system logs structured events in NDJSON format for:
- **Category reasoning results**: Track which categories return suggestions and with what confidence
- **Ensemble results**: Monitor backfill triggers, agreement bonuses, and final tag suggestions

## Configuration

Set the log file path via environment variable:
```bash
export TAGGER_LOG_PATH=/var/log/tagger.ndjson
```

Default: `/tmp/tagger.ndjson`

## Automatic Logging

Telemetry events are automatically logged when:
1. Each category reasoning completes (5 events per lecture)
2. Ensemble combiner produces final suggestions (1 event per lecture)

No code changes required - just use the API normally!

## Analyzing Logs

### Coverage & Performance Metrics

Run the analyzer to get system-wide statistics:

```bash
python analyze_logs.py /tmp/tagger.ndjson
```

**Output includes:**
- **Coverage**: % of lectures that received at least one tag
- **Backfill rate**: How often the empty-reasoning fallback triggered
- **Avg tags per lecture**: Mean, median, and max tag counts
- **Agreement rate**: Avg number of tags where reasoning and prototype agreed
- **Per-category non-empty rates**: Which categories contribute most often
- **Possible misses**: Cases where reasoning returned empty but prototype scored high

### Example Output

```
=== Tagger Telemetry Analysis ===
Lectures processed: 150
Coverage (final has ≥1 tag): 98.67% (148/150)
Backfill rate: 4.00% (6/150)
Avg final tags per lecture: 3.24 (median=3, max=7)
Avg agreement bonus count per lecture: 1.82

=== Per-Category Non-Empty Rates (Reasoning Step) ===
- Topic: 92.00% (138/150)
- Persona: 45.33% (68/150)
- Audience: 28.00% (42/150)
- Tone: 56.67% (85/150)
- Format: 34.00% (51/150)
```

## Human Review Sampling

Randomly sample results for manual quality checks:

```bash
python sample_human_review.py /tmp/tagger.ndjson 20
```

This will show:
- Lecture IDs
- Total tags suggested
- Agreement bonus count
- Backfill status
- Top 3 suggestions with detailed scores (combined, reasoning, prototype, threshold)

### Example Output

```
[1] Lecture 12345
    Total tags: 4 | Agreement bonus: 2 | Backfill: False
    Top 3 suggestions:
      - technology: combined=0.92 | reasoning=0.88 | prototype=0.82 (threshold=0.65)
      - innovation: combined=0.85 | reasoning=0.82 | prototype=0.75 (threshold=0.65)
      - leadership: combined=0.78 | reasoning=0.85 | prototype=0.55 (threshold=0.60)
```

## Use Cases

### 1. Verify Per-Category Reasoning
Check if categories are contributing as expected:
- Are some categories consistently empty?
- Is Topic category working for most lectures?

### 2. Monitor Backfill Effectiveness
Track edge cases:
- How often does backfill trigger?
- Are backfill suggestions reasonable?

### 3. Validate Agreement Bonus
Confirm the smart combiner is working:
- How often do reasoning and prototype agree?
- Are agreement bonuses helping quality?

### 4. Identify Quality Issues
Find problematic cases:
- Lectures with zero suggestions
- Mismatches between reasoning and prototype scores
- Categories that should fire but don't

## Log Rotation

Logs are append-only. To rotate:

```bash
# Archive current logs
mv /tmp/tagger.ndjson /var/log/archive/tagger-$(date +%Y%m%d).ndjson

# System will create new file automatically
```

## Performance Impact

**Zero blocking overhead:**
- Thread-safe async file writes
- No network calls
- Minimal CPU/memory usage
- Safe for production use

## Troubleshooting

**No logs appearing?**
- Check `TAGGER_LOG_PATH` environment variable
- Verify directory exists and is writable
- Confirm API is using ensemble or reasoning mode (not fast mode)

**Parser errors in analyzer?**
- Corrupted JSON lines will be skipped with warnings
- Check for disk space issues during logging

**Need more detail?**
- Individual AI call logs are in PostgreSQL `ai_calls` table
- Structured logs show request-level details
- Discord webhooks provide real-time alerts
