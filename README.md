# AI Tag Enrichment — Hebrew Lecture Tagging Batch System

This system analyzes Hebrew lecture content (title + description) and suggests relevant tag IDs using embeddings, prototype learning, and optional LLM arbitration.

## Overview

**What it does:**
- Reads Hebrew lectures from PostgreSQL (`enriched_lectures` table)
- Generates semantic embeddings using OpenAI's `text-embedding-3-large`
- Learns from your existing tagged lectures via prototype centroids
- Scores new lectures against tag prototypes with per-tag calibrated thresholds
- Optionally uses an LLM arbiter (with Structured Outputs) for borderline cases
- Outputs suggestions to CSV and optionally to a `lecture_tag_suggestions` table

**What it doesn't do (Phase 1):**
- No writes to your source `enriched_lectures` table (read-only)
- No Airtable integration (planned for Phase 2 approval UI)

## Requirements

- Python 3.11+
- PostgreSQL database with `enriched_lectures` table
- OpenAI API key (for embeddings + optional LLM arbiter)
- Tags CSV file with columns: `tag_id`, `name_he`, `synonyms_he` (optional)

## Installation

1. **Install dependencies:**
   ```bash
   # Dependencies are already installed via uv
   # See pyproject.toml for the full list
   ```

2. **Configure environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

3. **Required environment variables:**
   - `DATABASE_URL`: PostgreSQL connection string (use read-only user recommended)
   - `OPENAI_API_KEY`: Your OpenAI API key
   - `TAGS_CSV_PATH`: Path to your tags CSV file
   - `OUTPUT_CSV_PATH`: Where to save the output CSV

4. **Prepare your tags CSV:**
   ```csv
   tag_id,name_he,synonyms_he
   1,חדשנות,"יצירתיות, המצאה"
   2,מנהיגות,"הנהגה, ניהול"
   ```
   See `data/tags_example.csv` for a full example.

## Usage

Run the batch script:

```bash
python src/main.py
```

The script will:
1. Load tags from your CSV
2. Fetch active lectures from the database
3. Generate embeddings for lectures and tags
4. Build prototype centroids from existing tagged lectures
5. Calibrate per-tag thresholds for high precision
6. Score all lectures and generate suggestions
7. Save results to CSV (and optionally to database)
8. Generate a QA report

## Output

### CSV Output
File: `output/tag_suggestions.csv`

Columns:
- `lecture_id`: Internal lecture ID
- `lecture_external_id`: External reference ID
- `tag_id`: Suggested tag ID
- `tag_name_he`: Hebrew tag name
- `score`: Confidence score (0-1)
- `rationale`: Hebrew explanation of why this tag was suggested
- `model`: Model/method used (e.g., "prototype", "prototype+llm:gpt-4o-mini")

### Database Output (Optional)
If `WRITE_TO_DB=true`, creates and populates:

```sql
CREATE TABLE lecture_tag_suggestions (
    suggestion_id       BIGSERIAL PRIMARY KEY,
    lecture_id          BIGINT NOT NULL,
    lecture_external_id VARCHAR NOT NULL,
    tag_id              VARCHAR NOT NULL,
    tag_name_he         VARCHAR NOT NULL,
    score               NUMERIC(5,4) NOT NULL,
    rationale           TEXT,
    sources             JSONB DEFAULT '["title","description"]',
    model               TEXT NOT NULL,
    status              VARCHAR NOT NULL DEFAULT 'pending',
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (lecture_id, tag_id)
);
```

## Configuration

Key parameters in `src/config.py`:

**Thresholds:**
- `min_confidence_threshold`: 0.65 (minimum score to suggest a tag)
- `high_confidence_threshold`: 0.80 (high confidence, skip LLM)
- `llm_borderline_lower/upper`: 0.55-0.80 (range for LLM arbitration)

**Tag Selection:**
- `top_k_tags`: 7 (maximum tags per lecture)
- `min_k_tags`: 3 (minimum tags if available)

**Prototype Learning:**
- `low_data_tag_threshold`: 5 (tags with <5 examples get label blending)
- `prototype_weight`: 0.8 (weight for prototype similarity)
- `label_weight`: 0.2 (weight for tag label similarity)

**Models:**
- `embedding_model`: text-embedding-3-large (3072 dimensions)
- `llm_model`: gpt-4o-mini (for arbitration)

## Architecture

### Core Components

1. **`config.py`**: Configuration management from environment variables
2. **`database.py`**: PostgreSQL connection and queries
3. **`tags_loader.py`**: CSV loading and tag label text generation
4. **`embeddings.py`**: OpenAI embeddings generation with batching
5. **`prototype_knn.py`**: Prototype learning and threshold calibration
6. **`llm_arbiter.py`**: LLM-based refinement with Structured Outputs
7. **`scorer.py`**: Main scoring engine and suggestion generation
8. **`output.py`**: CSV/DB output and QA reporting
9. **`main.py`**: Orchestration and execution flow

### Method

1. **Embeddings**: Text is embedded using `text-embedding-3-large`
   - Lectures: `[כותרת] {title}\n[תיאור] {description}`
   - Tags: `תגית: {name_he} | נרדפות: {synonyms_he}`

2. **Prototype Learning**: For each tag, compute centroid from existing tagged lectures
   - Low-data tags (<5 examples) blend prototype + label similarity (80/20)

3. **Threshold Calibration**: Per-tag thresholds fit on holdout set to target 90% precision

4. **LLM Arbiter** (optional): For borderline scores (0.55-0.80), LLM validates using Structured Outputs
   - Returns only valid tag IDs (no free text)
   - High confidence tags (≥0.80) skip LLM to save cost

5. **Output**: Top 3-7 tags per lecture with score ≥0.65

## Database Schema Requirements

Your `enriched_lectures` table must have:

```sql
CREATE TABLE enriched_lectures (
    id BIGINT PRIMARY KEY,
    lecture_external_id VARCHAR,
    lecture_title TEXT,
    lecture_description TEXT,
    lecture_tags TEXT,        -- Optional: for learning
    lecture_tag_ids TEXT,     -- Optional: comma-separated tag IDs for learning
    is_active BOOLEAN,
    soft_deleted BOOLEAN
);
```

## Costs

**Embeddings (one-time):**
- ~$0.13 per 1M tokens
- Example: 10,000 lectures × 100 tokens each = 1M tokens ≈ $0.13

**LLM Arbiter (per lecture, borderline cases only):**
- ~$0.15 per 1M input tokens (gpt-4o-mini)
- Example: 2,000 borderline cases × 500 tokens each = 1M tokens ≈ $0.15

**Total for 10K lectures:** ~$0.30-$0.50 (depending on borderline %)

## QA & Evaluation

The batch generates a QA report showing:
- Coverage: % of lectures with suggestions
- Distribution: How many tags per lecture
- Score statistics: Min/max/mean/median scores
- Model usage: How many suggestions used LLM vs. prototype only

**Manual Review Recommended:**
Sample 50-100 suggestions across score ranges to validate precision and adjust thresholds if needed.

## Troubleshooting

**No prototypes built:**
- Ensure you have existing tagged lectures with `lecture_tag_ids` populated
- At least some tags need ≥1 example to build prototypes

**Low coverage:**
- Check threshold settings (may be too strict)
- Verify tag labels are in Hebrew and match lecture content semantics

**High LLM costs:**
- Increase `high_confidence_threshold` to skip more lectures
- Narrow `llm_borderline_lower/upper` range
- Set `USE_LLM=false` to disable arbiter entirely

**Database connection errors:**
- Verify `DATABASE_URL` format: `postgresql://user:pass@host:port/dbname`
- Ensure database user has SELECT permission on `enriched_lectures`
- For DB writes, ensure CREATE TABLE and INSERT permissions

## Security

- Never commit `.env` file (excluded via `.gitignore`)
- Use read-only database user for safety
- Rotate any API keys that were shared in plaintext
- All secrets loaded from environment variables only

## Phase 2 (Future)

- Approval UI for reviewing suggestions
- Airtable API integration to push accepted tags
- Incremental mode (only process new/updated lectures)
- A/B testing different threshold configurations

## Support

For issues or questions:
1. Check the `batch_run.log` file for detailed execution logs
2. Review the QA report for coverage and distribution stats
3. Verify environment variables and database connectivity
4. Test with a small sample first (filter by `LIMIT 100` in SQL)
