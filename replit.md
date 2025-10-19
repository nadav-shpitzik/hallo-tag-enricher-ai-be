# AI Tag Enrichment Batch System

## Overview
This project implements a one-off batch system for enriching Hebrew lectures with AI-suggested tags. It uses OpenAI embeddings and prototype learning to suggest tag IDs based on lecture titles and descriptions.

## Project Type
**Batch Processing System** (not a web application)
- One-time or on-demand execution
- Reads from PostgreSQL, outputs to CSV
- No continuous web server required

## Architecture

### Core Components
- **Embeddings**: OpenAI text-embedding-3-large for Hebrew semantic understanding
- **Prototype Learning**: Per-tag centroids from existing tagged lectures
- **Threshold Calibration**: Per-tag precision-optimized thresholds
- **LLM Arbiter**: Optional GPT-4o-mini for borderline case refinement
- **Output**: CSV and optional PostgreSQL table

### Data Flow
1. Load tags from CSV (Hebrew labels + synonyms)
2. Fetch lectures from PostgreSQL (`enriched_lectures` table)
3. Generate embeddings for lectures and tags
4. Build prototypes from existing tagged lectures
5. Score new lectures against prototypes
6. Optional LLM refinement for borderline cases
7. Output suggestions to CSV and/or database

## Recent Changes (2025-10-19)

### Initial Implementation
- Created modular Python architecture with 9 core modules
- Implemented embedding generation with OpenAI API batching
- Built prototype-kNN system with low-data tag handling
- Added per-tag threshold calibration for high precision
- Integrated LLM arbiter with Structured Outputs (JSON Schema)
- Created CSV and database output generators
- Added QA reporting (coverage, distribution, score stats)

### Files Structure
```
src/
  ├── config.py          # Configuration from environment
  ├── database.py        # PostgreSQL connection (read-only)
  ├── tags_loader.py     # CSV loading and normalization
  ├── embeddings.py      # OpenAI embeddings generation
  ├── prototype_knn.py   # Prototype learning and scoring
  ├── llm_arbiter.py     # LLM refinement with Structured Outputs
  ├── scorer.py          # Main scoring engine
  ├── output.py          # CSV/DB output and QA reports
  └── main.py            # Orchestration and execution

data/
  └── tags_example.csv   # Example tags format

output/                  # Generated suggestions (gitignored)

validate_setup.py        # Setup validation helper
```

## User Preferences
- **Language**: Python 3.11
- **Code Style**: Type hints, logging, error handling
- **Security**: Environment variables for all secrets, read-only DB access
- **Performance**: Batched API calls, efficient embedding generation

## How to Use

### First Time Setup
1. Copy `.env.example` to `.env`
2. Configure required variables:
   - `DATABASE_URL` (PostgreSQL connection)
   - `OPENAI_API_KEY` (your API key)
   - `TAGS_CSV_PATH` (path to tags CSV)
   - `OUTPUT_CSV_PATH` (where to save results)
3. Prepare your tags CSV with format: `tag_id,name_he,synonyms_he`
4. Run validation: `python validate_setup.py`

### Running the Batch
```bash
python src/main.py
```

This will:
- Generate embeddings for all lectures and tags
- Build prototypes from existing tagged lectures
- Score and suggest tags for all lectures
- Save results to CSV (and optionally database)
- Generate QA report

### Configuration Options
- `USE_LLM=true|false` - Enable/disable LLM arbiter (default: true)
- `WRITE_TO_DB=true|false` - Write to database table (default: false)
- See `src/config.py` for all tunable parameters

## Cost Estimates
- Embeddings: ~$0.13 per 1M tokens (~10K lectures)
- LLM arbiter: ~$0.15 per 1M tokens (borderline cases only)
- Total: ~$0.30-$0.50 for 10K lectures

## Phase 2 (Planned)
- Human approval UI for reviewing suggestions
- Airtable API integration for pushing accepted tags
- Incremental mode (only new/updated lectures)
- A/B testing for threshold optimization

## Dependencies
- openai (embeddings + LLM)
- psycopg2-binary (PostgreSQL)
- pandas (data processing)
- numpy (vector operations)
- scikit-learn (similarity computations)
- python-dotenv (environment config)

## Security Notes
- Uses read-only database user (recommended)
- All secrets via environment variables
- No secrets logged or committed to git
- `.env` file excluded in `.gitignore`

## Known Limitations
- Requires existing tagged lectures to build prototypes (at least 1 per tag)
- Hebrew-specific (embeddings model supports multilingual but system designed for Hebrew)
- One-off batch (not incremental by default)
- Phase 1: No writeback to source tables or Airtable
