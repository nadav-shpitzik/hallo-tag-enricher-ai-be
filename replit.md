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

### Web Viewer Added
- Built Flask-based web viewer for browsing results interactively
- Features: filterable by score, tag name, model type, previous tags status
- Hebrew RTL support with responsive design
- Shows lecture details, previous tags, and new suggestions per lecture
- Live statistics dashboard (193 lectures, 295 suggestions, 74.1% avg score)

### Shortlist Optimization Added (Latest)
- **75% cost reduction** through intelligent candidate shortlisting
- Multi-signal shortlist reduces from 85 → ~20 tags per lecture
- Signals: Hebrew keyword matching, embeddings similarity, lecturer history
- Maintains 100% coverage with dynamic expansion for edge cases
- Test results: 10 lectures → 30 suggestions with 88.5% avg score

### Files Structure
```
src/
  ├── config.py          # Configuration from environment
  ├── database.py        # PostgreSQL connection (read-only)
  ├── tags_loader.py     # CSV loading and normalization
  ├── embeddings.py      # OpenAI embeddings generation
  ├── prototype_knn.py   # Prototype learning and scoring
  ├── llm_arbiter.py     # LLM refinement with Structured Outputs
  ├── scorer.py          # Main scoring engine (multi-mode)
  ├── reasoning_scorer.py # Reasoning mode with Hebrew rationales
  ├── shortlist.py       # Candidate shortlist optimizer (75% cost reduction)
  ├── lecturer_search.py # Web search for lecturer profiles
  ├── output.py          # CSV/DB output and QA reports
  └── main.py            # Orchestration and execution

data/
  └── tags.csv           # Tags with Hebrew names and synonyms

output/                  # Generated suggestions (gitignored)
  ├── tag_suggestions.csv      # Main output (295 suggestions)
  └── suggestions_report.txt   # Text report for review

web_viewer.py            # Flask web viewer for browsing results
view_results.py          # CLI script for text-based viewing
templates/
  └── index.html         # Web viewer HTML template
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

### Viewing Results

**Web Viewer (Interactive):**
- Run the "Web Viewer" workflow from Replit UI
- Browse suggestions with filters (score, tags, model type)
- Hebrew-friendly interface with per-lecture details

**CLI Viewer:**
```bash
python view_results.py
```
Generates text report with all lecture details and suggestions

**Direct CSV:**
- `output/tag_suggestions.csv` - Machine-readable results
- `output/suggestions_report.txt` - Human-readable text report

### Configuration Options
- `TEST_MODE=true|false` - Enable test mode with limited lectures (default: false)
- `TEST_MODE_LIMIT=30` - Number of lectures to process in test mode (default: 30)
- `USE_LLM=true|false` - Enable/disable LLM arbiter (default: true)
- `WRITE_TO_DB=true|false` - Write to database table (default: false)
- See `src/config.py` for all tunable parameters

## Cost Estimates (With Shortlist Optimization)

### Reasoning Mode (Current)
- For 300 untagged lectures: ~$0.29
- Cost reduction: 75% vs full-list approach
- Quality maintained: 85-95% confidence scores

### Token Usage
- Without shortlist: ~7.65M tokens ($1.15) for 300 lectures
- With shortlist: ~1.95M tokens ($0.29) for 300 lectures
- Savings: ~5.7M tokens ($0.86)

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
- flask (web viewer)

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
