# Tag Suggestions API

## Overview
A stateless REST API for suggesting tags to Hebrew lectures using AI-powered semantic analysis. The system uses OpenAI embeddings and prototype learning to provide intelligent tag suggestions.

## Project Type
**Stateless REST API**
- No database dependencies
- All data provided via API payloads
- Pre-computed prototypes stored in Replit KV store
- Fast, scalable tag suggestion endpoint

## Architecture

### Core Components
1. **Training Endpoint** (`POST /train`)
   - Accepts training lectures with existing tags as JSON
   - Generates embeddings using OpenAI text-embedding-3-large
   - Computes tag prototypes (centroids) from training data
   - Calibrates per-tag confidence thresholds
   - Saves prototypes to Replit KV store

2. **Suggestion Endpoint** (`POST /suggest-tags`)
   - Accepts new lectures to tag as JSON
   - Loads pre-computed prototypes from KV store
   - Generates embeddings for input lectures
   - Scores against prototypes using cosine similarity
   - Returns tag suggestions with confidence scores

3. **Reload Endpoint** (`POST /reload-prototypes`)
   - Reloads prototypes from KV store without restart
   - Useful after retraining prototypes

### Data Flow
```
Training Flow:
1. Client sends training lectures + tags via POST /train
2. API generates embeddings for lectures and tags
3. API builds prototypes (centroids per tag)
4. API calibrates thresholds for precision
5. API saves prototypes to Replit KV store
6. Returns training summary

Suggestion Flow:
1. Client sends lectures + tags via POST /suggest-tags
2. API loads prototypes from KV store (cached in memory)
3. API generates embeddings for input lectures
4. API scores lectures using selected scoring mode:
   - FAST: Prototype similarity only (~1s)
   - FULL_QUALITY: Prototype + LLM arbiter for borderline (~2-3s) ✓ Recommended
   - REASONING: Pure LLM analysis with Hebrew rationales (~5-7s)
5. Returns suggestions above threshold with scores
```

### Scoring Modes

The API supports three scoring modes that balance quality, speed, and cost:

**1. Fast Mode (`"fast"`)**
- Uses only prototype similarity matching
- Fastest option (~1 second per lecture)
- Cheapest (~$0.0002 per lecture)
- Good baseline quality
- Best for: Batch processing, quick previews

**2. Full Quality Mode (`"full_quality"`)** ✓ **Recommended Default**
- Prototype scoring + LLM arbiter for borderline cases
- Balanced speed (~2-3 seconds per lecture)
- Moderate cost (~$0.001-0.003 per lecture)
- LLM reviews uncertain suggestions (0.50-0.80 confidence)
- Auto-approves high confidence (≥0.80)
- Adds "llm_refined" reason to arbiter-approved suggestions
- Best for: Production use, general tagging

**3. Reasoning Mode (`"reasoning"`)**
- Pure GPT-4o-mini analysis of lecture content
- Highest quality but slowest (~5-7 seconds per lecture)
- Most expensive (~$0.004-0.008 per lecture)
- Generates detailed Hebrew rationales for each suggestion
- Confidence scores calibrated (scaled by 0.85 to prevent over-confidence)
- Adds "rationale_he" field with explanations
- **Auto-enrichment**: Searches for lecturer bio using GPT-4o if `lecturer_id` or `lecturer_name` provided
- Lecturer bios validated and cached in database for instant reuse
- Best for: Critical accuracy needs, when explanations are valuable

**How to Use:**
Add `"scoring_mode": "full_quality"` to your /suggest-tags request, or set the `SCORING_MODE` environment variable to change the default.

## API Endpoints

### POST /train
Train prototypes from training data and save to KV store.

**Request:**
```json
{
  "lectures": [
    {
      "id": "lec_001",
      "lecture_title": "שיעור בתלמוד",
      "lecture_description": "עיון במסכת ברכות",
      "lecture_tag_ids": ["talmud", "gemara"]
    }
  ],
  "tags": {
    "talmud": {
      "tag_id": "talmud",
      "name_he": "תלמוד",
      "synonyms_he": "גמרא תלמוד בבלי"
    }
  }
}
```

**Response:**
```json
{
  "status": "success",
  "num_prototypes": 12,
  "num_lectures": 5,
  "num_tags": 12,
  "low_data_tags": 3
}
```

### POST /train-csv
Train prototypes from uploaded CSV files (alternative to JSON training).

**Request:**
Upload 3 CSV files as multipart/form-data:
- `lectures`: CSV with columns `airtable_id`, `title`, `description`, `lecturer_id`
- `labels`: CSV with columns `airtable_id`, `name`, `category`
- `lecture_labels`: CSV with columns `lecture_id`, `label_id` (junction table)

**Example CSV files:**

lectures.csv:
```csv
airtable_id,title,description,lecturer_id
lec_001,תלמוד בבלי,עיון במסכת ברכות,rec_lec1
```

labels.csv:
```csv
airtable_id,name,category
tag_talmud,תלמוד,נושא
```

lecture_labels.csv:
```csv
lecture_id,label_id
lec_001,tag_talmud
```

**Response:**
Same as POST /train

### GET /train-ui
Web interface for CSV upload and training. Provides a user-friendly form to upload the 3 CSV files and train the model with real-time progress tracking.

### POST /suggest-tags
Get tag suggestions for lectures based on pre-trained prototypes.

**Request:**
```json
{
  "lectures": [
    {
      "id": "test_001",
      "lecture_title": "קבלה ומיסטיקה",
      "lecture_description": "סודות הזוהר והקבלה"
    }
  ],
  "tags": {
    "kabbalah": {
      "tag_id": "kabbalah",
      "name_he": "קבלה",
      "synonyms_he": "סודות מיסטיקה"
    }
  }
}
```

**Response:**
```json
{
  "suggestions": [
    {
      "lecture_id": "test_001",
      "tag_id": "kabbalah",
      "tag_name_he": "קבלה",
      "score": 0.865,
      "rationale": "Prototype similarity score: 0.865"
    }
  ],
  "num_lectures": 1,
  "num_suggestions": 1
}
```

### POST /reload-prototypes
Reload prototypes from KV store without restarting the server.

**Response:**
```json
{
  "status": "success",
  "num_prototypes": 12
}
```

### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "prototypes_loaded": true,
  "num_prototypes": 12
}
```

### GET /
API information and available endpoints.

## How to Use

### 1. Initial Setup
Set the `OPENAI_API_KEY` environment variable (required for embeddings):
```bash
export OPENAI_API_KEY=your-api-key-here
```

### 2. Start the API Server
The API server runs automatically via the "API Server" workflow on port 5000.

### 3. Train Prototypes
Send your training data (lectures with existing tags) to train the model:

```bash
curl -X POST http://localhost:5000/train \
  -H "Content-Type: application/json" \
  -d @training_data.json
```

Training happens once or periodically when you have new training data. The prototypes are saved to Replit KV store and persist across server restarts.

### 4. Get Tag Suggestions
Send new lectures to get tag suggestions:

```bash
curl -X POST http://localhost:5000/suggest-tags \
  -H "Content-Type: application/json" \
  -d @request_data.json
```

## Testing

Run the test script to verify the complete flow:

```bash
python test_api.py
```

This will:
1. Check API health
2. Train prototypes with sample data
3. Get tag suggestions for test lectures
4. Test the reload endpoint

## Configuration

### Environment Variables
- `OPENAI_API_KEY` (required): Your OpenAI API key for embeddings
- `PORT` (optional): API server port (default: 5000)

### Model Settings
Configured in `src/config.py`:
- **Embedding Model**: `text-embedding-3-large` (3072 dimensions)
- **Target Precision**: 0.90 (high precision for tag suggestions)
- **Min Confidence**: 0.60 (threshold for suggestions)
- **Prototype Weight**: 0.8 (vs label weight 0.2 for low-data tags)

## Lecturer Bio Enrichment

**Available in Reasoning Mode only**

When you provide `lecturer_id` and/or `lecturer_name` in your request, the reasoning mode automatically:

1. **Searches for lecturer bio** using GPT-4o
   - First-time lookup: ~2-3 seconds, ~$0.005
   - Searches for professional background, expertise, teaching style
   - High accuracy with GPT-4
   
2. **Validates bio against lecture** using GPT-4o-mini
   - Checks if bio makes sense with lecture description
   - Prevents caching incorrect/mismatched bios
   - Only caches validated bios
   
3. **Caches in PostgreSQL database**
   - Instant retrieval on subsequent requests
   - Persists across server restarts
   - Table: `lecturer_bios`
   
4. **Enriches LLM prompt**
   - Adds lecturer expertise context
   - Improves label accuracy
   - Better understanding of lecture content

**Cost**: One-time ~$0.006 per unique lecturer (search + validation), then free (cached)

**Example Request:**
```json
{
  "scoring_mode": "reasoning",
  "lecture": {
    "id": "rec123",
    "title": "קבלה ומיסטיקה",
    "description": "...",
    "lecturer_id": "recXYZ789",
    "lecturer_name": "הרב משה כהן"
  },
  "labels": [...]
}
```

## Files Structure

```
api_server.py           # Main API server with all endpoints
train_prototypes.py     # Standalone training script (CLI/API mode)
test_api.py             # Test script for API validation

src/
  ├── config.py         # Configuration (no database dependencies)
  ├── embeddings.py     # OpenAI embeddings generation
  ├── prototype_knn.py  # Prototype learning and scoring
  ├── scorer.py         # Multi-mode scoring engine
  ├── reasoning_scorer.py # LLM-based reasoning scorer
  ├── llm_arbiter.py    # LLM refinement logic
  ├── lecturer_search.py # Lecturer bio search with DB caching
  ├── csv_parser.py     # CSV file parser for training uploads
  └── shortlist.py      # Candidate shortlist optimizer
```

## Recent Changes

### 2025-10-29: CSV Upload Feature
- **Added CSV training interface** at `/train-ui` for easy model training
- **New endpoint** `POST /train-csv` accepts 3 CSV files (lectures, labels, junction table)
- **CSV parser** (`src/csv_parser.py`) handles data transformation from CSV to training format
- **Web UI** with beautiful gradient design, file upload, progress tracking, and training stats
- **Benefits**: Upload CSVs directly from Airtable/database exports, no manual JSON formatting needed

### 2025-10-27: API Transformation
- **Removed all database dependencies** (PostgreSQL, psycopg2)
- **Stateless design**: All data via API payloads
- **Replit KV Store**: Prototypes stored in key-value store
- **Combined endpoints**: Training and suggestions in one API
- **Removed components**: Web viewer, Airtable sync, batch processing
- **Kept core AI logic**: Embeddings, prototypes, scoring, LLM arbiter

### Benefits
- ✓ **Portable**: No database setup required
- ✓ **Stateless**: Each request is independent
- ✓ **Fast**: Pre-computed prototypes loaded in memory
- ✓ **Scalable**: Train once, serve many
- ✓ **Simple**: Single API server, clear endpoints

## Dependencies
- `openai` - Embeddings and LLM
- `flask` - REST API framework
- `numpy` - Vector operations
- `scikit-learn` - Similarity computations
- `pandas` - Data processing
- `python-dotenv` - Environment config
- `replit` - KV store integration

## Cost Estimates

### Training (one-time or periodic)
- For 100 training lectures: ~$0.02 in embeddings
- For 50 tags: ~$0.001 in embeddings
- **Total**: ~$0.02 per training run

### Suggestions (per request)
- For 10 lectures: ~$0.002 in embeddings
- **Total**: ~$0.0002 per lecture

Training is infrequent (only when you have new training data), while suggestions are fast and cheap.

## Security Notes
- API key stored in environment variable
- No data persistence (except prototypes in KV store)
- No authentication (add as needed for production)
- CORS disabled (enable if needed for web clients)
