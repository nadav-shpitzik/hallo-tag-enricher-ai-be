# Tag Suggestions API

## Overview
A stateless REST API for suggesting tags to Hebrew lectures using AI-powered semantic analysis. The system leverages OpenAI embeddings and prototype learning to provide intelligent, high-precision tag suggestions. The project aims to deliver a fast, scalable, and cost-effective solution for automated content tagging, with a focus on accuracy and explainability for Hebrew content.

## User Preferences
Not specified.

## System Architecture

### Core Functionality
The system provides endpoints for:
- **Training**: Accepts training lectures with existing tags (JSON or CSV) to generate embeddings, compute tag prototypes (centroids), calibrate confidence thresholds, and save prototypes to PostgreSQL with versioning.
- **Auto-Training**: New `/get-data-and-train` endpoint that automatically fetches training data from an external API (`hallo-tags-manager.replit.app`) using X-API-KEY authentication, transforms the data format, and initiates background training without blocking the response. Returns immediately with HTTP 202 status.
- **Suggestion**: Provides tag suggestions for new lectures by loading pre-computed prototypes, generating embeddings for input lectures, and scoring against prototypes using cosine similarity.
- **Management**: Endpoints for reloading prototypes and viewing prototype versions and tag information.

### Scoring Modes
The API supports four scoring modes, balancing quality, speed, and cost:
1.  **Ensemble Mode (`"ensemble"`)**: (DEFAULT) Combines per-category reasoning model (80%) and prototype model (20%) with an agreement bonus for highest accuracy. Uses 5 focused prompts (one per category) for improved recall on long-tail labels. Includes lecturer bio auto-enrichment.
2.  **Fast Mode (`"fast"`)**: Uses only prototype similarity for the fastest and cheapest suggestions.
3.  **Full Quality Mode (`"full_quality"`)**: Uses prototype scoring with an LLM arbiter for borderline cases, balancing speed and quality.
4.  **Reasoning Mode (`"reasoning"`)**: Uses per-category GPT-4o analysis (5 parallel prompts) providing highest-quality suggestions with detailed Hebrew rationales and lecturer bio auto-enrichment. Optimized for better recall on long-tail labels.

### Data Flow
-   **Training Flow**: Client sends training data, API generates embeddings, builds prototypes, calibrates thresholds, and saves versioned prototypes to PostgreSQL.
-   **Suggestion Flow**: Client sends lectures, API loads/caches prototypes, generates embeddings, scores lectures using the selected scoring mode, and returns suggestions.

### UI/UX Decisions
-   A web interface (`/train-ui`) is provided for user-friendly model training with two options:
    - **CSV Upload**: Manual upload of three CSV files (lectures, labels, lecture_labels) with real-time progress tracking
    - **Auto-Fetch Button**: One-click button to automatically fetch latest training data from external API and initiate background training
-   The interface features a gradient design, clear visual separation between training methods, and informative success messages showing fetched data counts and training progress.

### Technical Implementations
-   **Stateless Design**: All lecture/label data is provided via API payloads.
-   **PostgreSQL Storage**: Prototypes are stored in PostgreSQL, enabling versioning and visibility.
-   **In-memory Caching**: Pre-computed prototypes are cached in memory for fast suggestion responses.
-   **Background Training**: The `/get-data-and-train` endpoint uses threading to run training asynchronously, allowing immediate API response while training completes in the background.
-   **Forced GPT-4o Model**: System hardcoded to use GPT-4o (not mini) for superior instruction-following and exact tag name matching. All 115 tags are passed to the LLM in reasoning/ensemble modes.
-   **Exact Tag Matching Prompts**: LLM prompts include explicit Hebrew examples of wrong behavior (missing ה prefix like "חברה ישראלית" vs "החברה הישראלית", invented tags like "עיתונאות" vs "מדיה ותקשורת") to enforce character-by-character matching.
-   **Lecturer Bio Enrichment**: In reasoning and ensemble modes, GPT-4o is used to search, validate, and cache lecturer bios in PostgreSQL (`lecturer_bios` table) to enrich LLM prompts and improve accuracy.
-   **Per-Category Reasoning**: (NEW) Reasoning scorer now runs 5 focused prompts instead of 1 mixed prompt - one per category (Topic, Persona, Tone, Format, Audience). Each category sees only its relevant tags (~20-30 tags instead of all 115), allowing the model to pay better attention and improve recall on long-tail labels. All 5 calls run in parallel using ThreadPoolExecutor for speed. Results are aggregated and fed into the ensemble scorer with the same weights (80% reasoning, 20% prototype).
-   **Optimized Prompt Engineering**: Category prompts refactored to eliminate redundancy and improve maintainability. Common rules (Hebrew output requirement, JSON format, exact tag matching, rationale length) are centralized in `BASE_RULES_HE` and composed with category-specific instructions in `CATEGORY_INSTRUCTIONS_HE`. This reduces token costs (~150 tokens saved per request across 5 parallel calls) and makes prompt updates easier to maintain while ensuring consistent behavior.
-   **Category Name Normalization**: System uses canonical English category names internally (Topic, Persona, Audience, Tone, Format) for consistency with config thresholds and scoring pipeline. Hebrew category names from source data (נושא, פרסונה, קהל יעד, טון, פורמט) are automatically normalized to English at ingestion via `CATEGORY_MAP_HE_TO_EN` mapping in `category_reasoning.py`.
-   **Smart Ensemble Combiner**: (NEW) Ensemble scorer implements intelligent fallback mechanisms:
    - **Backfill Logic**: If all 5 category reasoning calls return empty (edge case for very short/unclear lectures), the system automatically adds the single highest-scoring prototype tag that meets its category threshold, preventing zero-suggestion responses
    - **Guarded Agreement Bonus**: The +0.15 agreement bonus (applied when both reasoning and prototype agree on a tag) is only granted if the prototype score is strong enough (≥ threshold - 0.03 margin), preventing false confidence from weak prototype matches
    - **Prototype-Only Filtering**: Prototype suggestions that lack reasoning support must meet their category-specific threshold to be included, preserving precision
    - **Comprehensive Logging**: Tracks backfill triggers, agreement bonus counts, and top-3 suggestions per lecture for observability
-   **Telemetry System**: (NEW) Lean NDJSON-based telemetry for observability and verification:
    - **Thread-Safe Logging**: `src/telemetry.py` provides thread-safe NDJSON event logging to configurable path (default: `/tmp/tagger.ndjson`)
    - **Category Tracking**: Each category reasoning result is logged with chosen tag IDs, confidence scores, and rationales count
    - **Ensemble Tracking**: Each ensemble result is logged with backfill status, agreement bonus count, top-3 suggestions with detailed scores
    - **Offline Analysis**: `analyze_logs.py` script computes coverage, backfill rate, avg tags, agreement rate, per-category non-empty rates, and flags possible misses
    - **Human Review**: `sample_human_review.py` script randomly samples results for manual inspection and quality verification
    - **Zero Performance Impact**: Async file writes with no blocking, purely for observability
-   **Structured Logging**: Uses structured JSON logging with `request_id` correlation, performance metrics, business metrics, and error context for observability.
-   **LLM Cost Monitoring**: Tracks token usage and estimates costs for all OpenAI API calls.
-   **AI Call Tracking**: All ReasoningScorer AI calls are logged to PostgreSQL (`ai_calls` table) with full prompt/response content (JSONB), token counts, costs, duration, and status for auditing and debugging expensive GPT-4o calls.
-   **Discord Notifications**: Configurable Discord webhooks for comprehensive request summaries, performance data, quality metrics, and error details.

### Files Structure
-   `api_server.py`: Main API server.
-   `train_prototypes.py`: Standalone training script.
-   `analyze_logs.py`: Offline telemetry analyzer for computing metrics and identifying edge cases.
-   `sample_human_review.py`: Random sampling tool for human review of tagging results.
-   `src/`: Contains core modules like `config.py`, `embeddings.py`, `prototype_knn.py`, `prototype_storage.py`, `scorer.py`, `reasoning_scorer.py`, `category_reasoning.py`, `ensemble_scorer.py`, `llm_arbiter.py`, `lecturer_search.py`, `ai_call_logger.py`, `csv_parser.py`, `shortlist.py`, and `telemetry.py`.

## External Dependencies
-   **OpenAI**: Used for `text-embedding-3-large` embeddings and GPT-4o for LLM-based reasoning, arbitration, and lecturer bio enrichment.
-   **PostgreSQL**: Database for storing versioned prototypes and cached lecturer bios.
-   **Flask**: Python web framework for building the REST API.
-   **NumPy**: For numerical operations, particularly vector manipulation.
-   **Scikit-learn**: For similarity computations.
-   **Pandas**: For data processing, especially for CSV input.
-   **Python-dotenv**: For managing environment variables.
-   **Discord Webhooks**: Optional integration for sending API notifications.
-   **External Training API**: The `/get-data-and-train` endpoint fetches from `hallo-tags-manager.replit.app/v2/train-data`.

## Known Limitations
-   **SSL Verification**: The `/get-data-and-train` endpoint uses `verify=False` for HTTPS requests due to certificate chain validation issues in the Replit environment. Multiple SSL verification approaches were attempted (system cacert, certifi, default CA bundle) but failed consistently. This is acceptable for internal Replit-to-Replit communication with API key authentication. For production deployment, investigate certificate chain configuration.