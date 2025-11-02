# Tag Suggestions API

## Overview
A stateless REST API for suggesting tags to Hebrew lectures using AI-powered semantic analysis. The system leverages OpenAI embeddings and prototype learning to provide intelligent, high-precision tag suggestions. The project aims to deliver a fast, scalable, and cost-effective solution for automated content tagging, with a focus on accuracy and explainability for Hebrew content.

## User Preferences
Not specified.

## System Architecture

### Core Functionality
The system provides endpoints for:
- **Training**: Accepts training lectures with existing tags (JSON or CSV) to generate embeddings, compute tag prototypes (centroids), calibrate confidence thresholds, and save prototypes to PostgreSQL with versioning.
- **Suggestion**: Provides tag suggestions for new lectures by loading pre-computed prototypes, generating embeddings for input lectures, and scoring against prototypes using cosine similarity.
- **Management**: Endpoints for reloading prototypes and viewing prototype versions and tag information.

### Scoring Modes
The API supports four scoring modes, balancing quality, speed, and cost:
1.  **Ensemble Mode (`"ensemble"`)**: (NEW DEFAULT) Combines reasoning model (80%) and prototype model (20%) with an agreement bonus for highest accuracy. Includes lecturer bio auto-enrichment.
2.  **Fast Mode (`"fast"`)**: Uses only prototype similarity for the fastest and cheapest suggestions.
3.  **Full Quality Mode (`"full_quality"`)**: Uses prototype scoring with an LLM arbiter for borderline cases, balancing speed and quality.
4.  **Reasoning Mode (`"reasoning"`)**: Pure GPT-4o analysis providing highest-quality suggestions with detailed Hebrew rationales and lecturer bio auto-enrichment.

### Data Flow
-   **Training Flow**: Client sends training data, API generates embeddings, builds prototypes, calibrates thresholds, and saves versioned prototypes to PostgreSQL.
-   **Suggestion Flow**: Client sends lectures, API loads/caches prototypes, generates embeddings, scores lectures using the selected scoring mode, and returns suggestions.

### UI/UX Decisions
-   A web interface (`/train-ui`) is provided for user-friendly CSV upload and model training, featuring a gradient design and real-time progress tracking.

### Technical Implementations
-   **Stateless Design**: All lecture/label data is provided via API payloads.
-   **PostgreSQL Storage**: Prototypes are stored in PostgreSQL, enabling versioning and visibility.
-   **In-memory Caching**: Pre-computed prototypes are cached in memory for fast suggestion responses.
-   **Lecturer Bio Enrichment**: In reasoning and ensemble modes, GPT-4o is used to search, validate, and cache lecturer bios in PostgreSQL (`lecturer_bios` table) to enrich LLM prompts and improve accuracy.
-   **Category-Aware Prompting**: LLM prompts include explicit definitions for 5 tag categories (Topic, Persona, Tone, Format, Audience) and organize tags by category for improved context understanding. Structure supports future category-specific instructions.
-   **Structured Logging**: Uses structured JSON logging with `request_id` correlation, performance metrics, business metrics, and error context for observability.
-   **LLM Cost Monitoring**: Tracks token usage and estimates costs for all OpenAI API calls.
-   **Discord Notifications**: Configurable Discord webhooks for comprehensive request summaries, performance data, quality metrics, and error details.

### Files Structure
-   `api_server.py`: Main API server.
-   `train_prototypes.py`: Standalone training script.
-   `src/`: Contains core modules like `config.py`, `embeddings.py`, `prototype_knn.py`, `prototype_storage.py`, `scorer.py`, `reasoning_scorer.py`, `ensemble_scorer.py`, `llm_arbiter.py`, `lecturer_search.py`, `csv_parser.py`, and `shortlist.py`.

## External Dependencies
-   **OpenAI**: Used for `text-embedding-3-large` embeddings and GPT-4o for LLM-based reasoning, arbitration, and lecturer bio enrichment.
-   **PostgreSQL**: Database for storing versioned prototypes and cached lecturer bios.
-   **Flask**: Python web framework for building the REST API.
-   **NumPy**: For numerical operations, particularly vector manipulation.
-   **Scikit-learn**: For similarity computations.
-   **Pandas**: For data processing, especially for CSV input.
-   **Python-dotenv**: For managing environment variables.
-   **Discord Webhooks**: Optional integration for sending API notifications.