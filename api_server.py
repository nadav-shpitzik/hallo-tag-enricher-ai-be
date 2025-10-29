#!/usr/bin/env python3
"""
Stateless API for tag suggestions.

Endpoints:
- POST /train: Train prototypes from training data and save to PostgreSQL
- POST /suggest-tags: Get tag suggestions for lectures  
- POST /reload-prototypes: Reload prototypes from PostgreSQL
- GET /health: Health check
- GET /: API information
"""

import os
import json
import logging
import numpy as np
from typing import List, Dict, Optional
from flask import Flask, request, jsonify
from replit import db
from src.embeddings import EmbeddingsGenerator
from src.prototype_knn import PrototypeKNN
from src.config import Config
from src.llm_arbiter import LLMArbiter
from src.reasoning_scorer import ReasoningScorer
from src.lecturer_search import LecturerSearchService
from src.csv_parser import parse_csv_training_data
from src.prototype_storage import PrototypeStorage

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global state
prototypes_loaded = False
prototype_knn = None
tag_embeddings_cache = None
config = None


def load_prototypes_from_db():
    """Load prototypes from PostgreSQL database."""
    global prototypes_loaded, prototype_knn, tag_embeddings_cache, config
    
    try:
        storage = PrototypeStorage()
        result = storage.load_prototypes(version_name='default')
        
        if not result:
            logger.error("No prototypes found in database. Please train first.")
            return False
        
        tag_prototypes, tag_thresholds, tag_stats, tag_embeddings = result
        
        # Initialize config
        config = Config()
        
        # Create PrototypeKNN instance
        prototype_knn = PrototypeKNN(config)
        prototype_knn.tag_prototypes = tag_prototypes
        prototype_knn.tag_thresholds = tag_thresholds
        prototype_knn.tag_stats = tag_stats
        
        # Set tag embeddings cache
        tag_embeddings_cache = tag_embeddings
        
        prototypes_loaded = True
        logger.info(f"Loaded {len(prototype_knn.tag_prototypes)} prototypes from database")
        return True
        
    except Exception as e:
        logger.error(f"Error loading prototypes from database: {e}")
        import traceback
        traceback.print_exc()
        return False




def validate_training_data(lectures: List[Dict], tags_data: Dict) -> Dict[str, any]:
    """
    Validate training data for category diversity and data quality.
    
    Returns warnings and statistics about the training data.
    """
    warnings = []
    stats = {
        'total_lectures': len(lectures),
        'total_tags': len(tags_data),
        'categories': {},
        'low_data_tags': []
    }
    
    # Analyze category distribution
    category_counts = {}
    tag_example_counts = {}
    
    for tag_id, tag_info in tags_data.items():
        category = tag_info.get('category', 'Unknown')
        category_counts[category] = category_counts.get(category, 0) + 1
        tag_example_counts[tag_id] = 0
    
    # Count examples per tag (handle both v1 and v2 formats)
    for lecture in lectures:
        tag_ids = lecture.get('lecture_tag_ids') or lecture.get('label_ids', [])
        for tag_id in tag_ids:
            if tag_id in tag_example_counts:
                tag_example_counts[tag_id] += 1
    
    # Check for low-data tags
    for tag_id, count in tag_example_counts.items():
        if count < 5:
            tag_info = tags_data.get(tag_id, {})
            stats['low_data_tags'].append({
                'tag_id': tag_id,
                'category': tag_info.get('category', 'Unknown'),
                'examples': count
            })
    
    # Store category stats
    for category, count in category_counts.items():
        stats['categories'][category] = {
            'num_tags': count,
            'avg_examples': sum(tag_example_counts[tid] for tid, tinfo in tags_data.items() if tinfo.get('category') == category) / count if count > 0 else 0
        }
    
    # Generate warnings
    if not category_counts:
        warnings.append("No categories found in tags (consider adding 'category' field)")
    elif len(category_counts) < 2:
        warnings.append(f"Only {len(category_counts)} category found. Consider diversifying across Topic, Persona, Tone, Format, Audience")
    
    if len(stats['low_data_tags']) > len(tags_data) * 0.5:
        warnings.append(f"{len(stats['low_data_tags'])} tags have <5 examples (>{len(tags_data)//2} tags). Model may struggle with these.")
    
    if len(lectures) < 20:
        warnings.append(f"Only {len(lectures)} training lectures. Recommend at least 20-50 for reliable prototypes.")
    
    stats['warnings'] = warnings
    return stats


def train_from_data(training_data: dict) -> dict:
    """
    Train prototypes from training data and save to KV store.
    
    Expected input format:
    {
        "lectures": [...],
        "tags": {...}
    }
    """
    lectures = training_data.get('lectures', [])
    tags_data = training_data.get('tags', {})
    
    if not lectures:
        raise ValueError("No lectures provided in training data")
    
    if not tags_data:
        raise ValueError("No tags provided in training data")
    
    logger.info(f"Training on {len(lectures)} lectures with {len(tags_data)} tags")
    
    # Validate training data quality
    validation_stats = validate_training_data(lectures, tags_data)
    for warning in validation_stats['warnings']:
        logger.warning(f"Training validation: {warning}")
    
    # Initialize config
    train_config = Config()
    
    # Generate embeddings
    embeddings_gen = EmbeddingsGenerator(
        api_key=train_config.openai_api_key,
        model=train_config.embedding_model
    )
    
    # Generate lecture embeddings
    lecture_embeddings = embeddings_gen.generate_lecture_embeddings(lectures)
    logger.info(f"Generated embeddings for {len(lecture_embeddings)} lectures")
    
    # Generate tag label embeddings
    tag_label_texts = {}
    for tag_id, tag_info in tags_data.items():
        name = tag_info.get('name_he', '')
        synonyms = tag_info.get('synonyms_he', '')
        if synonyms:
            tag_label_texts[tag_id] = f"{name} {synonyms}"
        else:
            tag_label_texts[tag_id] = name
    
    tag_embeddings = embeddings_gen.generate_tag_embeddings(tag_label_texts)
    logger.info(f"Generated embeddings for {len(tag_embeddings)} tags")
    
    # Build prototypes
    train_prototype_knn = PrototypeKNN(train_config)
    train_prototype_knn.build_prototypes(lectures, lecture_embeddings, tags_data)
    
    # Calibrate thresholds
    train_prototype_knn.calibrate_thresholds(lectures, lecture_embeddings, tag_embeddings)
    
    # Save to PostgreSQL database
    storage = PrototypeStorage()
    version_id = storage.save_prototypes(
        tag_prototypes=train_prototype_knn.tag_prototypes,
        tag_thresholds=train_prototype_knn.tag_thresholds,
        tag_stats=train_prototype_knn.tag_stats,
        tag_embeddings=tag_embeddings,
        num_lectures=len(lectures),
        version_name='default'
    )
    logger.info(f"Saved prototypes to database as version {version_id}")
    
    # Return summary with validation stats
    return {
        'status': 'success',
        'num_prototypes': len(train_prototype_knn.tag_prototypes),
        'num_lectures': len(lectures),
        'num_tags': len(tags_data),
        'low_data_tags': sum(1 for s in train_prototype_knn.tag_stats.values() if s.get('is_low_data', False)),
        'validation': {
            'warnings': validation_stats['warnings'],
            'categories': validation_stats['categories'],
            'num_low_data_tags': len(validation_stats['low_data_tags'])
        }
    }


def score_lecture_fast(lecture: Dict, labels: List[Dict]) -> List[Dict]:
    """
    Fast scoring mode: Prototype similarity only with category-aware thresholds.
    
    Enhancements:
    - Category-specific confidence thresholds
    - Related lectures co-occurrence boost
    - Intelligent reasons generation
    
    Args:
        lecture: Lecture dict with id, title, description, lecturer info, etc.
        labels: List of label dicts with id, name_he, category, active
    
    Returns:
        List of suggestions with label_id, category, confidence, reasons
    """
    if not prototypes_loaded:
        raise RuntimeError("Prototypes not loaded. Please train first or reload prototypes.")
    
    # Convert to old format for embeddings
    lecture_for_embedding = {
        'id': lecture.get('id'),
        'lecture_title': lecture.get('title', ''),
        'lecture_description': lecture.get('description', '')
    }
    
    # Generate embedding
    embeddings_gen = EmbeddingsGenerator(
        api_key=config.openai_api_key,
        model=config.embedding_model
    )
    
    lecture_embeddings = embeddings_gen.generate_lecture_embeddings([lecture_for_embedding])
    lecture_id = lecture.get('id')
    
    if lecture_id not in lecture_embeddings:
        return []
    
    lecture_embedding = lecture_embeddings[lecture_id]
    
    # Get base scores from prototype KNN
    scores = prototype_knn.score_lecture(lecture_embedding, tag_embeddings_cache)
    
    # Create label lookup by id
    labels_by_id = {label['id']: label for label in labels if label.get('active', True)}
    
    # Extract related lectures labels for co-occurrence analysis
    related_labels = set()
    related_lectures = lecture.get('related_lectures', [])
    for related in related_lectures:
        related_labels.update(related.get('labels', []))
    
    # Prepare lecture text for keyword analysis
    title = lecture.get('title', '').lower()
    description = lecture.get('description', '').lower()
    
    # Convert to suggestions format with enhancements
    suggestions = []
    for label_id, base_score in scores.items():
        if label_id not in labels_by_id:
            continue
            
        label = labels_by_id[label_id]
        category = label.get('category', 'Unknown')
        label_name = label.get('name_he', '').lower()
        
        # Apply category-specific threshold
        category_threshold = config.category_thresholds.get(category, config.category_thresholds['default'])
        
        # Calculate confidence with related lectures boost
        confidence = base_score
        reasons = []
        
        # Check for related lectures co-occurrence
        if label_id in related_labels:
            confidence += config.related_lecture_boost
            reasons.append("related_cooccur")
        
        # Analyze score strength for keyword matching signals
        if base_score >= 0.80:
            reasons.append("desc_match")
        elif base_score >= 0.65:
            reasons.append("title_match")
        elif base_score >= 0.50:
            reasons.append("cooccur")
        else:
            # Low score - might be a default category prior
            reasons.append(f"default_{category.lower()}_prior")
        
        # Only include if confidence meets category threshold
        if confidence >= category_threshold:
            suggestion = {
                'label_id': label_id,
                'category': category,
                'confidence': min(float(confidence), 1.0),  # Cap at 1.0
                'reasons': reasons
            }
            suggestions.append(suggestion)
    
    # Sort by confidence descending
    suggestions.sort(key=lambda x: x['confidence'], reverse=True)
    
    return suggestions


def score_lecture_with_arbiter(lecture: Dict, labels: List[Dict]) -> List[Dict]:
    """
    Full quality mode: Prototype scoring + LLM arbiter for borderline cases.
    
    Flow:
    1. Get prototype scores for all labels
    2. Auto-approve high confidence (>= 0.80)
    3. Send borderline (0.60-0.80) to LLM arbiter for review
    4. Reject low confidence (< 0.60)
    
    Args:
        lecture: Lecture dict with id, title, description, etc.
        labels: List of label dicts
    
    Returns:
        List of high-quality suggestions
    """
    if not prototypes_loaded:
        raise RuntimeError("Prototypes not loaded. Please train first or reload prototypes.")
    
    # Get fast prototype scores first
    fast_suggestions = score_lecture_fast(lecture, labels)
    
    if not fast_suggestions:
        return []
    
    # Create labels lookup
    labels_by_id = {label['id']: label for label in labels}
    
    # Split into high, borderline, and low confidence
    high_confidence = []
    borderline = []
    
    for sugg in fast_suggestions:
        conf = sugg['confidence']
        if conf >= config.high_confidence_threshold:
            high_confidence.append(sugg)
        elif conf >= config.llm_borderline_lower:
            borderline.append(sugg)
        # Else: below borderline_lower, reject
    
    # If no borderline cases, return high confidence only
    if not borderline:
        logger.info(f"No borderline suggestions, returning {len(high_confidence)} high confidence")
        return high_confidence
    
    # Use LLM arbiter to refine borderline suggestions
    logger.info(f"LLM arbiter reviewing {len(borderline)} borderline suggestions")
    
    arbiter = LLMArbiter(api_key=config.openai_api_key, config=config)
    
    # Convert borderline to format expected by arbiter
    borderline_scores = {sugg['label_id']: sugg['confidence'] for sugg in borderline}
    candidate_tags = {label_id: labels_by_id[label_id] for label_id in borderline_scores.keys()}
    
    # Call arbiter
    approved_ids = arbiter.refine_suggestions(
        lecture_title=lecture.get('title', ''),
        lecture_description=lecture.get('description', ''),
        candidate_tags=candidate_tags,
        scores=borderline_scores
    )
    
    # Add approved borderline suggestions
    arbiter_approved = [
        sugg for sugg in borderline 
        if sugg['label_id'] in approved_ids
    ]
    
    # Add "llm_refined" to reasons for arbiter-approved suggestions
    for sugg in arbiter_approved:
        if 'reasons' not in sugg:
            sugg['reasons'] = []
        sugg['reasons'].append('llm_refined')
    
    logger.info(f"LLM arbiter approved {len(arbiter_approved)} of {len(borderline)} borderline")
    
    # Combine and sort
    final_suggestions = high_confidence + arbiter_approved
    final_suggestions.sort(key=lambda x: x['confidence'], reverse=True)
    
    return final_suggestions


def score_lecture_with_reasoning(lecture: Dict, labels: List[Dict]) -> List[Dict]:
    """
    Reasoning mode: Pure LLM-based scoring using GPT-4o-mini.
    
    Uses structured output to generate intelligent suggestions with Hebrew rationales.
    Highest quality but slowest and most expensive.
    
    Automatically fetches lecturer bio if lecturer_id or lecturer_name provided.
    
    Args:
        lecture: Lecture dict
        labels: List of label dicts
    
    Returns:
        List of LLM-generated suggestions
    """
    scorer = ReasoningScorer(
        model=config.llm_model,
        min_confidence=config.min_confidence_threshold,
        confidence_scale=config.reasoning_confidence_scale
    )
    
    # Fetch lecturer bio if available
    lecturer_profile = None
    lecturer_id = lecture.get('lecturer_id')
    lecturer_name = lecture.get('lecturer_name')
    lecture_description = lecture.get('description', '')
    
    if lecturer_id or lecturer_name:
        try:
            search_service = LecturerSearchService(api_key=config.openai_api_key)
            lecturer_profile = search_service.get_lecturer_profile(
                lecturer_id=lecturer_id,
                lecturer_name=lecturer_name,
                lecture_description=lecture_description
            )
            if lecturer_profile:
                logger.info(f"Enriching reasoning with lecturer bio: {lecturer_name or lecturer_id}")
        except Exception as e:
            logger.warning(f"Failed to fetch lecturer bio: {e}")
    
    # Convert lecture to format expected by reasoning scorer
    lecture_for_scorer = {
        'id': lecture.get('id'),
        'lecture_title': lecture.get('title', ''),
        'lecture_description': lecture.get('description', ''),
        'lecturer_name': lecturer_name or ''
    }
    
    # Convert labels to format expected by scorer
    labels_for_scorer = []
    for label in labels:
        if not label.get('active', True):
            continue
        labels_for_scorer.append({
            'tag_id': label['id'],
            'name_he': label.get('name_he', ''),
            'synonyms_he': label.get('synonyms_he', ''),
            'category': label.get('category', 'Unknown')
        })
    
    # Call reasoning scorer with lecturer profile
    llm_suggestions = scorer.score_lecture(
        lecture=lecture_for_scorer,
        all_tags=labels_for_scorer,
        lecturer_profile=lecturer_profile
    )
    
    # Convert to v2 format
    v2_suggestions = []
    for llm_sugg in llm_suggestions:
        # Find the label to get category
        label = next((l for l in labels if l['id'] == llm_sugg['tag_id']), None)
        if not label:
            continue
        
        v2_suggestions.append({
            'label_id': llm_sugg['tag_id'],
            'category': label.get('category', 'Unknown'),
            'confidence': llm_sugg['score'],
            'reasons': ['llm_reasoning'],
            'rationale_he': llm_sugg.get('rationale', '')
        })
    
    return v2_suggestions


def score_lecture_v2(lecture: Dict, labels: List[Dict], scoring_mode: str = None) -> List[Dict]:
    """
    Router function for scoring modes.
    
    Modes:
    - "fast": Prototype similarity only (fastest, cheapest)
    - "full_quality": Prototype + LLM arbiter (balanced, recommended)
    - "reasoning": Pure LLM reasoning (highest quality, most expensive)
    
    Args:
        lecture: Lecture dict
        labels: List of label dicts
        scoring_mode: Override config scoring mode
    
    Returns:
        List of suggestions
    """
    mode = scoring_mode or config.scoring_mode
    
    logger.info(f"Scoring lecture {lecture.get('id')} with mode: {mode}")
    
    if mode == "reasoning":
        return score_lecture_with_reasoning(lecture, labels)
    elif mode == "full_quality":
        return score_lecture_with_arbiter(lecture, labels)
    else:  # "fast" or default
        return score_lecture_fast(lecture, labels)


@app.route('/suggest-tags', methods=['POST'])
def suggest_tags():
    """
    Main API endpoint for tag suggestions (v2 format).
    
    Expected JSON format:
    {
        "request_id": "uuid",
        "model_version": "v1",
        "artifact_version": "labels-emb-2025-10-29",
        "scoring_mode": "full_quality" (optional: "fast", "full_quality", "reasoning"),
        "lecture": {
            "id": "rec123",
            "title": "...",
            "description": "...",
            "lecturer_id": "..." (optional),
            "lecturer_name": "..." (optional),
            "lecturer_role": "..." (optional),
            "language": "he" (optional),
            "related_lectures": [...] (optional)
        },
        "labels": [
            {
                "id": "lab_topic_mental_health",
                "name_he": "...",
                "category": "Topic",
                "active": true
            }
        ]
    }
    
    Returns:
    {
        "request_id": "uuid",
        "model_version": "v1",
        "artifact_version": "labels-emb-2025-10-29",
        "suggestions": [
            {
                "label_id": "lab_topic_mental_health",
                "category": "Topic",
                "confidence": 0.91,
                "reasons": ["desc_match", "related_cooccur"]
            }
        ]
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        request_id = data.get('request_id')
        model_version = data.get('model_version', 'v1')
        artifact_version = data.get('artifact_version', 'unknown')
        scoring_mode = data.get('scoring_mode')  # Optional override
        lecture = data.get('lecture')
        labels = data.get('labels', [])
        
        if not lecture:
            return jsonify({'error': 'No lecture provided'}), 400
        
        if not labels:
            return jsonify({'error': 'No labels provided'}), 400
        
        # Score lecture with optional mode override
        suggestions = score_lecture_v2(lecture, labels, scoring_mode=scoring_mode)
        
        response = {
            'request_id': request_id,
            'model_version': model_version,
            'artifact_version': artifact_version,
            'suggestions': suggestions
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error in suggest-tags: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/train', methods=['POST'])
def train():
    """
    Train prototypes from training data.
    
    Supports both old and new formats:
    
    New format (v2):
    {
        "lectures": [
            {
                "id": "rec123",
                "title": "...",
                "description": "...",
                "label_ids": ["lab_topic_mental_health"]
            }
        ],
        "labels": [
            {
                "id": "lab_topic_mental_health",
                "name_he": "...",
                "category": "Topic"
            }
        ]
    }
    
    Old format (v1):
    {
        "lectures": [
            {
                "id": "lec_123",
                "lecture_title": "...",
                "lecture_description": "...",
                "lecture_tag_ids": ["tag1", "tag2"]
            }
        ],
        "tags": {
            "tag1": {
                "tag_id": "tag1",
                "name_he": "...",
                "synonyms_he": "..."
            }
        }
    }
    """
    try:
        training_data = request.get_json()
        
        if not training_data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        # Detect format and convert to old format if needed
        if 'labels' in training_data and isinstance(training_data.get('labels'), list):
            # New format (v2) - convert to old format
            lectures = training_data.get('lectures', [])
            labels = training_data.get('labels', [])
            
            # Convert labels array to tags dict (preserve category)
            tags = {}
            for label in labels:
                tags[label['id']] = {
                    'tag_id': label['id'],
                    'name_he': label.get('name_he', ''),
                    'synonyms_he': label.get('synonyms_he', ''),
                    'category': label.get('category', 'Unknown')
                }
            
            # Convert lectures to old format
            converted_lectures = []
            for lecture in lectures:
                converted_lectures.append({
                    'id': lecture.get('id'),
                    'lecture_title': lecture.get('title', ''),
                    'lecture_description': lecture.get('description', ''),
                    'lecture_tag_ids': lecture.get('label_ids', [])
                })
            
            training_data = {
                'lectures': converted_lectures,
                'tags': tags
            }
        
        result = train_from_data(training_data)
        
        # Automatically reload prototypes after training
        load_prototypes_from_db()
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Training error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/train-csv', methods=['POST'])
def train_csv():
    """
    Train prototypes from uploaded CSV files.
    
    Expected files:
    - lectures.csv: Lecture data (id, title, description, lecturer_id)
    - labels.csv: Label data (id, name, category)
    - lecture_labels.csv: Junction table (lecture_id, label_id)
    """
    try:
        # Check for required files
        if 'lectures' not in request.files:
            return jsonify({'error': 'Missing lectures.csv file'}), 400
        if 'labels' not in request.files:
            return jsonify({'error': 'Missing labels.csv file'}), 400
        if 'lecture_labels' not in request.files:
            return jsonify({'error': 'Missing lecture_labels.csv file'}), 400
        
        # Read file contents
        lectures_csv = request.files['lectures'].read()
        labels_csv = request.files['labels'].read()
        lecture_labels_csv = request.files['lecture_labels'].read()
        
        logger.info("Parsing CSV files...")
        
        # Parse CSV data
        training_data = parse_csv_training_data(
            lectures_csv,
            labels_csv,
            lecture_labels_csv
        )
        
        logger.info(f"Parsed {len(training_data['lectures'])} lectures with {len(training_data['labels'])} labels")
        
        # Convert labels list to tags dict (same as /train endpoint)
        lectures = training_data.get('lectures', [])
        labels = training_data.get('labels', [])
        
        # Convert labels array to tags dict (preserve category)
        tags = {}
        for label in labels:
            tags[label['id']] = {
                'tag_id': label['id'],
                'name_he': label.get('name_he', ''),
                'synonyms_he': label.get('synonyms_he', ''),
                'category': label.get('category', 'Unknown')
            }
        
        # Convert lectures to old format
        converted_lectures = []
        for lecture in lectures:
            converted_lectures.append({
                'id': lecture.get('id'),
                'lecture_title': lecture.get('title', ''),
                'lecture_description': lecture.get('description', ''),
                'lecture_tag_ids': lecture.get('label_ids', [])
            })
        
        # Train using existing logic
        result = train_from_data({
            'lectures': converted_lectures,
            'tags': tags
        })
        
        # Automatically reload prototypes after training
        load_prototypes_from_db()
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"CSV training error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/reload-prototypes', methods=['POST'])
def reload_prototypes():
    """Reload prototypes from KV store without restarting server."""
    try:
        success = load_prototypes_from_db()
        
        if success:
            return jsonify({
                'status': 'success',
                'num_prototypes': len(prototype_knn.tag_prototypes)
            }), 200
        else:
            return jsonify({
                'error': 'Failed to load prototypes. Check logs for details.'
            }), 500
            
    except Exception as e:
        logger.error(f"Error reloading prototypes: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'prototypes_loaded': prototypes_loaded,
        'num_prototypes': len(prototype_knn.tag_prototypes) if prototypes_loaded else 0
    }), 200


@app.route('/prototype-versions', methods=['GET'])
def list_prototype_versions():
    """List all prototype versions stored in the database."""
    try:
        storage = PrototypeStorage()
        versions = storage.list_versions()
        return jsonify({
            'versions': versions,
            'count': len(versions)
        }), 200
    except Exception as e:
        logger.error(f"Error listing versions: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/tag-info/<tag_id>', methods=['GET'])
def get_tag_info(tag_id: str):
    """Get detailed information about a specific tag from the database."""
    try:
        storage = PrototypeStorage()
        info = storage.get_tag_info(tag_id, version_name='default')
        
        if not info:
            return jsonify({'error': f'Tag {tag_id} not found'}), 404
        
        return jsonify(info), 200
    except Exception as e:
        logger.error(f"Error getting tag info: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/train-ui', methods=['GET'])
def train_ui():
    """Web UI for CSV upload and training."""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Train Model - CSV Upload</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 20px;
            }
            .container {
                background: white;
                border-radius: 12px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                max-width: 600px;
                width: 100%;
                padding: 40px;
            }
            h1 {
                color: #2d3748;
                margin-bottom: 10px;
                font-size: 28px;
            }
            .subtitle {
                color: #718096;
                margin-bottom: 30px;
                font-size: 14px;
            }
            .file-upload {
                margin-bottom: 20px;
            }
            label {
                display: block;
                color: #4a5568;
                font-weight: 600;
                margin-bottom: 8px;
                font-size: 14px;
            }
            input[type="file"] {
                width: 100%;
                padding: 12px;
                border: 2px dashed #cbd5e0;
                border-radius: 8px;
                background: #f7fafc;
                cursor: pointer;
                transition: all 0.3s;
            }
            input[type="file"]:hover {
                border-color: #667eea;
                background: #edf2f7;
            }
            .btn {
                width: 100%;
                padding: 14px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 16px;
                font-weight: 600;
                cursor: pointer;
                margin-top: 20px;
                transition: transform 0.2s, box-shadow 0.2s;
            }
            .btn:hover:not(:disabled) {
                transform: translateY(-2px);
                box-shadow: 0 10px 20px rgba(102, 126, 234, 0.4);
            }
            .btn:disabled {
                opacity: 0.6;
                cursor: not-allowed;
            }
            .message {
                padding: 15px;
                border-radius: 8px;
                margin-top: 20px;
                display: none;
            }
            .message.success {
                background: #c6f6d5;
                color: #2f855a;
                border: 1px solid #9ae6b4;
            }
            .message.error {
                background: #fed7d7;
                color: #c53030;
                border: 1px solid #fc8181;
            }
            .message.info {
                background: #bee3f8;
                color: #2c5282;
                border: 1px solid #90cdf4;
            }
            .spinner {
                display: inline-block;
                width: 16px;
                height: 16px;
                border: 2px solid rgba(255,255,255,0.3);
                border-radius: 50%;
                border-top-color: white;
                animation: spin 0.6s linear infinite;
                margin-right: 8px;
            }
            @keyframes spin {
                to { transform: rotate(360deg); }
            }
            .stats {
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 15px;
                margin-top: 15px;
            }
            .stat {
                background: #f7fafc;
                padding: 15px;
                border-radius: 8px;
                text-align: center;
            }
            .stat-value {
                font-size: 24px;
                font-weight: bold;
                color: #667eea;
            }
            .stat-label {
                font-size: 12px;
                color: #718096;
                margin-top: 5px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎓 Train Model</h1>
            <p class="subtitle">Upload your CSV files to train the tag suggestion model</p>
            
            <form id="uploadForm">
                <div class="file-upload">
                    <label for="lectures">📚 Lectures CSV</label>
                    <input type="file" id="lectures" name="lectures" accept=".csv" required>
                </div>
                
                <div class="file-upload">
                    <label for="labels">🏷️ Labels CSV</label>
                    <input type="file" id="labels" name="labels" accept=".csv" required>
                </div>
                
                <div class="file-upload">
                    <label for="lecture_labels">🔗 Lecture-Labels CSV (Junction Table)</label>
                    <input type="file" id="lecture_labels" name="lecture_labels" accept=".csv" required>
                </div>
                
                <button type="submit" class="btn" id="submitBtn">
                    Train Model
                </button>
            </form>
            
            <div id="message" class="message"></div>
        </div>
        
        <script>
            const form = document.getElementById('uploadForm');
            const submitBtn = document.getElementById('submitBtn');
            const message = document.getElementById('message');
            
            form.addEventListener('submit', async (e) => {
                e.preventDefault();
                
                submitBtn.disabled = true;
                submitBtn.innerHTML = '<span class="spinner"></span>Training...';
                message.style.display = 'none';
                
                const formData = new FormData();
                formData.append('lectures', document.getElementById('lectures').files[0]);
                formData.append('labels', document.getElementById('labels').files[0]);
                formData.append('lecture_labels', document.getElementById('lecture_labels').files[0]);
                
                try {
                    const response = await fetch('/train-csv', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    
                    if (response.ok) {
                        message.className = 'message success';
                        message.style.display = 'block';
                        message.innerHTML = `
                            <strong>✅ Training Complete!</strong>
                            <div class="stats">
                                <div class="stat">
                                    <div class="stat-value">${result.num_lectures || 0}</div>
                                    <div class="stat-label">Lectures Processed</div>
                                </div>
                                <div class="stat">
                                    <div class="stat-value">${result.num_prototypes || 0}</div>
                                    <div class="stat-label">Prototypes Created</div>
                                </div>
                                <div class="stat">
                                    <div class="stat-value">${result.num_tags || 0}</div>
                                    <div class="stat-label">Tags</div>
                                </div>
                                <div class="stat">
                                    <div class="stat-value">${result.low_data_tags || 0}</div>
                                    <div class="stat-label">Low-Data Tags</div>
                                </div>
                            </div>
                        `;
                        form.reset();
                    } else {
                        throw new Error(result.error || 'Training failed');
                    }
                } catch (error) {
                    message.className = 'message error';
                    message.style.display = 'block';
                    message.innerHTML = `<strong>❌ Error:</strong> ${error.message}`;
                } finally {
                    submitBtn.disabled = false;
                    submitBtn.innerHTML = 'Train Model';
                }
            });
        </script>
    </body>
    </html>
    """
    return html


@app.route('/', methods=['GET'])
def index():
    """API documentation homepage with payload examples."""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Tag Suggestions API</title>
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background: #f5f5f5;
                line-height: 1.6;
            }
            h1 { color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }
            h2 { color: #555; margin-top: 30px; }
            .endpoint {
                background: white;
                border-radius: 8px;
                padding: 20px;
                margin: 20px 0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .method {
                display: inline-block;
                padding: 4px 12px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 14px;
                margin-right: 10px;
            }
            .post { background: #4CAF50; color: white; }
            .get { background: #2196F3; color: white; }
            .path { font-family: 'Courier New', monospace; font-size: 18px; color: #333; }
            pre {
                background: #282c34;
                color: #abb2bf;
                padding: 15px;
                border-radius: 5px;
                overflow-x: auto;
                font-size: 13px;
            }
            .description { color: #666; margin: 10px 0; }
            .status {
                display: inline-block;
                padding: 8px 16px;
                border-radius: 20px;
                font-weight: bold;
                margin: 10px 0;
            }
            .status.loaded { background: #4CAF50; color: white; }
            .status.not-loaded { background: #f44336; color: white; }
            code { background: #e8e8e8; padding: 2px 6px; border-radius: 3px; font-family: 'Courier New', monospace; }
            .cta-button {
                display: inline-block;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 12px 24px;
                border-radius: 8px;
                text-decoration: none;
                font-weight: 600;
                margin: 20px 10px 20px 0;
                box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
                transition: transform 0.2s, box-shadow 0.2s;
            }
            .cta-button:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 16px rgba(102, 126, 234, 0.4);
            }
            .header-section {
                background: white;
                border-radius: 8px;
                padding: 20px;
                margin: 20px 0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
        </style>
    </head>
    <body>
        <h1>🏷️ Tag Suggestions API</h1>
        
        <div class="header-section">
            <p>AI-powered Hebrew lecture tagging using OpenAI embeddings and prototype learning.</p>
            <div class="status """ + ("loaded" if prototypes_loaded else "not-loaded") + """">
                """ + ("✓ Prototypes Loaded: " + str(len(prototype_knn.tag_prototypes)) if prototypes_loaded else "⚠ No Prototypes - Train First") + """
            </div>
            <a href="/train-ui" class="cta-button">🎓 Upload CSV & Train Model</a>
        </div>

        <h2>📋 API Endpoints</h2>

        <div class="endpoint">
            <span class="method post">POST</span>
            <span class="path">/train</span>
            <p class="description">Train prototypes from training data and save to KV store.</p>
            
            <h3>Request Example:</h3>
            <pre>{
  "lectures": [
    {
      "id": "lec_001",
      "lecture_title": "שיעור בתלמוד",
      "lecture_description": "עיון במסכת ברכות",
      "lecture_tag_ids": ["talmud", "gemara"]
    },
    {
      "id": "lec_002",
      "lecture_title": "קבלה ומיסטיקה",
      "lecture_description": "סודות הזוהר",
      "lecture_tag_ids": ["kabbalah", "zohar"]
    }
  ],
  "tags": {
    "talmud": {
      "tag_id": "talmud",
      "name_he": "תלמוד",
      "synonyms_he": "גמרא תלמוד בבלי"
    },
    "kabbalah": {
      "tag_id": "kabbalah",
      "name_he": "קבלה",
      "synonyms_he": "סודות מיסטיקה זוהר"
    }
  }
}</pre>
            
            <h3>Response Example:</h3>
            <pre>{
  "status": "success",
  "num_prototypes": 12,
  "num_lectures": 5,
  "num_tags": 12,
  "low_data_tags": 3
}</pre>
        </div>

        <div class="endpoint">
            <span class="method post">POST</span>
            <span class="path">/suggest-tags</span>
            <p class="description">Get tag suggestions for a single lecture based on pre-trained prototypes.</p>
            
            <h3>⚙️ Scoring Modes</h3>
            <p>Choose the right balance of quality, speed, and cost:</p>
            <ul>
                <li><strong>"fast"</strong>: Prototype similarity only (~1s, cheapest) - Good baseline quality</li>
                <li><strong>"full_quality"</strong>: Prototype + LLM arbiter (~2-3s, balanced) - ✓ Recommended default</li>
                <li><strong>"reasoning"</strong>: Pure LLM analysis (~5-7s, expensive) - Highest quality with Hebrew rationales</li>
            </ul>
            <p>Add <code>"scoring_mode": "full_quality"</code> to your request to override the default.</p>
            
            <h3>Request Example (v2):</h3>
            <pre>{
  "request_id": "c5f6f5f2-6c36-4b6f-9d6f-1d0b02b24a11",
  "model_version": "v1",
  "artifact_version": "labels-emb-2025-10-29",
  "scoring_mode": "full_quality",
  "lecture": {
    "id": "rec17SffStTL231k8",
    "title": "על חרדה והתמודדות",
    "description": "אסי עזר ומיכל עזר סטיין הם אחים... כלים יומיומיים להתמודדות.",
    "lecturer_id": "recasCfleyvOlrhkf",
    "lecturer_name": "אסי עזר ומיכל עזר סטיין",
    "lecturer_role": "אסי עזר, יוצר ומנחה; מיכל עזר סטיין, מאמנת אישית",
    "language": "he",
    "related_lectures": [
      { "id": "recR59LwPxi07sk6g", "title": "לחיות עם חרדה", "labels": ["lab_persona_celebs"] }
    ]
  },
  "labels": [
    { "id": "lab_persona_celebs", "name_he": "סלבס", "category": "Persona", "active": true },
    { "id": "lab_topic_mental_health", "name_he": "בריאות הנפש", "category": "Topic", "active": true },
    { "id": "lab_tone_personal", "name_he": "אישי", "category": "Tone", "active": true },
    { "id": "lab_format_talk", "name_he": "הרצאה", "category": "Format", "active": true }
  ]
}</pre>
            
            <h3>Response Example:</h3>
            <pre>{
  "request_id": "c5f6f5f2-6c36-4b6f-9d6f-1d0b02b24a11",
  "model_version": "v1",
  "artifact_version": "labels-emb-2025-10-29",
  "suggestions": [
    {
      "label_id": "lab_topic_mental_health",
      "category": "Topic",
      "confidence": 0.91,
      "reasons": ["desc_match"]
    },
    {
      "label_id": "lab_persona_celebs",
      "category": "Persona",
      "confidence": 0.84,
      "reasons": ["desc_match"]
    },
    {
      "label_id": "lab_tone_personal",
      "category": "Tone",
      "confidence": 0.67,
      "reasons": ["title_match"]
    }
  ]
}</pre>
        </div>

        <div class="endpoint">
            <span class="method post">POST</span>
            <span class="path">/reload-prototypes</span>
            <p class="description">Reload prototypes from KV store without restarting the server.</p>
            
            <h3>Request:</h3>
            <pre>No payload required</pre>
            
            <h3>Response Example:</h3>
            <pre>{
  "status": "success",
  "num_prototypes": 12
}</pre>
        </div>

        <div class="endpoint">
            <span class="method get">GET</span>
            <span class="path">/health</span>
            <p class="description">Health check endpoint to verify API status.</p>
            
            <h3>Response Example:</h3>
            <pre>{
  "status": "ok",
  "prototypes_loaded": true,
  "num_prototypes": 12
}</pre>
        </div>

        <h2>🚀 Quick Start</h2>
        <div class="endpoint">
            <h3>1. Train Prototypes (One-time Setup)</h3>
            <pre>curl -X POST http://localhost:5000/train \\
  -H "Content-Type: application/json" \\
  -d @training_data.json</pre>

            <h3>2. Get Tag Suggestions</h3>
            <pre>curl -X POST http://localhost:5000/suggest-tags \\
  -H "Content-Type: application/json" \\
  -d @request_data.json</pre>

            <h3>3. Check Health</h3>
            <pre>curl http://localhost:5000/health</pre>
        </div>

        <h2>📊 Data Requirements (v2 Format)</h2>
        <div class="endpoint">
            <h3>Lecture Object</h3>
            <ul>
                <li><code>id</code> (string, required): Unique lecture identifier (e.g., "rec17SffStTL231k8")</li>
                <li><code>title</code> (string, required): Hebrew lecture title</li>
                <li><code>description</code> (string, required): Hebrew lecture description</li>
                <li><code>lecturer_id</code> (string, optional): Lecturer identifier</li>
                <li><code>lecturer_name</code> (string, optional): Lecturer name</li>
                <li><code>lecturer_role</code> (string, optional): Lecturer role or bio</li>
                <li><code>language</code> (string, optional): Language code (e.g., "he")</li>
                <li><code>related_lectures</code> (array, optional): Related lecture references</li>
            </ul>

            <h3>Label Object</h3>
            <ul>
                <li><code>id</code> (string, required): Unique label identifier (e.g., "lab_topic_mental_health")</li>
                <li><code>name_he</code> (string, required): Hebrew label name</li>
                <li><code>category</code> (string, required): Label category (Topic, Persona, Tone, Format, Audience, etc.)</li>
                <li><code>active</code> (boolean, optional): Whether label is active (default: true)</li>
            </ul>

            <h3>Request Metadata</h3>
            <ul>
                <li><code>request_id</code> (string, optional): UUID for request tracking</li>
                <li><code>model_version</code> (string, optional): Model version (default: "v1")</li>
                <li><code>artifact_version</code> (string, optional): Training artifact version (e.g., "labels-emb-2025-10-29")</li>
            </ul>
        </div>

        <h2>⚙️ Scoring Modes Comparison</h2>
        <div class="endpoint">
            <table style="width: 100%; border-collapse: collapse; margin: 20px 0;">
                <tr style="background: #f0f0f0; border-bottom: 2px solid #ddd;">
                    <th style="padding: 10px; text-align: left;">Mode</th>
                    <th style="padding: 10px; text-align: left;">How It Works</th>
                    <th style="padding: 10px; text-align: left;">Speed</th>
                    <th style="padding: 10px; text-align: left;">Cost/Lecture</th>
                    <th style="padding: 10px; text-align: left;">Best For</th>
                </tr>
                <tr style="border-bottom: 1px solid #eee;">
                    <td style="padding: 10px;"><strong>fast</strong></td>
                    <td style="padding: 10px;">Vector similarity against prototypes</td>
                    <td style="padding: 10px;">~1s</td>
                    <td style="padding: 10px;">$0.0002</td>
                    <td style="padding: 10px;">Quick baseline, batch processing</td>
                </tr>
                <tr style="border-bottom: 1px solid #eee; background: #f9fff9;">
                    <td style="padding: 10px;"><strong>full_quality</strong> ✓</td>
                    <td style="padding: 10px;">Prototypes + LLM reviews borderline cases</td>
                    <td style="padding: 10px;">~2-3s</td>
                    <td style="padding: 10px;">$0.001-0.003</td>
                    <td style="padding: 10px;"><strong>Recommended default - best balance</strong></td>
                </tr>
                <tr style="border-bottom: 1px solid #eee;">
                    <td style="padding: 10px;"><strong>reasoning</strong></td>
                    <td style="padding: 10px;">GPT-4o-mini reads and analyzes lecture</td>
                    <td style="padding: 10px;">~5-7s</td>
                    <td style="padding: 10px;">$0.004-0.008</td>
                    <td style="padding: 10px;">Maximum quality, Hebrew explanations</td>
                </tr>
            </table>
            <p style="color: #666; font-size: 14px;">
                💡 <strong>Tip:</strong> Start with <code>full_quality</code> for the best balance. 
                Use <code>reasoning</code> when you need the absolute highest accuracy or want detailed Hebrew rationales.
                Use <code>fast</code> only for large-scale batch processing where speed matters most.
            </p>
        </div>

        <h2>⚙️ Configuration</h2>
        <div class="endpoint">
            <p><strong>Environment Variables:</strong></p>
            <ul>
                <li><code>OPENAI_API_KEY</code> (required): Your OpenAI API key</li>
                <li><code>PORT</code> (optional): Server port (default: 5000)</li>
                <li><code>SCORING_MODE</code> (optional): Default scoring mode ("fast", "full_quality", "reasoning"). Default: "full_quality"</li>
            </ul>
            
            <p><strong>Model Settings:</strong></p>
            <ul>
                <li>Embedding Model: <code>text-embedding-3-large</code> (3072 dimensions)</li>
                <li>LLM Model: <code>gpt-4o-mini</code> (for arbiter & reasoning modes)</li>
                <li>Target Precision: 0.90 (high precision for tag suggestions)</li>
                <li>Category-Aware Thresholds: Topic (0.65), Persona (0.60), Tone (0.55), Format (0.50), Audience (0.60)</li>
            </ul>
        </div>

        <footer style="text-align: center; margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #666;">
            <p>Tag Suggestions API v1.0 | Powered by OpenAI & Replit</p>
        </footer>
    </body>
    </html>
    """
    return html


if __name__ == '__main__':
    # Load prototypes on startup
    logger.info("Starting Tag Suggestions API...")
    load_prototypes_from_db()
    
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
