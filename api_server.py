#!/usr/bin/env python3
"""
Stateless API for tag suggestions.

Endpoints:
- POST /train: Train prototypes from training data and save to KV store
- POST /suggest-tags: Get tag suggestions for lectures  
- POST /reload-prototypes: Reload prototypes from KV store
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


def load_prototypes_from_kv():
    """Load prototypes from Replit KV store."""
    global prototypes_loaded, prototype_knn, tag_embeddings_cache, config
    
    try:
        if "prototypes" not in db.keys():
            logger.error("No prototypes found in KV store. Please train first.")
            return False
        
        prototypes_json = db["prototypes"]
        prototypes_data = json.loads(prototypes_json)
        
        # Initialize config
        config = Config()
        
        # Create PrototypeKNN instance
        prototype_knn = PrototypeKNN(config)
        
        # Load tag prototypes (convert lists back to numpy arrays)
        prototype_knn.tag_prototypes = {
            tag_id: np.array(proto, dtype=np.float32)
            for tag_id, proto in prototypes_data['tag_prototypes'].items()
        }
        
        # Load tag thresholds
        prototype_knn.tag_thresholds = prototypes_data['tag_thresholds']
        
        # Load tag stats
        prototype_knn.tag_stats = prototypes_data['tag_stats']
        
        # Load tag embeddings
        tag_embeddings_cache = {
            tag_id: np.array(emb, dtype=np.float32)
            for tag_id, emb in prototypes_data['tag_embeddings'].items()
        }
        
        prototypes_loaded = True
        logger.info(f"Loaded {len(prototype_knn.tag_prototypes)} prototypes from KV store")
        return True
        
    except Exception as e:
        logger.error(f"Error loading prototypes: {e}")
        import traceback
        traceback.print_exc()
        return False


def serialize_prototypes(prototype_knn_inst: PrototypeKNN, tag_embeddings: Dict[str, np.ndarray]) -> dict:
    """Convert prototype data to JSON-serializable format."""
    return {
        'tag_prototypes': {
            tag_id: proto.tolist() 
            for tag_id, proto in prototype_knn_inst.tag_prototypes.items()
        },
        'tag_thresholds': prototype_knn_inst.tag_thresholds,
        'tag_stats': prototype_knn_inst.tag_stats,
        'tag_embeddings': {
            tag_id: emb.tolist()
            for tag_id, emb in tag_embeddings.items()
        }
    }


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
    
    # Serialize and save to KV store
    prototypes_data = serialize_prototypes(train_prototype_knn, tag_embeddings)
    
    # Save to Replit KV store
    db["prototypes"] = json.dumps(prototypes_data)
    logger.info("Saved prototypes to Replit KV store")
    
    # Return summary
    return {
        'status': 'success',
        'num_prototypes': len(train_prototype_knn.tag_prototypes),
        'num_lectures': len(lectures),
        'num_tags': len(tags_data),
        'low_data_tags': sum(1 for s in train_prototype_knn.tag_stats.values() if s.get('is_low_data', False))
    }


def score_lectures(lectures: List[Dict], tags: Dict[str, Dict]) -> List[Dict]:
    """
    Score lectures against prototypes.
    
    Args:
        lectures: List of lecture dicts with id, lecture_title, lecture_description
        tags: Dict of tag data
    
    Returns:
        List of suggestions with lecture_id, tag_id, score, rationale
    """
    if not prototypes_loaded:
        raise RuntimeError("Prototypes not loaded. Please train first or reload prototypes.")
    
    # Generate embeddings for input lectures
    embeddings_gen = EmbeddingsGenerator(
        api_key=config.openai_api_key,
        model=config.embedding_model
    )
    
    lecture_embeddings = embeddings_gen.generate_lecture_embeddings(lectures)
    
    # Score each lecture
    all_suggestions = []
    
    for lecture in lectures:
        lecture_id = lecture.get('id')
        if lecture_id not in lecture_embeddings:
            continue
        
        lecture_embedding = lecture_embeddings[lecture_id]
        
        # Get scores from prototype KNN
        scores = prototype_knn.score_lecture(lecture_embedding, tag_embeddings_cache)
        
        # Convert to suggestions format
        for tag_id, score in scores.items():
            if tag_id in tags:
                suggestion = {
                    'lecture_id': lecture_id,
                    'tag_id': tag_id,
                    'tag_name_he': tags[tag_id].get('name_he', ''),
                    'score': float(score),
                    'rationale': f"Prototype similarity score: {score:.3f}"
                }
                all_suggestions.append(suggestion)
    
    logger.info(f"Generated {len(all_suggestions)} suggestions for {len(lectures)} lectures")
    return all_suggestions


@app.route('/suggest-tags', methods=['POST'])
def suggest_tags():
    """
    Main API endpoint for tag suggestions.
    
    Expected JSON format:
    {
        "lectures": [
            {
                "id": "lec_123",
                "lecture_title": "...",
                "lecture_description": "...",
                "lecturer_name": "..." (optional)
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
    
    Returns:
    {
        "suggestions": [
            {
                "lecture_id": "lec_123",
                "tag_id": "tag1",
                "tag_name_he": "...",
                "score": 0.85,
                "rationale": "..."
            }
        ]
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        lectures = data.get('lectures', [])
        tags = data.get('tags', {})
        
        if not lectures:
            return jsonify({'error': 'No lectures provided'}), 400
        
        if not tags:
            return jsonify({'error': 'No tags provided'}), 400
        
        # Score lectures
        suggestions = score_lectures(lectures, tags)
        
        return jsonify({
            'suggestions': suggestions,
            'num_lectures': len(lectures),
            'num_suggestions': len(suggestions)
        }), 200
        
    except Exception as e:
        logger.error(f"Error in suggest-tags: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/train', methods=['POST'])
def train():
    """
    Train prototypes from training data.
    
    Expected JSON format:
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
        
        result = train_from_data(training_data)
        
        # Automatically reload prototypes after training
        load_prototypes_from_kv()
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Training error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/reload-prototypes', methods=['POST'])
def reload_prototypes():
    """Reload prototypes from KV store without restarting server."""
    try:
        success = load_prototypes_from_kv()
        
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


@app.route('/', methods=['GET'])
def index():
    """API info endpoint."""
    return jsonify({
        'service': 'Tag Suggestions API',
        'version': '1.0',
        'endpoints': {
            'POST /train': 'Train prototypes from training data',
            'POST /suggest-tags': 'Get tag suggestions for lectures',
            'POST /reload-prototypes': 'Reload prototypes from KV store',
            'GET /health': 'Health check'
        },
        'prototypes_loaded': prototypes_loaded
    }), 200


if __name__ == '__main__':
    # Load prototypes on startup
    logger.info("Starting Tag Suggestions API...")
    load_prototypes_from_kv()
    
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
