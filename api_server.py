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


def score_lecture_v2(lecture: Dict, labels: List[Dict]) -> List[Dict]:
    """
    Score a single lecture against prototypes (v2 format).
    
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
    
    # Get scores from prototype KNN
    scores = prototype_knn.score_lecture(lecture_embedding, tag_embeddings_cache)
    
    # Create label lookup by id
    labels_by_id = {label['id']: label for label in labels if label.get('active', True)}
    
    # Convert to suggestions format
    suggestions = []
    for label_id, score in scores.items():
        if label_id in labels_by_id:
            label = labels_by_id[label_id]
            
            # Determine reasons based on score
            reasons = []
            if score >= 0.8:
                reasons.append("desc_match")
            elif score >= 0.6:
                reasons.append("title_match")
            else:
                reasons.append("cooccur")
            
            suggestion = {
                'label_id': label_id,
                'category': label.get('category', 'Unknown'),
                'confidence': float(score),
                'reasons': reasons
            }
            suggestions.append(suggestion)
    
    # Sort by confidence descending
    suggestions.sort(key=lambda x: x['confidence'], reverse=True)
    
    return suggestions


@app.route('/suggest-tags', methods=['POST'])
def suggest_tags():
    """
    Main API endpoint for tag suggestions (v2 format).
    
    Expected JSON format:
    {
        "request_id": "uuid",
        "model_version": "v1",
        "artifact_version": "labels-emb-2025-10-29",
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
        lecture = data.get('lecture')
        labels = data.get('labels', [])
        
        if not lecture:
            return jsonify({'error': 'No lecture provided'}), 400
        
        if not labels:
            return jsonify({'error': 'No labels provided'}), 400
        
        # Score lecture
        suggestions = score_lecture_v2(lecture, labels)
        
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
            
            # Convert labels array to tags dict
            tags = {}
            for label in labels:
                tags[label['id']] = {
                    'tag_id': label['id'],
                    'name_he': label.get('name_he', ''),
                    'synonyms_he': label.get('synonyms_he', '')
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
        </style>
    </head>
    <body>
        <h1>🏷️ Tag Suggestions API</h1>
        <p>AI-powered Hebrew lecture tagging using OpenAI embeddings and prototype learning.</p>
        <div class="status """ + ("loaded" if prototypes_loaded else "not-loaded") + """">
            """ + ("✓ Prototypes Loaded: " + str(len(prototype_knn.tag_prototypes)) if prototypes_loaded else "⚠ No Prototypes - Train First") + """
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
            <p class="description">Get tag suggestions for new lectures based on pre-trained prototypes.</p>
            
            <h3>Request Example:</h3>
            <pre>{
  "lectures": [
    {
      "id": "test_001",
      "lecture_title": "שיעור על הלכה",
      "lecture_description": "דיני שבת והלכות יומיות"
    },
    {
      "id": "test_002",
      "lecture_title": "פילוסופיה יהודית",
      "lecture_description": "מחשבת הרמב״ם והרמב״ן"
    }
  ],
  "tags": {
    "halakha": {
      "tag_id": "halakha",
      "name_he": "הלכה",
      "synonyms_he": "דינים משפט עברי"
    },
    "philosophy": {
      "tag_id": "philosophy",
      "name_he": "פילוסופיה",
      "synonyms_he": "מחשבה השקפה"
    }
  }
}</pre>
            
            <h3>Response Example:</h3>
            <pre>{
  "suggestions": [
    {
      "lecture_id": "test_001",
      "tag_id": "halakha",
      "tag_name_he": "הלכה",
      "score": 0.872,
      "rationale": "Prototype similarity score: 0.872"
    },
    {
      "lecture_id": "test_002",
      "tag_id": "philosophy",
      "tag_name_he": "פילוסופיה",
      "score": 0.815,
      "rationale": "Prototype similarity score: 0.815"
    }
  ],
  "num_lectures": 2,
  "num_suggestions": 2
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

        <h2>📊 Data Requirements</h2>
        <div class="endpoint">
            <h3>Lecture Object</h3>
            <ul>
                <li><code>id</code> (string): Unique lecture identifier</li>
                <li><code>lecture_title</code> (string): Hebrew lecture title</li>
                <li><code>lecture_description</code> (string): Hebrew lecture description</li>
                <li><code>lecture_tag_ids</code> (array, training only): Existing tag IDs for this lecture</li>
            </ul>

            <h3>Tag Object</h3>
            <ul>
                <li><code>tag_id</code> (string): Unique tag identifier</li>
                <li><code>name_he</code> (string): Hebrew tag name</li>
                <li><code>synonyms_he</code> (string): Hebrew synonyms (space-separated)</li>
            </ul>
        </div>

        <h2>⚙️ Configuration</h2>
        <div class="endpoint">
            <p><strong>Environment Variables:</strong></p>
            <ul>
                <li><code>OPENAI_API_KEY</code> (required): Your OpenAI API key</li>
                <li><code>PORT</code> (optional): Server port (default: 5000)</li>
            </ul>
            
            <p><strong>Model Settings:</strong></p>
            <ul>
                <li>Embedding Model: <code>text-embedding-3-large</code> (3072 dimensions)</li>
                <li>Target Precision: 0.90 (high precision for tag suggestions)</li>
                <li>Min Confidence: 0.60 (threshold for suggestions)</li>
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
    load_prototypes_from_kv()
    
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
