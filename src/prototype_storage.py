"""
PostgreSQL storage for tag prototypes.

Stores prototypes with versioning for better visibility and tracking.
"""

import os
import json
import logging
import numpy as np
from typing import Dict, Optional, Tuple
from datetime import datetime
import psycopg2
from psycopg2.extras import Json

logger = logging.getLogger(__name__)


class PrototypeStorage:
    """Manages prototype storage in PostgreSQL."""
    
    def __init__(self):
        """Initialize database connection."""
        self.db_url = os.getenv('DATABASE_URL')
        if not self.db_url:
            raise ValueError("DATABASE_URL environment variable not set")
        self._ensure_schema()
    
    def _get_connection(self):
        """Get database connection."""
        return psycopg2.connect(self.db_url)
    
    def _ensure_schema(self):
        """Create tables if they don't exist."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                # Prototype versions table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS prototype_versions (
                        id SERIAL PRIMARY KEY,
                        version_name VARCHAR(255) DEFAULT 'default',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        num_lectures INTEGER,
                        num_tags INTEGER,
                        num_prototypes INTEGER,
                        is_active BOOLEAN DEFAULT TRUE
                    )
                """)
                
                # Tag prototypes table (stores embeddings and metadata)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS tag_prototypes (
                        id SERIAL PRIMARY KEY,
                        version_id INTEGER REFERENCES prototype_versions(id) ON DELETE CASCADE,
                        tag_id VARCHAR(255) NOT NULL,
                        tag_name_he TEXT,
                        category VARCHAR(100),
                        prototype_vector JSONB NOT NULL,
                        threshold FLOAT,
                        num_examples INTEGER,
                        avg_similarity FLOAT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(version_id, tag_id)
                    )
                """)
                
                # Tag embeddings table (separate from prototypes)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS tag_embeddings (
                        id SERIAL PRIMARY KEY,
                        version_id INTEGER REFERENCES prototype_versions(id) ON DELETE CASCADE,
                        tag_id VARCHAR(255) NOT NULL,
                        embedding_vector JSONB NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(version_id, tag_id)
                    )
                """)
                
                # AI calls tracking table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS ai_calls (
                        id SERIAL PRIMARY KEY,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        request_id VARCHAR(255),
                        call_type VARCHAR(100) NOT NULL,
                        model VARCHAR(100) NOT NULL,
                        lecture_id VARCHAR(255),
                        prompt_messages JSONB NOT NULL,
                        response_content JSONB,
                        input_tokens INTEGER,
                        output_tokens INTEGER,
                        total_tokens INTEGER,
                        estimated_cost_usd FLOAT,
                        duration_ms FLOAT,
                        status VARCHAR(50),
                        error_message TEXT
                    )
                """)
                
                # Create indexes for faster lookups
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_tag_prototypes_version 
                    ON tag_prototypes(version_id)
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_tag_prototypes_tag_id 
                    ON tag_prototypes(tag_id)
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_tag_embeddings_version 
                    ON tag_embeddings(version_id)
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_ai_calls_request_id 
                    ON ai_calls(request_id)
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_ai_calls_created_at 
                    ON ai_calls(created_at)
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_ai_calls_call_type 
                    ON ai_calls(call_type)
                """)
                
                conn.commit()
                logger.info("Prototype storage schema ensured")
    
    def save_prototypes(
        self,
        tag_prototypes: Dict[str, np.ndarray],
        tag_thresholds: Dict[str, float],
        tag_stats: Dict[str, dict],
        tag_embeddings: Dict[str, np.ndarray],
        num_lectures: int,
        tags_data: Dict[str, dict] = None,
        version_name: str = 'default'
    ) -> int:
        """
        Save prototypes to database with versioning.
        
        Returns the version_id of the saved prototypes.
        """
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                # Deactivate previous versions with same name
                cur.execute("""
                    UPDATE prototype_versions 
                    SET is_active = FALSE 
                    WHERE version_name = %s AND is_active = TRUE
                """, (version_name,))
                
                # Create new version
                cur.execute("""
                    INSERT INTO prototype_versions 
                    (version_name, num_lectures, num_tags, num_prototypes)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id
                """, (
                    version_name,
                    num_lectures,
                    len(tag_prototypes),
                    len(tag_prototypes)
                ))
                version_id = cur.fetchone()[0]
                
                # Save tag prototypes
                for tag_id, prototype in tag_prototypes.items():
                    stats = tag_stats.get(tag_id, {})
                    threshold = tag_thresholds.get(tag_id, 0.5)
                    
                    # Get tag metadata from tags_data if available
                    tag_info = tags_data.get(tag_id, {}) if tags_data else {}
                    tag_name_he = tag_info.get('name_he', '')
                    category = tag_info.get('category', 'Unknown')
                    
                    cur.execute("""
                        INSERT INTO tag_prototypes 
                        (version_id, tag_id, tag_name_he, category, 
                         prototype_vector, threshold, num_examples, avg_similarity)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        version_id,
                        tag_id,
                        tag_name_he,
                        category,
                        Json(prototype.tolist()),
                        threshold,
                        stats.get('num_examples', 0),
                        stats.get('avg_similarity', 0.0)
                    ))
                
                # Save tag embeddings
                for tag_id, embedding in tag_embeddings.items():
                    cur.execute("""
                        INSERT INTO tag_embeddings 
                        (version_id, tag_id, embedding_vector)
                        VALUES (%s, %s, %s)
                    """, (
                        version_id,
                        tag_id,
                        Json(embedding.tolist())
                    ))
                
                conn.commit()
                logger.info(f"Saved {len(tag_prototypes)} prototypes as version {version_id}")
                return version_id
    
    def load_prototypes(
        self,
        version_name: str = 'default'
    ) -> Optional[Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, dict], Dict[str, np.ndarray]]]:
        """
        Load prototypes from database.
        
        Returns (tag_prototypes, tag_thresholds, tag_stats, tag_embeddings) or None.
        """
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                # Get active version
                cur.execute("""
                    SELECT id FROM prototype_versions 
                    WHERE version_name = %s AND is_active = TRUE
                    ORDER BY created_at DESC
                    LIMIT 1
                """, (version_name,))
                
                result = cur.fetchone()
                if not result:
                    logger.warning(f"No active version found for '{version_name}'")
                    return None
                
                version_id = result[0]
                
                # Load tag prototypes
                cur.execute("""
                    SELECT tag_id, tag_name_he, category, prototype_vector, 
                           threshold, num_examples, avg_similarity
                    FROM tag_prototypes
                    WHERE version_id = %s
                """, (version_id,))
                
                tag_prototypes = {}
                tag_thresholds = {}
                tag_stats = {}
                
                for row in cur.fetchall():
                    tag_id = row[0]
                    tag_prototypes[tag_id] = np.array(row[3], dtype=np.float32)
                    tag_thresholds[tag_id] = row[4]
                    tag_stats[tag_id] = {
                        'tag_name': row[1],
                        'category': row[2],
                        'num_examples': row[5],
                        'avg_similarity': row[6]
                    }
                
                # Load tag embeddings
                cur.execute("""
                    SELECT tag_id, embedding_vector
                    FROM tag_embeddings
                    WHERE version_id = %s
                """, (version_id,))
                
                tag_embeddings = {}
                for row in cur.fetchall():
                    tag_id = row[0]
                    tag_embeddings[tag_id] = np.array(row[1], dtype=np.float32)
                
                logger.info(f"Loaded {len(tag_prototypes)} prototypes from version {version_id}")
                return tag_prototypes, tag_thresholds, tag_stats, tag_embeddings
    
    def list_versions(self) -> list:
        """List all prototype versions with metadata."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, version_name, created_at, num_lectures, 
                           num_tags, num_prototypes, is_active
                    FROM prototype_versions
                    ORDER BY created_at DESC
                """)
                
                versions = []
                for row in cur.fetchall():
                    versions.append({
                        'id': row[0],
                        'version_name': row[1],
                        'created_at': row[2].isoformat() if row[2] else None,
                        'num_lectures': row[3],
                        'num_tags': row[4],
                        'num_prototypes': row[5],
                        'is_active': row[6]
                    })
                
                return versions
    
    def get_tag_info(self, tag_id: str, version_name: str = 'default') -> Optional[dict]:
        """Get detailed information about a specific tag."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT tp.tag_name_he, tp.category, tp.threshold, 
                           tp.num_examples, tp.avg_similarity,
                           jsonb_array_length(tp.prototype_vector) as vector_dim
                    FROM tag_prototypes tp
                    JOIN prototype_versions pv ON tp.version_id = pv.id
                    WHERE tp.tag_id = %s AND pv.version_name = %s AND pv.is_active = TRUE
                """, (tag_id, version_name))
                
                row = cur.fetchone()
                if not row:
                    return None
                
                return {
                    'tag_id': tag_id,
                    'tag_name_he': row[0],
                    'category': row[1],
                    'threshold': row[2],
                    'num_examples': row[3],
                    'avg_similarity': row[4],
                    'vector_dimension': row[5]
                }
