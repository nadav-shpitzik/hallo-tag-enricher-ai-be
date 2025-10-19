import psycopg2
from psycopg2.extras import RealDictCursor
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class DatabaseConnection:
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.connection = None
    
    def __enter__(self):
        self.connection = psycopg2.connect(self.database_url)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.connection:
            self.connection.close()
    
    def fetch_lectures(self, only_untagged: bool = False) -> List[Dict[str, Any]]:
        if not self.connection:
            raise RuntimeError("Database connection not established")
        with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
            query = """
                SELECT 
                    id,
                    lecture_external_id,
                    lecture_title,
                    lecture_description,
                    lecturer_name,
                    lecture_tags,
                    lecture_tag_ids
                FROM enriched_lectures
                WHERE is_active = true 
                  AND soft_deleted = false
            """
            if only_untagged:
                query += """
                  AND (lecture_tag_ids IS NULL OR lecture_tag_ids = '[]'::jsonb OR jsonb_array_length(lecture_tag_ids) = 0)
                """
            query += "\nORDER BY id"
            
            cursor.execute(query)
            results = cursor.fetchall()
            tag_filter = "untagged" if only_untagged else "all active"
            logger.info(f"Fetched {len(results)} {tag_filter} lectures from database")
            return [dict(row) for row in results]
    
    def create_suggestions_table(self) -> None:
        """Create suggestions table - now handled by migrations, kept for backwards compatibility."""
        if not self.connection:
            raise RuntimeError("Database connection not established")
        logger.info("Suggestions table should exist from migration, skipping creation")
    
    def upsert_suggestions(self, suggestions: List[Dict[str, Any]]) -> None:
        if not self.connection:
            raise RuntimeError("Database connection not established")
        with self.connection.cursor() as cursor:
            query = """
                INSERT INTO lecture_tag_suggestions 
                    (lecture_id, lecture_external_id, tag_id, tag_name_he, score, rationale, model, sources)
                VALUES 
                    (%(lecture_id)s, %(lecture_external_id)s, %(tag_id)s, %(tag_name_he)s, 
                     %(score)s, %(rationale)s, %(model)s, %(sources)s)
                ON CONFLICT (lecture_id, tag_id) 
                DO UPDATE SET
                    score = EXCLUDED.score,
                    rationale = EXCLUDED.rationale,
                    model = EXCLUDED.model,
                    sources = EXCLUDED.sources,
                    created_at = CURRENT_TIMESTAMP
            """
            cursor.executemany(query, suggestions)
            self.connection.commit()
            logger.info(f"Upserted {len(suggestions)} tag suggestions to database")
    
    def create_event(self, suggestion_id: int, action: str, actor: Optional[str] = None, 
                     previous_status: Optional[str] = None, new_status: Optional[str] = None,
                     details: Optional[Dict[str, Any]] = None) -> None:
        """Create an audit event for a suggestion."""
        if not self.connection:
            raise RuntimeError("Database connection not established")
        
        import json
        with self.connection.cursor() as cursor:
            cursor.execute("""
                INSERT INTO suggestion_events 
                    (suggestion_id, action, actor, previous_status, new_status, details)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (suggestion_id, action, actor, previous_status, new_status, 
                  json.dumps(details) if details else None))
            self.connection.commit()
    
    def get_suggestions_by_status(self, status: str) -> List[Dict[str, Any]]:
        """Get all suggestions with a specific status."""
        if not self.connection:
            raise RuntimeError("Database connection not established")
        
        with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("""
                SELECT * FROM lecture_tag_suggestions
                WHERE status = %s
                ORDER BY created_at DESC
            """, (status,))
            return [dict(row) for row in cursor.fetchall()]
    
    def update_suggestion_status(self, suggestion_id: int, new_status: str, 
                                 actor: Optional[str] = None) -> bool:
        """Update suggestion status and create audit event."""
        if not self.connection:
            raise RuntimeError("Database connection not established")
        
        with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
            # Get current status
            cursor.execute("""
                SELECT status FROM lecture_tag_suggestions
                WHERE suggestion_id = %s
            """, (suggestion_id,))
            row = cursor.fetchone()
            if not row:
                return False
            
            previous_status = row['status']
            
            # Update status with approval metadata
            if new_status == 'approved':
                cursor.execute("""
                    UPDATE lecture_tag_suggestions
                    SET status = %s, approved_by = %s, approved_at = CURRENT_TIMESTAMP
                    WHERE suggestion_id = %s
                """, (new_status, actor, suggestion_id))
            elif new_status == 'synced':
                cursor.execute("""
                    UPDATE lecture_tag_suggestions
                    SET status = %s, synced_at = CURRENT_TIMESTAMP
                    WHERE suggestion_id = %s
                """, (new_status, suggestion_id))
            else:
                cursor.execute("""
                    UPDATE lecture_tag_suggestions
                    SET status = %s
                    WHERE suggestion_id = %s
                """, (new_status, suggestion_id))
            
            # Create audit event - map status to allowed actions
            action_map = {
                'approved': 'approve',
                'rejected': 'reject',
                'synced': 'sync_ok',
                'failed': 'sync_fail'
            }
            action = action_map.get(new_status, new_status)
            self.create_event(suggestion_id, action, actor, previous_status, new_status)
            
            self.connection.commit()
            logger.info(f"Updated suggestion {suggestion_id}: {previous_status} -> {new_status}")
            return True
    
    def enqueue_for_sync(self, suggestion_id: int) -> None:
        """Add a suggestion to the Airtable sync queue."""
        if not self.connection:
            raise RuntimeError("Database connection not established")
        
        with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
            # Get suggestion details
            cursor.execute("""
                SELECT lecture_external_id, tag_id 
                FROM lecture_tag_suggestions
                WHERE suggestion_id = %s
            """, (suggestion_id,))
            row = cursor.fetchone()
            if not row:
                raise ValueError(f"Suggestion {suggestion_id} not found")
            
            # Insert into sync queue (ON CONFLICT DO NOTHING for idempotency)
            cursor.execute("""
                INSERT INTO airtable_sync_items 
                    (lecture_external_id, tag_id, suggestion_id, status)
                VALUES (%s, %s, %s, 'queued')
                ON CONFLICT (lecture_external_id, tag_id) DO NOTHING
            """, (row['lecture_external_id'], row['tag_id'], suggestion_id))
            
            # Create enqueue event
            self.create_event(suggestion_id, 'enqueue', None, None, None)
            
            self.connection.commit()
            logger.info(f"Enqueued suggestion {suggestion_id} for Airtable sync")
