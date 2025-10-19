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
    
    def fetch_lectures(self) -> List[Dict[str, Any]]:
        if not self.connection:
            raise RuntimeError("Database connection not established")
        with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
            query = """
                SELECT 
                    id,
                    lecture_external_id,
                    lecture_title AS name,
                    lecture_description AS description,
                    lecturer_name,
                    lecture_tags,
                    lecture_tag_ids
                FROM enriched_lectures
                WHERE is_active = true 
                  AND soft_deleted = false
                ORDER BY id
            """
            cursor.execute(query)
            results = cursor.fetchall()
            logger.info(f"Fetched {len(results)} active lectures from database")
            return [dict(row) for row in results]
    
    def create_suggestions_table(self) -> None:
        if not self.connection:
            raise RuntimeError("Database connection not established")
        with self.connection.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS lecture_tag_suggestions (
                    suggestion_id       BIGSERIAL PRIMARY KEY,
                    lecture_id          BIGINT NOT NULL REFERENCES enriched_lectures(id) ON DELETE CASCADE,
                    lecture_external_id VARCHAR NOT NULL,
                    tag_id              VARCHAR NOT NULL,
                    tag_name_he         VARCHAR NOT NULL,
                    score               NUMERIC(5,4) NOT NULL,
                    rationale           TEXT,
                    sources             JSONB DEFAULT '["title","description"]',
                    model               TEXT NOT NULL,
                    status              VARCHAR NOT NULL DEFAULT 'pending',
                    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE (lecture_id, tag_id)
                )
            """)
            self.connection.commit()
            logger.info("Created/verified lecture_tag_suggestions table")
    
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
