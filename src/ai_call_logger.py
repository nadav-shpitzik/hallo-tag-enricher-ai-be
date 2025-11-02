"""
Logger for tracking AI/LLM API calls in the database.
"""

import os
import json
import psycopg2
from psycopg2.extras import Json
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class AICallLogger:
    """Logs AI API calls to PostgreSQL for tracking and auditing."""
    
    def __init__(self):
        """Initialize database connection."""
        self.db_url = os.getenv('DATABASE_URL')
        if not self.db_url:
            logger.warning("DATABASE_URL not set - AI call logging disabled")
            self.enabled = False
        else:
            self.enabled = True
    
    def _get_connection(self):
        """Get database connection."""
        return psycopg2.connect(self.db_url)
    
    def log_call(
        self,
        call_type: str,
        model: str,
        prompt_messages: List[Dict[str, Any]],
        response_content: Optional[Dict[str, Any]] = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        total_tokens: int = 0,
        estimated_cost_usd: float = 0.0,
        duration_ms: float = 0.0,
        status: str = "success",
        error_message: Optional[str] = None,
        request_id: Optional[str] = None,
        lecture_id: Optional[str] = None
    ) -> Optional[int]:
        """
        Log an AI API call to the database.
        
        Args:
            call_type: Type of call (e.g., "reasoning_scorer", "llm_arbiter")
            model: Model name (e.g., "gpt-4o")
            prompt_messages: The messages sent to the API
            response_content: The parsed response from the API
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            total_tokens: Total tokens used
            estimated_cost_usd: Estimated cost in USD
            duration_ms: Duration in milliseconds
            status: Status of the call ("success" or "error")
            error_message: Error message if status is "error"
            request_id: Request ID for correlation
            lecture_id: Lecture ID if applicable
            
        Returns:
            The ID of the inserted record, or None if logging is disabled
        """
        if not self.enabled:
            return None
        
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO ai_calls (
                            created_at,
                            request_id,
                            call_type,
                            model,
                            lecture_id,
                            prompt_messages,
                            response_content,
                            input_tokens,
                            output_tokens,
                            total_tokens,
                            estimated_cost_usd,
                            duration_ms,
                            status,
                            error_message
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                        )
                        RETURNING id
                    """, (
                        datetime.now(),
                        request_id,
                        call_type,
                        model,
                        lecture_id,
                        Json(prompt_messages),
                        Json(response_content) if response_content else None,
                        input_tokens,
                        output_tokens,
                        total_tokens,
                        estimated_cost_usd,
                        duration_ms,
                        status,
                        error_message
                    ))
                    
                    call_id = cur.fetchone()[0]
                    conn.commit()
                    
                    logger.debug(f"Logged AI call {call_id} ({call_type}, {model})")
                    return call_id
                    
        except Exception as e:
            logger.error(f"Failed to log AI call to database: {e}")
            return None
    
    def get_recent_calls(
        self,
        call_type: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get recent AI calls from the database.
        
        Args:
            call_type: Filter by call type (optional)
            limit: Maximum number of records to return
            
        Returns:
            List of AI call records
        """
        if not self.enabled:
            return []
        
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    if call_type:
                        cur.execute("""
                            SELECT 
                                id, created_at, request_id, call_type, model,
                                lecture_id, input_tokens, output_tokens, total_tokens,
                                estimated_cost_usd, duration_ms, status
                            FROM ai_calls
                            WHERE call_type = %s
                            ORDER BY created_at DESC
                            LIMIT %s
                        """, (call_type, limit))
                    else:
                        cur.execute("""
                            SELECT 
                                id, created_at, request_id, call_type, model,
                                lecture_id, input_tokens, output_tokens, total_tokens,
                                estimated_cost_usd, duration_ms, status
                            FROM ai_calls
                            ORDER BY created_at DESC
                            LIMIT %s
                        """, (limit,))
                    
                    columns = [desc[0] for desc in cur.description]
                    return [dict(zip(columns, row)) for row in cur.fetchall()]
                    
        except Exception as e:
            logger.error(f"Failed to fetch AI calls from database: {e}")
            return []
    
    def get_call_details(self, call_id: int) -> Optional[Dict[str, Any]]:
        """
        Get full details of a specific AI call including prompts and responses.
        
        Args:
            call_id: The ID of the call to retrieve
            
        Returns:
            Full call record including prompt_messages and response_content
        """
        if not self.enabled:
            return None
        
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT 
                            id, created_at, request_id, call_type, model,
                            lecture_id, prompt_messages, response_content,
                            input_tokens, output_tokens, total_tokens,
                            estimated_cost_usd, duration_ms, status, error_message
                        FROM ai_calls
                        WHERE id = %s
                    """, (call_id,))
                    
                    row = cur.fetchone()
                    if not row:
                        return None
                    
                    columns = [desc[0] for desc in cur.description]
                    return dict(zip(columns, row))
                    
        except Exception as e:
            logger.error(f"Failed to fetch AI call details from database: {e}")
            return None
