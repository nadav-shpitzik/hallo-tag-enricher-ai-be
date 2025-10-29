"""
Lecturer bio search service with database caching.

Uses GPT-4o-mini to search for lecturer biographies and caches results
in PostgreSQL for fast subsequent lookups.
"""

import logging
import os
from typing import Optional
from datetime import datetime
import psycopg2
from openai import OpenAI

logger = logging.getLogger(__name__)


class LecturerSearchService:
    """Service for fetching and caching lecturer biographies."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the lecturer search service.
        
        Args:
            api_key: OpenAI API key (defaults to env var)
        """
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4o-mini"
        self.database_url = os.getenv('DATABASE_URL')
        
    def get_lecturer_profile(
        self, 
        lecturer_id: Optional[str] = None,
        lecturer_name: Optional[str] = None
    ) -> Optional[str]:
        """
        Get lecturer bio, checking database cache first.
        
        Args:
            lecturer_id: Unique lecturer identifier (preferred)
            lecturer_name: Lecturer name (fallback)
            
        Returns:
            Bio text or None if not found
        """
        if not lecturer_id and not lecturer_name:
            logger.warning("No lecturer_id or lecturer_name provided")
            return None
        
        # Try database cache first
        cached_bio = self._get_from_cache(lecturer_id, lecturer_name)
        if cached_bio:
            logger.info(f"Lecturer bio found in cache: {lecturer_id or lecturer_name}")
            return cached_bio
        
        # Not in cache - search using LLM
        if not lecturer_name:
            logger.warning(f"Cannot search for bio without lecturer_name (ID: {lecturer_id})")
            return None
            
        logger.info(f"Searching for bio: {lecturer_name}")
        bio = self._search_with_llm(lecturer_name)
        
        # Save to cache if found
        if bio and lecturer_id:
            self._save_to_cache(lecturer_id, lecturer_name, bio)
        
        return bio
    
    def _get_from_cache(
        self,
        lecturer_id: Optional[str],
        lecturer_name: Optional[str]
    ) -> Optional[str]:
        """Check database for cached bio."""
        if not self.database_url:
            return None
            
        try:
            conn = psycopg2.connect(self.database_url)
            cursor = conn.cursor()
            
            # Try by ID first, then by name
            if lecturer_id:
                cursor.execute(
                    "SELECT bio_text FROM lecturer_bios WHERE lecturer_id = %s",
                    (lecturer_id,)
                )
            elif lecturer_name:
                cursor.execute(
                    "SELECT bio_text FROM lecturer_bios WHERE lecturer_name = %s LIMIT 1",
                    (lecturer_name,)
                )
            else:
                return None
            
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            
            return result[0] if result else None
            
        except Exception as e:
            logger.error(f"Error fetching from cache: {e}")
            return None
    
    def _save_to_cache(
        self,
        lecturer_id: str,
        lecturer_name: str,
        bio_text: str
    ) -> None:
        """Save bio to database cache."""
        if not self.database_url:
            return
            
        try:
            conn = psycopg2.connect(self.database_url)
            cursor = conn.cursor()
            
            cursor.execute(
                """
                INSERT INTO lecturer_bios (lecturer_id, lecturer_name, bio_text, searched_at, source)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (lecturer_id) 
                DO UPDATE SET 
                    lecturer_name = EXCLUDED.lecturer_name,
                    bio_text = EXCLUDED.bio_text,
                    searched_at = EXCLUDED.searched_at
                """,
                (lecturer_id, lecturer_name, bio_text, datetime.now(), f'gpt-search:{self.model}')
            )
            
            conn.commit()
            cursor.close()
            conn.close()
            
            logger.info(f"Saved bio to cache: {lecturer_id}")
            
        except Exception as e:
            logger.error(f"Error saving to cache: {e}")
    
    def _search_with_llm(self, lecturer_name: str) -> Optional[str]:
        """
        Search for lecturer bio using GPT-4o-mini.
        
        Args:
            lecturer_name: Name of lecturer to search for
            
        Returns:
            Bio summary or None if not found
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": """You are a research assistant helping to find information about lecturers and speakers.
                        
When given a lecturer's name, search for their professional background, expertise, and areas of teaching.
Focus on:
- Their main topics of expertise
- Academic background
- Teaching style or approach
- Notable works or contributions

Keep the response concise (2-3 sentences) and focus on information relevant to understanding their lectures.
If you cannot find reliable information, respond with "לא נמצא מידע" (information not found)."""
                    },
                    {
                        "role": "user",
                        "content": f"מצא מידע מקצועי על המרצה: {lecturer_name}"
                    }
                ],
                temperature=0.3,
                max_tokens=200
            )
            
            bio = response.choices[0].message.content.strip()
            
            # Check if LLM couldn't find info
            if "לא נמצא מידע" in bio or "information not found" in bio.lower():
                logger.info(f"No reliable info found for: {lecturer_name}")
                return None
            
            logger.info(f"Found bio for {lecturer_name}: {bio[:100]}...")
            return bio
            
        except Exception as e:
            logger.error(f"Error searching with LLM: {e}")
            return None
