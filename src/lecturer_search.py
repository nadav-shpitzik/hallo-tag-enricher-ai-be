"""
Lecturer bio search service with database caching.

Uses GPT-4o-mini to search for lecturer biographies and caches results
in PostgreSQL for fast subsequent lookups.
"""

import logging
import os
from typing import Optional, Tuple
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
        self.search_model = "gpt-4o"  # Better accuracy for bio search
        self.validation_model = "gpt-4o-mini"  # Fast validation
        self.database_url = os.getenv('DATABASE_URL')
        
    def get_lecturer_profile(
        self, 
        lecturer_id: Optional[str] = None,
        lecturer_name: Optional[str] = None,
        lecture_description: Optional[str] = None
    ) -> Optional[str]:
        """
        Get lecturer bio, checking database cache first.
        
        Args:
            lecturer_id: Unique lecturer identifier (preferred)
            lecturer_name: Lecturer name (fallback)
            lecture_description: Lecture content to validate bio against
            
        Returns:
            Bio text or None if not found or validation failed
        """
        if not lecturer_id and not lecturer_name:
            logger.warning("No lecturer_id or lecturer_name provided")
            return None
        
        # Try database cache first
        cached_bio, cache_hit = self._get_from_cache(lecturer_id, lecturer_name)
        if cache_hit:
            logger.info(f"Lecturer bio cache hit: {lecturer_id or lecturer_name} (bio: {'found' if cached_bio else 'not found'})")
            return cached_bio
        
        # Not in cache - search using LLM
        if not lecturer_name:
            logger.warning(f"Cannot search for bio without lecturer_name (ID: {lecturer_id})")
            return None
            
        logger.info(f"Searching for bio: {lecturer_name}")
        bio = self._search_with_llm(lecturer_name)
        
        # Validate bio against lecture description if both available
        if bio and lecture_description:
            is_valid = self._validate_bio_with_lecture(bio, lecturer_name, lecture_description)
            if not is_valid:
                logger.warning(f"Bio validation failed for {lecturer_name} - not caching")
                return None  # Don't use or cache incorrect bio
        
        # Save to cache (even if None - to avoid repeated searches)
        if lecturer_id:
            self._save_to_cache(lecturer_id, lecturer_name, bio)
        
        return bio
    
    def _get_from_cache(
        self,
        lecturer_id: Optional[str],
        lecturer_name: Optional[str]
    ) -> Tuple[Optional[str], bool]:
        """
        Check database for cached bio.
        
        Returns:
            (bio_text, cache_hit) - bio_text can be None if searched but not found,
                                    cache_hit is True if we found a cached entry
        """
        if not self.database_url:
            return (None, False)
            
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
            
            # Return (bio, cache_hit)
            if result:
                return (result[0], True)  # Cache hit - bio could be None if not found
            return (None, False)  # Cache miss - not in cache at all
            
        except Exception as e:
            logger.error(f"Error fetching from cache: {e}")
            return (None, False)
    
    def _save_to_cache(
        self,
        lecturer_id: str,
        lecturer_name: str,
        bio_text: Optional[str]
    ) -> None:
        """Save bio to database cache (None if not found)."""
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
                (lecturer_id, lecturer_name, bio_text, datetime.now(), f'gpt-search:{self.search_model}')
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
                model=self.search_model,
                messages=[
                    {
                        "role": "system",
                        "content": """You are a research assistant helping to find information about lecturers and speakers.

When given a name, provide a concise professional background (2-3 sentences) focusing on:
- Their main expertise or field
- Professional background or achievements
- Areas they teach or speak about

Write in Hebrew. If the person is well-known, provide relevant details. Make a reasonable attempt even with limited information - better to provide some context than none."""
                    },
                    {
                        "role": "user",
                        "content": f"ספר לי בקצרה על: {lecturer_name}"
                    }
                ],
                temperature=0.3,
                max_tokens=200
            )
            
            bio = response.choices[0].message.content.strip()
            
            # Return the bio (LLM will provide context even with limited info)
            if bio and len(bio) > 10:  # Basic sanity check
                logger.info(f"Found bio for {lecturer_name}: {bio[:100]}...")
                return bio
            else:
                logger.info(f"No bio generated for: {lecturer_name}")
                return None
            
        except Exception as e:
            logger.error(f"Error searching with LLM: {e}")
            return None
    
    def _validate_bio_with_lecture(
        self,
        bio: str,
        lecturer_name: str,
        lecture_description: str
    ) -> bool:
        """
        Validate that the bio makes sense with the lecture description.
        
        Prevents caching incorrect bios (e.g., wrong person with same name).
        
        Args:
            bio: The lecturer bio to validate
            lecturer_name: Lecturer name
            lecture_description: Lecture content
            
        Returns:
            True if bio seems consistent with lecture, False otherwise
        """
        try:
            response = self.client.chat.completions.create(
                model=self.validation_model,
                messages=[
                    {
                        "role": "system",
                        "content": """You are validating if a lecturer bio matches a lecture description.

Check if the bio's expertise/field is consistent with the lecture topic. 
If there's a clear mismatch (e.g., bio says "tech entrepreneur" but lecture is about sports/judo), return FALSE.
If bio could reasonably match or is neutral, return TRUE.

Respond ONLY with TRUE or FALSE."""
                    },
                    {
                        "role": "user",
                        "content": f"""Bio: {bio}

Lecture: {lecture_description}

Does this bio make sense for someone giving this lecture?"""
                    }
                ],
                temperature=0,
                max_tokens=10
            )
            
            result = response.choices[0].message.content.strip().upper()
            is_valid = "TRUE" in result
            
            logger.info(f"Bio validation for {lecturer_name}: {result} -> {is_valid}")
            return is_valid
            
        except Exception as e:
            logger.error(f"Error validating bio: {e}")
            # On error, assume valid (don't block valid bios due to validation failure)
            return True
