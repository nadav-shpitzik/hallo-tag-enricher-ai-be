"""
Airtable sync module for updating lecturer tags.
Syncs approved tag suggestions from database to Airtable מרצים table.
"""
import logging
import os
import time
from typing import List, Dict, Optional, Tuple, Callable, Any
from pyairtable import Api
from dataclasses import dataclass

logger = logging.getLogger(__name__)

def retry_with_backoff(
    func: Callable,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    max_delay: float = 30.0
) -> Callable:
    """
    Decorator for retrying a function with exponential backoff.
    
    Args:
        func: Function to retry
        max_retries: Maximum number of retry attempts (default: 3)
        initial_delay: Initial delay in seconds (default: 1.0)
        backoff_factor: Multiplier for delay between retries (default: 2.0)
        max_delay: Maximum delay between retries in seconds (default: 30.0)
    
    Returns:
        Wrapped function with retry logic
    """
    def wrapper(*args, **kwargs) -> Any:
        delay = initial_delay
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                if attempt == max_retries:
                    logger.error(
                        f"Failed after {max_retries + 1} attempts: {func.__name__}. "
                        f"Last error: {e}"
                    )
                    raise
                
                logger.warning(
                    f"Attempt {attempt + 1}/{max_retries + 1} failed for {func.__name__}: {e}. "
                    f"Retrying in {delay:.1f}s..."
                )
                
                time.sleep(delay)
                delay = min(delay * backoff_factor, max_delay)
        
        raise last_exception
    
    return wrapper

@dataclass
class SyncResult:
    """Result of syncing a single lecturer."""
    lecturer_external_id: str
    lecturer_name: str
    tags_before: List[str]
    tags_after: List[str]
    tags_added: List[str]
    success: bool
    error: Optional[str] = None

class AirtableSync:
    """Handles syncing approved tag suggestions to Airtable."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_id: Optional[str] = None,
        table_name: str = "מרצים"
    ):
        """
        Initialize Airtable sync client.
        
        Args:
            api_key: Airtable personal access token (default: AIRTABLE_API_KEY env var)
            base_id: Airtable base ID (default: AIRTABLE_BASE_ID env var)
            table_name: Name of the lecturers table (default: "מרצים")
        """
        self.api_key = api_key or os.getenv('AIRTABLE_API_KEY')
        self.base_id = base_id or os.getenv('AIRTABLE_BASE_ID')
        self.table_name = table_name
        
        if not self.api_key:
            raise ValueError(
                "AIRTABLE_API_KEY not set. "
                "Get your personal access token from: https://airtable.com/create/tokens"
            )
        
        if not self.base_id:
            raise ValueError(
                "AIRTABLE_BASE_ID not set. "
                "Find it in your Airtable base URL (starts with 'app')"
            )
        
        # Initialize API client
        self.api = Api(self.api_key)
        self.table = self.api.table(self.base_id, self.table_name)
        
        logger.info(f"Initialized Airtable sync for base {self.base_id}, table '{self.table_name}'")
    
    def get_lecturer_record(self, lecturer_external_id: str) -> Optional[Dict]:
        """
        Get a lecturer record from Airtable by their external ID.
        
        Args:
            lecturer_external_id: Airtable record ID (e.g., "recXXXXXXXXXXXXXX")
        
        Returns:
            Airtable record dict or None if not found
        """
        @retry_with_backoff
        def _get_with_retry():
            return self.table.get(lecturer_external_id)
        
        try:
            record = _get_with_retry()
            return record
        except Exception as e:
            logger.warning(f"Lecturer {lecturer_external_id} not found in Airtable after retries: {e}")
            return None
    
    def get_current_tags(self, record: Dict) -> List[str]:
        """
        Extract current tags from a lecturer record.
        
        Args:
            record: Airtable record dict
        
        Returns:
            List of current tag IDs (from תגיות field)
        """
        fields = record.get('fields', {})
        
        # Try both possible field names
        tags = fields.get('תגיות', fields.get('tags', []))
        
        if tags is None:
            return []
        
        # Airtable linked records are lists of record IDs
        if isinstance(tags, list):
            return tags
        
        return []
    
    def update_lecturer_tags(
        self,
        lecturer_external_id: str,
        new_tags: List[str],
        preserve_existing: bool = True
    ) -> Tuple[bool, Optional[str]]:
        """
        Update tags for a lecturer in Airtable.
        
        Args:
            lecturer_external_id: Airtable record ID
            new_tags: List of tag IDs to add
            preserve_existing: If True, use set-union (default). If False, replace all tags.
        
        Returns:
            Tuple of (success: bool, error_message: Optional[str])
        """
        try:
            # Get current record
            record = self.get_lecturer_record(lecturer_external_id)
            if not record:
                return False, f"Lecturer {lecturer_external_id} not found in Airtable"
            
            # Get current tags
            current_tags = self.get_current_tags(record)
            
            # Compute final tags (set-union if preserving, else replace)
            if preserve_existing:
                final_tags = list(set(current_tags) | set(new_tags))
            else:
                final_tags = new_tags
            
            # Only update if there are changes
            if set(final_tags) == set(current_tags):
                logger.debug(f"No tag changes for lecturer {lecturer_external_id}")
                return True, None
            
            # Update Airtable with retry logic
            @retry_with_backoff
            def _update_with_retry():
                return self.table.update(lecturer_external_id, {'תגיות': final_tags})
            
            _update_with_retry()
            
            added_count = len(set(final_tags) - set(current_tags))
            logger.info(
                f"Updated lecturer {lecturer_external_id}: "
                f"{len(current_tags)} → {len(final_tags)} tags (+{added_count} new)"
            )
            
            return True, None
            
        except Exception as e:
            error_msg = f"Failed to update lecturer {lecturer_external_id}: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def sync_lecturer(
        self,
        lecturer_external_id: str,
        lecturer_name: str,
        approved_tag_ids: List[str]
    ) -> SyncResult:
        """
        Sync approved tags for a single lecturer.
        
        Args:
            lecturer_external_id: Airtable record ID
            lecturer_name: Lecturer's name (for logging)
            approved_tag_ids: List of approved tag IDs to add
        
        Returns:
            SyncResult with before/after state
        """
        # Get current state
        record = self.get_lecturer_record(lecturer_external_id)
        if not record:
            return SyncResult(
                lecturer_external_id=lecturer_external_id,
                lecturer_name=lecturer_name,
                tags_before=[],
                tags_after=[],
                tags_added=[],
                success=False,
                error=f"Lecturer not found in Airtable"
            )
        
        tags_before = self.get_current_tags(record)
        
        # Compute tags to add (only new ones)
        tags_to_add = [tag for tag in approved_tag_ids if tag not in tags_before]
        
        if not tags_to_add:
            logger.info(f"No new tags to add for {lecturer_name} - all already exist")
            return SyncResult(
                lecturer_external_id=lecturer_external_id,
                lecturer_name=lecturer_name,
                tags_before=tags_before,
                tags_after=tags_before,
                tags_added=[],
                success=True
            )
        
        # Update Airtable with set-union
        success, error = self.update_lecturer_tags(
            lecturer_external_id,
            approved_tag_ids,
            preserve_existing=True
        )
        
        if success:
            tags_after = list(set(tags_before) | set(approved_tag_ids))
            return SyncResult(
                lecturer_external_id=lecturer_external_id,
                lecturer_name=lecturer_name,
                tags_before=tags_before,
                tags_after=tags_after,
                tags_added=tags_to_add,
                success=True
            )
        else:
            return SyncResult(
                lecturer_external_id=lecturer_external_id,
                lecturer_name=lecturer_name,
                tags_before=tags_before,
                tags_after=tags_before,
                tags_added=[],
                success=False,
                error=error
            )
    
    def test_connection(self) -> bool:
        """
        Test connection to Airtable by fetching one record.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Try to fetch first record
            records = self.table.all(max_records=1)
            logger.info(f"✓ Airtable connection successful - found {len(records)} records in '{self.table_name}'")
            return True
        except Exception as e:
            logger.error(f"✗ Airtable connection failed: {e}")
            return False
