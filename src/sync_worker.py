"""
Sync worker for pushing approved tag suggestions to Airtable.
"""
import logging
import os
from typing import List, Dict, Optional
from datetime import datetime
import psycopg2
from psycopg2.extras import RealDictCursor
from src.airtable_sync import AirtableSync, SyncResult
from src.database import DatabaseConnection

logger = logging.getLogger(__name__)

class SyncWorker:
    """Coordinates syncing approved suggestions from database to Airtable."""
    
    def __init__(
        self,
        database_url: Optional[str] = None,
        airtable_api_key: Optional[str] = None,
        airtable_base_id: Optional[str] = None,
        dry_run: bool = False
    ):
        """
        Initialize sync worker.
        
        Args:
            database_url: PostgreSQL connection string
            airtable_api_key: Airtable access token
            airtable_base_id: Airtable base ID
            dry_run: If True, don't actually update Airtable (default: False)
        """
        self.database_url = database_url or os.getenv('DATABASE_URL')
        self.dry_run = dry_run
        
        if not self.database_url:
            raise ValueError("DATABASE_URL not set")
        
        # Initialize Airtable sync
        self.airtable = AirtableSync(
            api_key=airtable_api_key,
            base_id=airtable_base_id
        )
        
        logger.info(f"Initialized sync worker (dry_run={dry_run})")
    
    def get_approved_suggestions_by_lecturer(self) -> Dict[str, Dict]:
        """
        Get all approved tag suggestions grouped by lecturer.
        
        Returns:
            Dict mapping lecturer_external_id to:
                {
                    'lecturer_name': str,
                    'lecturer_external_id': str,
                    'tag_ids': List[str],
                    'suggestion_ids': List[int]
                }
        """
        with DatabaseConnection(self.database_url) as db:
            with db.connection.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT 
                        l.lecturer_external_id,
                        l.lecturer_name,
                        s.tag_id,
                        s.suggestion_id,
                        s.status
                    FROM lecture_tag_suggestions s
                    JOIN enriched_lectures l ON s.lecture_id = l.id
                    WHERE s.status = 'approved'
                        AND l.lecturer_external_id IS NOT NULL
                    ORDER BY l.lecturer_external_id, s.tag_id
                """)
                
                rows = cur.fetchall()
        
        # Group by lecturer
        lecturers = {}
        for row in rows:
            lecturer_id = row['lecturer_external_id']
            
            if lecturer_id not in lecturers:
                lecturers[lecturer_id] = {
                    'lecturer_name': row['lecturer_name'],
                    'lecturer_external_id': lecturer_id,
                    'tag_ids': set(),
                    'suggestion_ids': []
                }
            
            # Add tag (using set to deduplicate)
            lecturers[lecturer_id]['tag_ids'].add(row['tag_id'])
            lecturers[lecturer_id]['suggestion_ids'].append(row['suggestion_id'])
        
        # Convert sets to lists
        for lecturer in lecturers.values():
            lecturer['tag_ids'] = list(lecturer['tag_ids'])
        
        logger.info(
            f"Found {len(lecturers)} lecturers with approved suggestions "
            f"({sum(len(l['tag_ids']) for l in lecturers.values())} unique tags)"
        )
        
        return lecturers
    
    def mark_suggestions_synced(
        self,
        suggestion_ids: List[int],
        success: bool,
        error: Optional[str] = None
    ):
        """
        Mark suggestions as synced in the database.
        
        Args:
            suggestion_ids: List of suggestion IDs to update
            success: Whether sync was successful
            error: Error message if sync failed
        """
        if self.dry_run:
            logger.info(f"[DRY RUN] Would mark {len(suggestion_ids)} suggestions as synced")
            return
        
        new_status = 'synced' if success else 'failed'
        
        with DatabaseConnection(self.database_url) as db:
            with db.connection.cursor() as cur:
                cur.execute("""
                    UPDATE lecture_tag_suggestions
                    SET status = %s,
                        synced_at = %s,
                        sync_error = %s
                    WHERE suggestion_id = ANY(%s)
                """, (new_status, datetime.now(), error, suggestion_ids))
                
                db.connection.commit()
        
        logger.info(f"Marked {len(suggestion_ids)} suggestions as '{new_status}'")
    
    def sync_all(self) -> Dict[str, any]:
        """
        Sync all approved suggestions to Airtable.
        
        Returns:
            Summary dict with statistics
        """
        # Get approved suggestions by lecturer
        lecturers_data = self.get_approved_suggestions_by_lecturer()
        
        if not lecturers_data:
            logger.info("No approved suggestions to sync")
            return {
                'total_lecturers': 0,
                'successful': 0,
                'failed': 0,
                'tags_added': 0,
                'results': []
            }
        
        results = []
        successful_count = 0
        failed_count = 0
        total_tags_added = 0
        
        for lecturer_id, lecturer_info in lecturers_data.items():
            lecturer_name = lecturer_info['lecturer_name']
            tag_ids = lecturer_info['tag_ids']
            suggestion_ids = lecturer_info['suggestion_ids']
            
            logger.info(
                f"Syncing {len(tag_ids)} tags for {lecturer_name} "
                f"(Airtable ID: {lecturer_id})"
            )
            
            if self.dry_run:
                logger.info(f"[DRY RUN] Would add tags: {tag_ids}")
                result = SyncResult(
                    lecturer_external_id=lecturer_id,
                    lecturer_name=lecturer_name,
                    tags_before=[],
                    tags_after=tag_ids,
                    tags_added=tag_ids,
                    success=True
                )
                successful_count += 1
                total_tags_added += len(tag_ids)
            else:
                # Sync to Airtable
                result = self.airtable.sync_lecturer(
                    lecturer_external_id=lecturer_id,
                    lecturer_name=lecturer_name,
                    approved_tag_ids=tag_ids
                )
                
                # Mark suggestions as synced/failed
                self.mark_suggestions_synced(
                    suggestion_ids,
                    success=result.success,
                    error=result.error
                )
                
                if result.success:
                    successful_count += 1
                    total_tags_added += len(result.tags_added)
                else:
                    failed_count += 1
            
            results.append(result)
        
        summary = {
            'total_lecturers': len(lecturers_data),
            'successful': successful_count,
            'failed': failed_count,
            'tags_added': total_tags_added,
            'results': results
        }
        
        logger.info(
            f"\n{'='*60}\n"
            f"Sync Summary:\n"
            f"  Lecturers processed: {summary['total_lecturers']}\n"
            f"  Successful: {summary['successful']}\n"
            f"  Failed: {summary['failed']}\n"
            f"  Tags added: {summary['tags_added']}\n"
            f"{'='*60}"
        )
        
        return summary
    
    def test_connection(self) -> bool:
        """Test both database and Airtable connections."""
        logger.info("Testing connections...")
        
        # Test database
        try:
            with DatabaseConnection(self.database_url) as db:
                with db.connection.cursor() as cur:
                    cur.execute("SELECT COUNT(*) FROM lecture_tag_suggestions WHERE status = 'approved'")
                    count = cur.fetchone()[0]
                    logger.info(f"✓ Database connection successful - {count} approved suggestions")
        except Exception as e:
            logger.error(f"✗ Database connection failed: {e}")
            return False
        
        # Test Airtable
        if not self.airtable.test_connection():
            return False
        
        logger.info("✓ All connections successful")
        return True
