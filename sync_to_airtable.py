#!/usr/bin/env python3
"""
CLI command to sync approved tag suggestions to Airtable.
"""
import sys
import os
import logging
import argparse

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from sync_worker import SyncWorker
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(
        description='Sync approved tag suggestions to Airtable מרצים table'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Test mode: show what would be synced without actually updating Airtable'
    )
    parser.add_argument(
        '--test-connection',
        action='store_true',
        help='Test database and Airtable connections only'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize sync worker
        worker = SyncWorker(dry_run=args.dry_run)
        
        if args.test_connection:
            logger.info("\n" + "="*60)
            logger.info("TESTING CONNECTIONS")
            logger.info("="*60 + "\n")
            
            success = worker.test_connection()
            
            if success:
                logger.info("\n✓ All connections successful!")
                return 0
            else:
                logger.error("\n✗ Connection test failed")
                return 1
        
        # Run sync
        logger.info("\n" + "="*60)
        if args.dry_run:
            logger.info("DRY RUN MODE - No changes will be made to Airtable")
        else:
            logger.info("SYNCING APPROVED SUGGESTIONS TO AIRTABLE")
        logger.info("="*60 + "\n")
        
        summary = worker.sync_all()
        
        # Print detailed results
        if summary['results']:
            logger.info("\nDetailed Results:")
            logger.info("-" * 60)
            
            for result in summary['results']:
                if result.success:
                    status_icon = "✓"
                    status_color = "SUCCESS"
                else:
                    status_icon = "✗"
                    status_color = "FAILED"
                
                logger.info(
                    f"{status_icon} {result.lecturer_name} "
                    f"({result.lecturer_external_id})"
                )
                logger.info(f"   Tags before: {len(result.tags_before)}")
                logger.info(f"   Tags after:  {len(result.tags_after)}")
                logger.info(f"   Added:       {len(result.tags_added)} new tags")
                
                if result.tags_added:
                    logger.info(f"   New tags:    {', '.join(result.tags_added[:5])}" + 
                               (f" (+{len(result.tags_added) - 5} more)" if len(result.tags_added) > 5 else ""))
                
                if result.error:
                    logger.error(f"   Error:       {result.error}")
                
                logger.info("")
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("FINAL SUMMARY")
        logger.info("="*60)
        logger.info(f"Lecturers processed: {summary['total_lecturers']}")
        logger.info(f"Successful:          {summary['successful']}")
        logger.info(f"Failed:              {summary['failed']}")
        logger.info(f"Total tags added:    {summary['tags_added']}")
        logger.info("="*60 + "\n")
        
        if args.dry_run:
            logger.info("✓ Dry run completed successfully")
            logger.info("Run without --dry-run to actually sync to Airtable")
        elif summary['failed'] == 0:
            logger.info("✓ All lecturers synced successfully!")
        else:
            logger.warning(f"⚠ {summary['failed']} lecturer(s) failed to sync")
            return 1
        
        return 0
        
    except ValueError as e:
        logger.error(f"\n✗ Configuration error: {e}")
        logger.error("\nPlease ensure you have set the required environment variables:")
        logger.error("  - DATABASE_URL")
        logger.error("  - AIRTABLE_API_KEY (your personal access token)")
        logger.error("  - AIRTABLE_BASE_ID")
        return 1
    except Exception as e:
        logger.error(f"\n✗ Sync failed with error: {e}", exc_info=True)
        return 1

if __name__ == '__main__':
    sys.exit(main())
