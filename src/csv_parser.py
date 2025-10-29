"""
CSV parser for training data uploads.

Handles lectures.csv, labels.csv, and lecture_labels.csv junction table.
"""

import csv
import io
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


def parse_csv_training_data(
    lectures_csv: bytes,
    labels_csv: bytes,
    lecture_labels_csv: bytes
) -> Dict[str, Any]:
    """
    Parse 3 CSV files and transform to training format.
    
    Args:
        lectures_csv: Lectures CSV file content
        labels_csv: Labels CSV file content
        lecture_labels_csv: Junction table CSV content
        
    Returns:
        Training data dict with 'lectures' and 'labels' keys
    """
    # Parse labels first
    labels_dict = _parse_labels_csv(labels_csv)
    
    # Parse lecture-label mappings
    lecture_label_map = _parse_lecture_labels_csv(lecture_labels_csv)
    
    # Parse lectures and attach their labels
    lectures_list = _parse_lectures_csv(lectures_csv, lecture_label_map)
    
    logger.info(f"Parsed {len(lectures_list)} lectures with {len(labels_dict)} labels")
    
    # Convert to training format expected by train_from_data
    # (uses 'labels' list format which will be converted in /train endpoint)
    return {
        'lectures': lectures_list,
        'labels': list(labels_dict.values())
    }


def _parse_labels_csv(csv_content: bytes) -> Dict[str, Dict]:
    """Parse labels CSV into dict keyed by label ID."""
    labels = {}
    
    csv_file = io.StringIO(csv_content.decode('utf-8'))
    reader = csv.DictReader(csv_file)
    
    for row in reader:
        label_id = row.get('airtable_id') or row.get('id')
        if not label_id:
            continue
            
        labels[label_id] = {
            'id': label_id,
            'name_he': row.get('name', ''),
            'category': _normalize_category(row.get('category', '')),
            'synonyms_he': '',
            'active': True
        }
    
    return labels


def _parse_lecture_labels_csv(csv_content: bytes) -> Dict[str, List[str]]:
    """Parse junction table into dict mapping lecture_id -> [label_ids]."""
    lecture_labels = {}
    
    csv_file = io.StringIO(csv_content.decode('utf-8'))
    reader = csv.DictReader(csv_file)
    
    for row in reader:
        lecture_id = row.get('lecture_id')
        label_id = row.get('label_id')
        
        if lecture_id and label_id:
            if lecture_id not in lecture_labels:
                lecture_labels[lecture_id] = []
            lecture_labels[lecture_id].append(label_id)
    
    return lecture_labels


def _parse_lectures_csv(
    csv_content: bytes,
    lecture_label_map: Dict[str, List[str]]
) -> List[Dict]:
    """Parse lectures CSV and attach labels from junction table."""
    lectures = []
    
    csv_file = io.StringIO(csv_content.decode('utf-8'))
    reader = csv.DictReader(csv_file)
    
    for row in reader:
        lecture_id = row.get('airtable_id') or row.get('id')
        if not lecture_id:
            continue
        
        # Get labels for this lecture
        label_ids = lecture_label_map.get(lecture_id, [])
        
        # Skip lectures with no labels (can't train on them)
        if not label_ids:
            continue
        
        lectures.append({
            'id': lecture_id,
            'title': row.get('title', ''),
            'description': row.get('description', ''),
            'lecturer_id': row.get('lecturer_id', ''),
            'label_ids': label_ids
        })
    
    return lectures


def _normalize_category(category: str) -> str:
    """Normalize Hebrew category names to English equivalents."""
    category_map = {
        'נושא': 'Topic',
        'קהל יעד': 'Audience',
        'פרסונה': 'Persona',
        'טון': 'Tone',
        'פורמט': 'Format'
    }
    return category_map.get(category, category or 'Topic')
