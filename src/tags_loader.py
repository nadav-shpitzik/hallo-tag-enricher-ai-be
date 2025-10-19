import pandas as pd
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class TagsLoader:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.tags = {}
    
    def load(self) -> Dict[str, Dict[str, str]]:
        df = pd.read_csv(self.csv_path)
        
        df.columns = df.columns.str.strip().str.lower()
        
        required_columns = ['tag_id', 'name_he']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        df['tag_id'] = df['tag_id'].astype(str).str.strip()
        df['name_he'] = df['name_he'].astype(str).str.strip()
        
        if 'synonyms_he' in df.columns:
            df['synonyms_he'] = df['synonyms_he'].fillna('').astype(str).str.strip()
        else:
            df['synonyms_he'] = ''
        
        self.tags = {}
        for _, row in df.iterrows():
            tag_id = str(row['tag_id'])
            name_he = str(row['name_he'])
            synonyms_he = str(row['synonyms_he'])
            self.tags[tag_id] = {
                'tag_id': tag_id,
                'name_he': name_he,
                'synonyms_he': synonyms_he,
                'label_text': self._create_label_text(name_he, synonyms_he)
            }
        
        logger.info(f"Loaded {len(self.tags)} tags from {self.csv_path}")
        return self.tags
    
    def _create_label_text(self, name_he: str, synonyms_he: str) -> str:
        text = f"תגית: {name_he}"
        if synonyms_he:
            synonyms_clean = synonyms_he.replace(',', ' | ').replace(';', ' | ')
            text += f" | נרדפות: {synonyms_clean}"
        return text
    
    def get_tag_ids(self) -> List[str]:
        return list(self.tags.keys())
    
    def get_tag_label_texts(self) -> Dict[str, str]:
        return {tag_id: data['label_text'] for tag_id, data in self.tags.items()}
    
    def get_tag_name(self, tag_id: str) -> str:
        return self.tags.get(tag_id, {}).get('name_he', tag_id)
