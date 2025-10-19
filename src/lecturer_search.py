import logging
import json
import os
from typing import Dict, Optional
from openai import OpenAI

logger = logging.getLogger(__name__)

class LecturerSearchService:
    def __init__(self):
        self.client = OpenAI()
        self.cache_file = "output/lecturer_cache.json"
        self.cache = self._load_cache()
    
    def _load_cache(self) -> Dict[str, str]:
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load lecturer cache: {e}")
                return {}
        return {}
    
    def _save_cache(self):
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)
    
    def get_lecturer_profile(self, lecturer_name: Optional[str]) -> Optional[str]:
        if not lecturer_name or not lecturer_name.strip():
            return None
        
        lecturer_key = lecturer_name.strip().lower()
        
        if lecturer_key in self.cache:
            logger.debug(f"Lecturer profile found in cache: {lecturer_name}")
            return self.cache[lecturer_key]
        
        logger.info(f"Searching for lecturer: {lecturer_name}")
        profile = self._search_lecturer(lecturer_name)
        
        if profile:
            self.cache[lecturer_key] = profile
            self._save_cache()
        
        return profile
    
    def _search_lecturer(self, lecturer_name: str) -> Optional[str]:
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": """אתה עוזר מחקר שמתמחה במציאת מידע על מרצים ומרצות בישראל.
משימתך: לספק סיכום קצר (2-3 משפטים) על התחומים והמומחיות של המרצה.
התמקד בנושאים הרלוונטיים להרצאות: תחומי עניין, מקצועות, התמחות, רקע מקצועי.
אם אינך יודע או לא בטוח - ציין זאת בבירור."""
                    },
                    {
                        "role": "user",
                        "content": f"מה תחומי המומחיות והעניין של המרצה/ת: {lecturer_name}?"
                    }
                ],
                temperature=0.3,
                max_tokens=200
            )
            
            profile = response.choices[0].message.content.strip()
            
            if "לא יודע" in profile or "אין לי מידע" in profile or "לא בטוח" in profile:
                return None
            
            return profile
            
        except Exception as e:
            logger.error(f"Error searching for lecturer {lecturer_name}: {e}")
            return None
    
    def get_bulk_profiles(self, lecturer_names: list) -> Dict[str, Optional[str]]:
        unique_names = list(set(lecturer_names))
        profiles = {}
        
        for name in unique_names:
            profiles[name] = self.get_lecturer_profile(name)
        
        logger.info(f"Retrieved {sum(1 for p in profiles.values() if p)} profiles for {len(unique_names)} unique lecturers")
        return profiles
