from openai import OpenAI
from typing import List, Dict, Set
import logging
import json

logger = logging.getLogger(__name__)


class LLMArbiter:
    def __init__(self, api_key: str, config):
        self.client = OpenAI(api_key=api_key)
        self.config = config
    
    def refine_suggestions(
        self,
        lecture_title: str,
        lecture_description: str,
        candidate_tags: Dict[str, Dict],
        scores: Dict[str, float]
    ) -> Set[str]:
        borderline_tags = self._filter_borderline_tags(candidate_tags, scores)
        
        if not borderline_tags:
            return set()
        
        candidates_list = [
            {
                'tag_id': tag_id,
                'name_he': candidate_tags[tag_id]['name_he'],
                'score': scores[tag_id]
            }
            for tag_id in borderline_tags
        ]
        
        candidates_list.sort(key=lambda x: x['score'], reverse=True)
        candidates_list = candidates_list[:self.config.max_llm_candidates]
        
        selected_ids = self._call_llm(lecture_title, lecture_description, candidates_list)
        
        return set(selected_ids)
    
    def _filter_borderline_tags(
        self, 
        candidate_tags: Dict[str, Dict], 
        scores: Dict[str, float]
    ) -> List[str]:
        borderline = []
        
        for tag_id, score in scores.items():
            if (self.config.llm_borderline_lower <= score < self.config.llm_borderline_upper):
                borderline.append(tag_id)
        
        return borderline
    
    def _call_llm(
        self, 
        title: str, 
        description: str, 
        candidates: List[Dict]
    ) -> List[str]:
        system_prompt = """אתה מומחה לתיוג הרצאות בעברית.
תפקידך לבחור תגיות רלוונטיות מתוך רשימת מועמדים.
החזר רק את מזהי התגיות (tag_id) שמתאימים באמת להרצאה.
העדף דיוק גבוה - אם אתה לא בטוח, אל תכלול תגית.
אם אין תגיות מתאימות, החזר רשימה ריקה."""

        candidates_text = "\n".join([
            f"- {c['tag_id']}: {c['name_he']} (ציון: {c['score']:.3f})"
            for c in candidates
        ])
        
        user_prompt = f"""הרצאה:
כותרת: {title}
תיאור: {description}

מועמדי תגיות:
{candidates_text}

בחר את מזהי התגיות הרלוונטיים ביותר."""

        try:
            response = self.client.chat.completions.create(
                model=self.config.llm_model,
                temperature=self.config.llm_temperature,
                max_tokens=self.config.llm_max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "tag_selection",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "selected_tag_ids": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                }
                            },
                            "required": ["selected_tag_ids"],
                            "additionalProperties": False
                        }
                    }
                }
            )
            
            content = response.choices[0].message.content
            if not content:
                logger.warning("LLM returned empty content")
                return []
            result = json.loads(content)
            selected_ids = result.get('selected_tag_ids', [])
            
            valid_ids = [tag_id for tag_id in selected_ids 
                        if any(c['tag_id'] == tag_id for c in candidates)]
            
            logger.debug(f"LLM selected {len(valid_ids)} tags from {len(candidates)} candidates")
            return valid_ids
            
        except Exception as e:
            logger.error(f"Error calling LLM arbiter: {e}")
            return []
