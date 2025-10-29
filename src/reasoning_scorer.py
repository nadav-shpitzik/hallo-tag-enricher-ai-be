import logging
from typing import List, Dict, Optional
from openai import OpenAI
import json
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class TagSuggestion(BaseModel):
    tag_id: str
    tag_name_he: str
    confidence: float
    rationale_he: str

class TaggingResponse(BaseModel):
    suggestions: List[TagSuggestion]
    reasoning_summary: str

class ReasoningScorer:
    def __init__(self, model: str = "gpt-4o-mini", min_confidence: float = 0.85, confidence_scale: float = 0.85):
        self.client = OpenAI()
        self.model = model
        self.min_confidence = min_confidence
        self.confidence_scale = confidence_scale  # Calibration factor for over-confident LLMs
    
    def score_lecture(
        self,
        lecture: Dict,
        all_tags: List[Dict],
        lecturer_profile: Optional[str] = None,
        candidate_tags: Optional[List[Dict]] = None
    ) -> List[Dict]:
        tags_to_consider = candidate_tags if candidate_tags and len(candidate_tags) > 0 else all_tags
        
        logger.debug(f"Considering {len(tags_to_consider)} tags for lecture {lecture.get('id')}")
        
        prompt = self._build_prompt(lecture, tags_to_consider, lecturer_profile)
        
        try:
            response = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": """אתה מומחה בתיוג הרצאות בעברית. תפקידך לקרוא את תוכן ההרצאה ולהציע תגיות רלוונטיות.

כללים חשובים:
1. היה **שמרן** ברמת הביטחון - הצע רק תגיות שהן ממש רלוונטיות
2. השתמש ברמות ביטחון שונות: 0.60-0.70 לרלוונטיות בסיסית, 0.70-0.80 לרלוונטיות טובה, 0.80-0.95 רק לרלוונטיות מצוינת ומובהקת
3. התמקד בנושא המרכזי של ההרצאה - אל תציע יותר מדי תגיות
4. השתמש במידע על המרצה כדי להבין טוב יותר את תוכן ההרצאה
5. תן נימוק ברור בעברית למה התגית מתאימה
6. אם אין תגיות מתאימות - אל תציע כלום
7. העדף דיוק (precision) על פני כיסוי (recall) - עדיף פחות תגיות נכונות מאשר תגיות שגויות"""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.2,
                response_format=TaggingResponse
            )
            
            result = response.choices[0].message.parsed
            
            if result is None:
                logger.warning(f"No parsed result for lecture {lecture.get('id')}")
                return []
            
            # Create tag_id validation set
            valid_tag_ids = {tag['tag_id'] for tag in all_tags}
            
            formatted_suggestions = []
            for sugg in result.suggestions:
                # Validate tag_id exists in our tags list
                if sugg.tag_id not in valid_tag_ids:
                    logger.warning(f"LLM hallucinated invalid tag_id '{sugg.tag_id}' for tag '{sugg.tag_name_he}' - skipping")
                    continue
                
                # Apply confidence calibration (LLMs tend to be over-confident)
                calibrated_confidence = sugg.confidence * self.confidence_scale
                
                # Only include if calibrated confidence meets threshold
                if calibrated_confidence >= self.min_confidence:
                    formatted_suggestions.append({
                        'tag_id': sugg.tag_id,
                        'tag_name_he': sugg.tag_name_he,
                        'score': calibrated_confidence,
                        'rationale': sugg.rationale_he,
                        'model': f'reasoning:{self.model}'
                    })
            
            logger.debug(f"Lecture {lecture.get('id')}: {len(formatted_suggestions)} high-confidence suggestions (filtered from {len(result.suggestions)} total)")
            
            return formatted_suggestions
            
        except Exception as e:
            logger.error(f"Error scoring lecture {lecture.get('id')} with reasoning model: {e}")
            return []
    
    def _build_prompt(
        self,
        lecture: Dict,
        tags: List[Dict],
        lecturer_profile: Optional[str]
    ) -> str:
        prompt_parts = []
        
        prompt_parts.append("# הרצאה לתיוג\n")
        prompt_parts.append(f"**כותרת:** {lecture.get('lecture_title', 'לא צוין')}\n")
        
        if lecture.get('lecture_description'):
            prompt_parts.append(f"**תיאור:** {lecture['lecture_description']}\n")
        
        if lecture.get('lecturer_name'):
            prompt_parts.append(f"**מרצה:** {lecture['lecturer_name']}\n")
            
            if lecturer_profile:
                prompt_parts.append(f"**רקע על המרצה:** {lecturer_profile}\n")
        
        prompt_parts.append(f"\n# תגיות זמינות ({len(tags)} אופציות)\n")
        
        for tag in tags:
            tag_line = f"- **{tag['tag_id']}**: {tag['name_he']}"
            if tag.get('synonyms_he'):
                tag_line += f" (שמות נוספים: {tag['synonyms_he']})"
            prompt_parts.append(tag_line + "\n")
        
        prompt_parts.append("\n# משימה\n")
        prompt_parts.append("על בסיס תוכן ההרצאה והרקע על המרצה, הצע תגיות מתאימות.\n")
        prompt_parts.append("לכל תגית ציין:\n")
        prompt_parts.append("1. מזהה התגית (tag_id)\n")
        prompt_parts.append("2. שם התגית בעברית\n")
        prompt_parts.append("3. רמת ביטחון (0.0-1.0)\n")
        prompt_parts.append("4. נימוק בעברית למה התגית מתאימה\n")
        
        return "".join(prompt_parts)
    
    def score_batch(
        self,
        lectures: List[Dict],
        all_tags: List[Dict],
        lecturer_profiles: Dict[str, Optional[str]]
    ) -> Dict[int, List[Dict]]:
        all_suggestions = {}
        
        for i, lecture in enumerate(lectures):
            lecture_id = lecture['id']
            
            existing_tags = lecture.get('lecture_tag_ids') or []
            if existing_tags and len(existing_tags) > 0:
                logger.debug(f"Skipping lecture {lecture_id} - already has {len(existing_tags)} tags")
                continue
            
            lecturer_profile = None
            if lecture.get('lecturer_name'):
                lecturer_profile = lecturer_profiles.get(lecture['lecturer_name'])
            
            suggestions = self.score_lecture(
                lecture,
                all_tags,
                lecturer_profile,
                None
            )
            
            if suggestions:
                all_suggestions[lecture_id] = suggestions
            
            if (i + 1) % 10 == 0:
                logger.info(f"Scored {i + 1}/{len(lectures)} lectures with reasoning model")
        
        logger.info(f"Generated {sum(len(s) for s in all_suggestions.values())} suggestions for {len(all_suggestions)} lectures using reasoning model")
        
        return all_suggestions
