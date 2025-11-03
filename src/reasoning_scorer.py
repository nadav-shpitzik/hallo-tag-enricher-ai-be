import logging
import time
from typing import List, Dict, Optional
from openai import OpenAI
import json
from pydantic import BaseModel
from src.logging_utils import StructuredLogger, track_operation, _request_context
from src.ai_call_logger import AICallLogger

logger = StructuredLogger(__name__)
ai_call_logger = AICallLogger()

class TagSuggestion(BaseModel):
    tag_name_he: str  # LLM only returns the Hebrew name
    confidence: float
    rationale_he: str

class TaggingResponse(BaseModel):
    suggestions: List[TagSuggestion]
    reasoning_summary: str

class ReasoningScorer:
    def __init__(self, model: str = "gpt-4o", min_confidence: float = 0.80, confidence_scale: float = 0.85):
        self.client = OpenAI()
        self.model = model
        self.min_confidence = min_confidence
        self.confidence_scale = confidence_scale  # Calibration factor for over-confident LLMs
    
    def _estimate_llm_tokens(self, messages: List[Dict]) -> tuple[int, int]:
        """Estimate input/output tokens (rough: 1 token ~ 4 chars)."""
        input_chars = sum(len(str(m.get('content', ''))) for m in messages)
        input_tokens = input_chars // 4
        output_tokens = 200  # Conservative estimate for structured output
        return input_tokens, output_tokens
    
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
        
        messages = [
            {
                "role": "system",
                "content": """אתה מומחה בתיוג הרצאות בעברית. תפקידך לקרוא את תוכן ההרצאה ולהציע תגיות רלוונטיות.

## קטגוריות תגיות
תגיות מחולקות ל-5 קטגוריות, כל אחת משרתת מטרה שונה:

1. **נושא (Topic)**: התוכן המרכזי של ההרצאה - על מה היא עוסקת?
   - דוגמאות: פילוסופיה, הורות, הייטק, זוגיות, גיאופוליטיקה, כלכלה, חיים בריאים

2. **פרסונה (Persona)**: מי המרצה או איזה סוג דמות מדבר?
   - דוגמאות: אושיות רשת, מוזיקאים, מקצוענים, גיבורים, מנחי קבוצות

3. **טון (Tone)**: האווירה והגישה הרגשית של ההרצאה
   - דוגמאות: סיפור אישי, מצחיק, מרגש, פרקטי, מניע לפעולה

4. **פורמט (Format)**: המבנה והסגנון של ההרצאה
   - דוגמאות: פאנל, שיחה פתוחה, הכשרה מעשית, סיור, הנחיית אירועים

5. **קהל יעד (Audience)**: למי ההרצאה מיועדת?
   - דוגמאות: הרצאות למורים, הרצאות לנשים, הרצאות להייטק, דוברי אנגלית, הרצאות לגיל השלישי

## כללים חשובים
1. היה **שמרן** ברמת הביטחון - הצע רק תגיות שהן ממש רלוונטיות
2. השתמש ברמות ביטחון שונות: 0.60-0.70 לרלוונטיות בסיסית, 0.70-0.80 לרלוונטיות טובה, 0.80-0.95 רק לרלוונטיות מצוינת ומובהקת
3. התמקד בנושא המרכזי של ההרצאה - אל תציע יותר מדי תגיות
4. **שים לב לקטגוריה** של כל תגית - זה עוזר להבין את ההקשר והשימוש שלה
5. השתמש במידע על המרצה כדי להבין טוב יותר את תוכן ההרצאה
6. תן נימוק ברור בעברית למה התגית מתאימה
7. אם אין תגיות מתאימות - אל תציע כלום
8. העדף דיוק (precision) על פני כיסוי (recall) - עדיף פחות תגיות נכונות מאשר תגיות שגויות

## הנחיות לפי קטגוריה
(מקום להנחיות ספציפיות לכל קטגוריה בעתיד)"""
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        # Get request_id from context for correlation
        request_id = getattr(_request_context, 'request_id', None)
        
        # Track call timing
        call_start_time = time.time()
        
        try:
            
            with track_operation("reasoning_llm_call", logger, lecture_id=lecture.get('id'), num_tags=len(tags_to_consider)):
                response = self.client.beta.chat.completions.parse(
                    model=self.model,
                    messages=messages,
                    temperature=0.2,
                    response_format=TaggingResponse
                )
                
                # Calculate call duration
                call_duration_ms = (time.time() - call_start_time) * 1000
                
                # Track LLM usage (with fallback estimation)
                usage = response.usage
                if usage and hasattr(usage, 'prompt_tokens'):
                    input_tokens = usage.prompt_tokens
                    output_tokens = usage.completion_tokens
                    total_tokens = usage.total_tokens
                else:
                    # Estimate when API doesn't provide usage
                    input_tokens, output_tokens = self._estimate_llm_tokens(messages)
                    total_tokens = input_tokens + output_tokens
                
                # Estimate cost (gpt-4o: $5.00/1M input, $15.00/1M output)
                cost = (input_tokens / 1_000_000 * 5.00) + (output_tokens / 1_000_000 * 15.00)
                
                logger.info(
                    "LLM reasoning call completed",
                    model=self.model,
                    lecture_id=lecture.get('id'),
                    num_candidate_tags=len(tags_to_consider),
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                    estimated_cost_usd=round(cost, 6),
                    usage_source="api" if usage and hasattr(usage, 'prompt_tokens') else "estimated"
                )
            
            result = response.choices[0].message.parsed
            
            # Prepare response content for database logging (only tag names now)
            response_content = None
            if result:
                response_content = {
                    'suggestions': [
                        {
                            'tag_name_he': sugg.tag_name_he,
                            'confidence': sugg.confidence,
                            'rationale_he': sugg.rationale_he
                        }
                        for sugg in result.suggestions
                    ],
                    'reasoning_summary': result.reasoning_summary
                }
            
            # Log AI call to database
            ai_call_logger.log_call(
                call_type="reasoning_scorer",
                model=self.model,
                prompt_messages=messages,
                response_content=response_content,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                estimated_cost_usd=cost,
                duration_ms=call_duration_ms,
                status="success",
                request_id=request_id,
                lecture_id=lecture.get('id')
            )
            
            if result is None:
                logger.warning(f"No parsed result for lecture {lecture.get('id')}")
                return []
            
            # Create name -> tag_id mapping for post-processing
            name_to_tag = {}
            for tag in all_tags:
                name_he = tag.get('name_he', '').strip()
                if name_he:
                    name_to_tag[name_he] = tag
                    
                    # Also add synonyms as aliases
                    synonyms = tag.get('synonyms_he', '').strip()
                    if synonyms:
                        for synonym in synonyms.split(','):
                            synonym = synonym.strip()
                            if synonym:
                                name_to_tag[synonym] = tag
            
            formatted_suggestions = []
            for sugg in result.suggestions:
                # Map tag_name_he -> tag_id using our mapping
                tag_name = sugg.tag_name_he.strip()
                
                if tag_name not in name_to_tag:
                    logger.warning(
                        f"LLM returned tag name '{tag_name}' which doesn't match any known tag - skipping",
                        request_id=getattr(_request_context, 'request_id', None)
                    )
                    continue
                
                matched_tag = name_to_tag[tag_name]
                tag_id = matched_tag['tag_id']
                
                # Apply confidence calibration (LLMs tend to be over-confident)
                calibrated_confidence = sugg.confidence * self.confidence_scale
                
                # Only include if calibrated confidence meets threshold
                if calibrated_confidence >= self.min_confidence:
                    formatted_suggestions.append({
                        'tag_id': tag_id,
                        'tag_name_he': tag_name,
                        'score': calibrated_confidence,
                        'rationale': sugg.rationale_he,
                        'model': f'reasoning:{self.model}'
                    })
            
            logger.info(
                "Reasoning scoring completed",
                lecture_id=lecture.get('id'),
                num_suggestions=len(formatted_suggestions),
                filtered_from=len(result.suggestions)
            )
            
            return formatted_suggestions
            
        except Exception as e:
            # Log failed AI call to database
            call_duration_ms = (time.time() - call_start_time) * 1000 if 'call_start_time' in locals() else 0
            
            ai_call_logger.log_call(
                call_type="reasoning_scorer",
                model=self.model,
                prompt_messages=messages,
                response_content=None,
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                estimated_cost_usd=0.0,
                duration_ms=call_duration_ms,
                status="error",
                error_message=str(e),
                request_id=request_id,
                lecture_id=lecture.get('id')
            )
            
            logger.error(
                "Error in reasoning scorer",
                lecture_id=lecture.get('id'),
                error_type=type(e).__name__,
                error_message=str(e)
            )
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
        prompt_parts.append("התגיות מקובצות לפי קטגוריה:\n\n")
        
        tags_by_category = {}
        for tag in tags:
            category = tag.get('category', 'Unknown')
            if category not in tags_by_category:
                tags_by_category[category] = []
            tags_by_category[category].append(tag)
        
        category_order = ['Topic', 'Persona', 'Tone', 'Format', 'Audience', 'Unknown']
        for category in category_order:
            if category not in tags_by_category:
                continue
            
            category_tags = tags_by_category[category]
            prompt_parts.append(f"### {category}\n")
            
            for tag in category_tags:
                # Show tag name first since that's what LLM will use
                tag_line = f"- **{tag.get('name_he', '')}**"
                if tag.get('synonyms_he'):
                    tag_line += f" (שמות נוספים: {tag['synonyms_he']})"
                prompt_parts.append(tag_line + "\n")
            
            prompt_parts.append("\n")
        
        prompt_parts.append("\n# משימה\n")
        prompt_parts.append("על בסיס תוכן ההרצאה והרקע על המרצה, הצע תגיות מתאימות **מתוך רשימת התגיות שסופקה בלבד**.\n\n")
        prompt_parts.append("לכל תגית ציין:\n")
        prompt_parts.append("1. שם התגית בעברית (**העתק בדיוק** מהרשימה למעלה)\n")
        prompt_parts.append("2. רמת ביטחון (0.0-1.0)\n")
        prompt_parts.append("3. נימוק בעברית למה התגית מתאימה\n\n")
        
        prompt_parts.append("## פורמט פלט נדרש (דוגמה)\n")
        prompt_parts.append('```json\n')
        prompt_parts.append('{\n')
        prompt_parts.append('  "suggestions": [\n')
        prompt_parts.append('    {\n')
        prompt_parts.append('      "tag_name_he": "בריאות הנפש",\n')
        prompt_parts.append('      "confidence": 0.88,\n')
        prompt_parts.append('      "rationale_he": "נימוק מפורט בעברית למה התגית מתאימה להרצאה זו"\n')
        prompt_parts.append('    }\n')
        prompt_parts.append('  ],\n')
        prompt_parts.append('  "reasoning_summary": "בהתבסס על התוכן, נבחרו תגיות עם קשר ברור לנושא המרכזי."\n')
        prompt_parts.append('}\n')
        prompt_parts.append('```\n\n')
        prompt_parts.append("⚠️ **חשוב ביותר:** השתמש רק בשמות תגיות **המדויקים** מהרשימה למעלה. העתק את השם כמו שהוא, כולל ניקוד ואותיות. המערכת תשייך את ה-ID אוטומטית.\n")
        
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
