from openai import OpenAI
from typing import List, Dict, Set
import logging
import json
from src.logging_utils import StructuredLogger, track_operation

logger = StructuredLogger(__name__)


class LLMArbiter:
    def __init__(self, api_key: str, config):
        self.client = OpenAI(api_key=api_key)
        self.config = config
    
    def _estimate_llm_tokens(self, messages: List[Dict]) -> tuple[int, int]:
        """Estimate input/output tokens (rough: 1 token ~ 4 chars)."""
        input_chars = sum(len(str(m.get('content', ''))) for m in messages)
        input_tokens = input_chars // 4
        output_tokens = 50  # Conservative estimate for JSON selection
        return input_tokens, output_tokens
    
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
                'category': candidate_tags[tag_id].get('category', 'Unknown'),
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

## קטגוריות תגיות
תגיות מחולקות ל-5 קטגוריות:
- **נושא (Topic)**: על מה ההרצאה עוסקת - דוגמאות: פילוסופיה, הורות, זוגיות, כלכלה
- **פרסונה (Persona)**: מי המרצה - דוגמאות: אושיות רשת, מוזיקאים, מקצוענים, גיבורים
- **טון (Tone)**: אווירה ורגש - דוגמאות: סיפור אישי, מצחיק, מרגש, פרקטי
- **פורמט (Format)**: מבנה ההרצאה - דוגמאות: פאנל, שיחה פתוחה, סיור, הכשרה מעשית
- **קהל יעד (Audience)**: למי מיועד - דוגמאות: הרצאות למורים, הרצאות להייטק, דוברי אנגלית

החזר רק את מזהי התגיות (tag_id) שמתאימים באמת להרצאה.
העדף דיוק גבוה - אם אתה לא בטוח, אל תכלול תגית.
אם אין תגיות מתאימות, החזר רשימה ריקה."""

        candidates_text = "\n".join([
            f"- {c['tag_id']}: {c['name_he']} [{c.get('category', 'Unknown')}] (ציון: {c['score']:.3f})"
            for c in candidates
        ])
        
        user_prompt = f"""הרצאה:
כותרת: {title}
תיאור: {description}

מועמדי תגיות:
{candidates_text}

בחר את מזהי התגיות הרלוונטיים ביותר."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        try:
            with track_operation("arbiter_llm_call", logger, num_candidates=len(candidates)):
                response = self.client.chat.completions.create(
                    model=self.config.llm_model,
                    temperature=self.config.llm_temperature,
                    max_tokens=self.config.llm_max_tokens,
                    messages=messages,
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
                
                # Estimate cost (gpt-4o-mini: ~$0.15/1M input, ~$0.60/1M output)
                cost = (input_tokens / 1_000_000 * 0.15) + (output_tokens / 1_000_000 * 0.60)
                
                logger.info(
                    "LLM arbiter call completed",
                    model=self.config.llm_model,
                    num_candidates=len(candidates),
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                    estimated_cost_usd=round(cost, 6),
                    usage_source="api" if usage and hasattr(usage, 'prompt_tokens') else "estimated"
                )
            
            content = response.choices[0].message.content
            if not content:
                logger.warning("LLM returned empty content")
                return []
            result = json.loads(content)
            selected_ids = result.get('selected_tag_ids', [])
            
            valid_ids = [tag_id for tag_id in selected_ids 
                        if any(c['tag_id'] == tag_id for c in candidates)]
            
            logger.info(
                "LLM arbiter selection completed",
                num_selected=len(valid_ids),
                num_candidates=len(candidates)
            )
            return valid_ids
            
        except Exception as e:
            logger.error(
                "Error calling LLM arbiter",
                error_type=type(e).__name__,
                error_message=str(e)
            )
            return []
