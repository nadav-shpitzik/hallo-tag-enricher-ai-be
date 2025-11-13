import logging
import time
from typing import List, Dict, Optional, Literal, get_args
from openai import OpenAI
import json
from pydantic import BaseModel, Field, create_model
from src.logging_utils import StructuredLogger, track_operation, _request_context
from src.ai_call_logger import AICallLogger
from src.category_reasoning import (
    run_per_category_reasoning,
    category_results_to_suggestions,
    CATEGORIES
)

logger = StructuredLogger(__name__)
ai_call_logger = AICallLogger()

class TagSuggestion(BaseModel):
    """Single tag suggestion with confidence and rationale."""
    tag_name_he: str = Field(description="Exact tag name from the provided list")
    confidence: float = Field(description="Confidence score 0.0-1.0")
    rationale_he: str = Field(description="Hebrew rationale for why this tag fits")

class TaggingResponse(BaseModel):
    """Response model for tag suggestions."""
    suggestions: List[TagSuggestion] = Field(description="List of tag suggestions")
    reasoning_summary: str = Field(description="Hebrew summary of reasoning process")

class ReasoningScorer:
    def __init__(self, model: str = "gpt-4o", min_confidence: float = 0.80, confidence_scale: float = 0.85):
        self.client = OpenAI()
        self.model = model
        self.min_confidence = min_confidence
        self.confidence_scale = confidence_scale  # Calibration factor for over-confident LLMs
        
        # Create name -> tag mapping for all tags (populated in score_lecture)
        self.name_to_tag = {}
    
    def _estimate_llm_tokens(self, messages: List[Dict]) -> tuple[int, int]:
        """Estimate input/output tokens (rough: 1 token ~ 4 chars)."""
        input_chars = sum(len(str(m.get('content', ''))) for m in messages)
        input_tokens = input_chars // 4
        output_tokens = 200  # Conservative estimate for structured output
        return input_tokens, output_tokens
    
    def _call_category_llm(self, system_text: str, user_text: str, category: str) -> str:
        """
        Call LLM for one category. Returns raw text response.
        This is used by the category reasoning module.
        """
        messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text}
        ]
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        
        return response.choices[0].message.content or "{}"
    
    def score_lecture(
        self,
        lecture: Dict,
        all_tags: List[Dict],
        lecturer_profile: Optional[str] = None,
        candidate_tags: Optional[List[Dict]] = None
    ) -> List[Dict]:
        # Note: For per-category reasoning, we ignore candidate_tags and use all_tags
        # This ensures each category sees all its tags for better recall
        tags_to_consider = all_tags
        
        logger.info(f"Using per-category reasoning with {len(tags_to_consider)} total tags for lecture {lecture.get('id')}")
        
        # Build name -> tag mapping for validation
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
        
        # Get request_id from context for correlation
        request_id = getattr(_request_context, 'request_id', None)
        
        # Track overall timing
        overall_start_time = time.time()
        
        try:
            # Create a wrapper for category LLM calls that tracks tokens and cost
            total_input_tokens = 0
            total_output_tokens = 0
            category_call_details = []
            
            def llm_caller_with_tracking(system_text: str, user_text: str, category: str) -> str:
                """Wrapper that calls LLM and tracks usage for each category."""
                nonlocal total_input_tokens, total_output_tokens
                
                call_start = time.time()
                
                messages = [
                    {"role": "system", "content": system_text},
                    {"role": "user", "content": user_text}
                ]
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.2,
                    response_format={"type": "json_object"}
                )
                
                call_duration_ms = (time.time() - call_start) * 1000
                
                # Track usage
                usage = response.usage
                if usage and hasattr(usage, 'prompt_tokens'):
                    input_tokens = usage.prompt_tokens
                    output_tokens = usage.completion_tokens
                else:
                    # Fallback estimation
                    input_tokens, output_tokens = self._estimate_llm_tokens(messages)
                
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens
                
                # Track category details for logging
                category_call_details.append({
                    'category': category,
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'duration_ms': call_duration_ms
                })
                
                return response.choices[0].message.content or "{}"
            
            # Run per-category reasoning (5 parallel LLM calls)
            with track_operation("per_category_reasoning", logger, lecture_id=lecture.get('id'), num_tags=len(tags_to_consider)):
                category_results = run_per_category_reasoning(
                    lecture=lecture,
                    all_labels=tags_to_consider,
                    lecturer_profile=lecturer_profile,
                    llm_caller=llm_caller_with_tracking
                )
            
            # Calculate total metrics
            overall_duration_ms = (time.time() - overall_start_time) * 1000
            total_tokens = total_input_tokens + total_output_tokens
            cost = (total_input_tokens / 1_000_000 * 5.00) + (total_output_tokens / 1_000_000 * 15.00)
            
            # Count total suggestions across all categories
            total_suggestions_raw = sum(len(r.chosen_ids) for r in category_results.values())
            
            logger.info(
                "Per-category reasoning completed",
                model=self.model,
                lecture_id=lecture.get('id'),
                num_categories=len(category_results),
                total_tags=len(tags_to_consider),
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                total_tokens=total_tokens,
                estimated_cost_usd=round(cost, 6),
                total_raw_suggestions=total_suggestions_raw,
                category_breakdown=[
                    {
                        'category': cat,
                        'tags_selected': len(res.chosen_ids),
                        'input_tokens': detail['input_tokens'],
                        'output_tokens': detail['output_tokens']
                    }
                    for cat, res in category_results.items()
                    for detail in category_call_details if detail['category'] == cat
                ]
            )
            
            # Convert category results to standard suggestions format
            # Build tags_data dict for conversion
            tags_data = {tag['tag_id']: tag for tag in all_tags}
            
            raw_suggestions = category_results_to_suggestions(category_results, tags_data)
            
            # Apply confidence calibration and filtering
            formatted_suggestions = []
            for sugg in raw_suggestions:
                # Apply confidence calibration (LLMs tend to be over-confident)
                calibrated_confidence = sugg['score'] * self.confidence_scale
                
                # Only include if calibrated confidence meets threshold
                if calibrated_confidence >= self.min_confidence:
                    formatted_suggestions.append({
                        'tag_id': sugg['tag_id'],
                        'tag_name_he': sugg['tag_name_he'],
                        'score': calibrated_confidence,
                        'rationale': sugg['rationale'],
                        'category': sugg['category'],
                        'model': f'category_reasoning:{self.model}'
                    })
            
            # Log AI calls to database (aggregate all 5 category calls)
            response_content = {
                'category_results': {
                    cat: {
                        'chosen_ids': res.chosen_ids,
                        'confidence': res.confidence,
                        'rationales': res.rationales
                    }
                    for cat, res in category_results.items()
                },
                'total_suggestions': len(formatted_suggestions),
                'filtered_from': total_suggestions_raw
            }
            
            ai_call_logger.log_call(
                call_type="per_category_reasoning",
                model=self.model,
                prompt_messages=[{'note': f'5 category calls: {", ".join(CATEGORIES)}'}],
                response_content=response_content,
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                total_tokens=total_tokens,
                estimated_cost_usd=cost,
                duration_ms=overall_duration_ms,
                status="success",
                request_id=request_id,
                lecture_id=lecture.get('id')
            )
            
            logger.info(
                "Category reasoning scoring completed",
                lecture_id=lecture.get('id'),
                num_suggestions=len(formatted_suggestions),
                filtered_from=total_suggestions_raw,
                categories_used=list(category_results.keys())
            )
            
            return formatted_suggestions
            
        except Exception as e:
            # Log failed AI call to database
            call_duration_ms = (time.time() - overall_start_time) * 1000 if 'overall_start_time' in locals() else 0
            
            ai_call_logger.log_call(
                call_type="per_category_reasoning",
                model=self.model,
                prompt_messages=[{'error': 'Failed during category reasoning execution'}],
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
                "Error in per-category reasoning scorer",
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
        
        # Debug logging to see what categories were found
        logger.info(f"Tags by category: {list(tags_by_category.keys())}, total tags: {len(tags)}")
        
        # Map Hebrew category names to English display names
        category_display = {
            'נושא': 'Topic (נושא)',
            'פרסונה': 'Persona (פרסונה)',
            'טון': 'Tone (טון)',
            'פורמט': 'Format (פורמט)',
            'קהל יעד': 'Audience (קהל יעד)',
            # Also support English categories for backward compatibility
            'Topic': 'Topic (נושא)',
            'Persona': 'Persona (פרסונה)',
            'Tone': 'Tone (טון)',
            'Format': 'Format (פורמט)',
            'Audience': 'Audience (קהל יעד)',
            'Unknown': 'Other'
        }
        
        # Render all categories that were found
        for category in tags_by_category.keys():
            category_tags = tags_by_category[category]
            display_name = category_display.get(category, category)
            prompt_parts.append(f"### {display_name}\n")
            
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
        prompt_parts.append("## ⚠️ לפני שליחת התשובה - בדוק שנית!\n")
        prompt_parts.append("**חובה:** ודא שכל שם תגית מועתק **תו-תו** מהרשימה למעלה.\n\n")
        prompt_parts.append("טעויות נפוצות שצריך להימנע מהן:\n")
        prompt_parts.append("- ❌ אל תשמיט את ה' הידיעה (דוגמה: \"חברה ישראלית\" במקום \"החברה הישראלית\")\n")
        prompt_parts.append("- ❌ אל תמציא תגיות חדשות (דוגמה: \"עיתונאות\" כשיש \"מדיה ותקשורת\")\n")
        prompt_parts.append("- ❌ אל תשנה רווחים או סימנים (דוגמה: \"בריאות-הנפש\" במקום \"בריאות הנפש\")\n")
        prompt_parts.append("- ✅ העתק **בדיוק** כמו שמופיע ברשימה\n\n")
        prompt_parts.append("המערכת תשייך את ה-ID אוטומטית לפי השם שתציין.\n")
        
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
