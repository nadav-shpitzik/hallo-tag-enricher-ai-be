"""
Per-category reasoning module for improved tag suggestion recall.

Instead of running one large prompt with all ~115 tags mixed together,
this module runs 5 focused prompts - one per category (Topic, Persona, 
Audience, Tone, Format). Each category sees only its relevant tags,
allowing the model to pay better attention and catch long-tail labels.
"""

import json
import re
import logging
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.telemetry import log_line

logger = logging.getLogger(__name__)

CATEGORIES = ["Topic", "Persona", "Audience", "Tone", "Format"]


@dataclass
class CategoryResult:
    """Result from analyzing one category."""
    category: str
    chosen_ids: List[str]
    confidence: Dict[str, float]
    rationales: Dict[str, str]


def group_labels_by_category(labels: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Group labels by category for focused prompting.
    
    Args:
        labels: List of tag dicts with keys: tag_id, name_he, synonyms_he, category
        
    Returns:
        Dict mapping category name to list of tags in that category
    """
    buckets = {c: [] for c in CATEGORIES}
    
    for lb in labels:
        cat = lb.get("category")
        if cat and cat in buckets:
            syns = lb.get("synonyms_he", "")
            # Handle both string and list formats for synonyms
            if isinstance(syns, str):
                syn_list = [s.strip() for s in syns.split(",") if s.strip()]
            elif isinstance(syns, list):
                syn_list = syns
            else:
                syn_list = []
            
            buckets[cat].append({
                "tag_id": lb["tag_id"],
                "name_he": lb.get("name_he", ""),
                "synonyms_he": syn_list,
            })
    
    return buckets


# Base rules shared across all category prompts (reduces token cost and improves maintainability)
BASE_RULES_HE = """
## כללים כלליים
1. **כל הפלטים חייבים להיות בעברית** - תגיות, נימוקים, והסברים
2. החזר JSON בלבד לפי הסכמה המדויקת
3. **חובה** להשתמש בשמות התגיות בדיוק כפי שהן מופיעות ברשימה (כולל ה' הידיעה, רווחים, וכל תו)
4. תן נימוק קצר (עד 120 תווים) בעברית למה התגית מתאימה
5. אם אין תגיות מתאימות - החזר רשימה ריקה
"""

# Category-specific instructions (focused guidance per category)
CATEGORY_INSTRUCTIONS_HE = {
    "Topic": """## הנחיות ספציפיות לנושא
- בחר רק תגים שמתאימים באמת לתוכן ההרצאה
- היה שמרן - עדיף פחות תגיות נכונות מאשר תגיות שגויות
- השתמש ברמות ביטחון שונות: 0.60-0.70 לרלוונטיות בסיסית, 0.70-0.85 לרלוונטיות טובה, 0.85-0.95 לרלוונטיות מובהקת""",
    
    "Persona": """## הנחיות ספציפיות לפרסונה
- זהה את סוג הדמות של המרצה: אושיות רשת, מוזיקאים, מקצוענים, גיבורים, מנחי קבוצות וכו'
- אל תנחש - בחר רק אם יש עדויות ברורות מהביוגרפיה או התיאור
- מותר להחזיר רשימה ריקה אם סוג הדמות לא ברור""",
    
    "Audience": """## הנחיות ספציפיות לקהל יעד
- זהה למי ההרצאה מיועדת: מורים, נשים, אנשי הייטק, דוברי אנגלית, גיל שלישי וכו'
- בחר רק אם יש סימנים ברורים בכותרת או בתיאור
- אם אין אינדיקציה ברורה לקהל ספציפי - החזר רשימה ריקה""",
    
    "Tone": """## הנחיות ספציפיות לטון
- זהה את הטון: סיפור אישי, מצחיק, מרגש, פרקטי, מניע לפעולה וכו'
- דלג אם הטון לא ברור מהתיאור
- ניתן לבחור יותר מטון אחד אם הרצאה משלבת כמה סגנונות""",
    
    "Format": """## הנחיות ספציפיות לפורמט
- זהה את הפורמט: פאנל, שיחה פתוחה, הכשרה מעשית, סיור, הנחיית אירועים וכו'
- אם אין רמז ברור לפורמט - החזר רשימה ריקה
- בדרך כלל יש פורמט אחד דומיננטי להרצאה"""
}

# Composed system prompts (base + category-specific)
SYSTEM_PROMPTS_HE = {
    "Topic": f"""אתה מסווג נושאים להרצאות בעברית. תפקידך לבחור תגיות נושא רלוונטיות.
{BASE_RULES_HE}
{CATEGORY_INSTRUCTIONS_HE["Topic"]}""",
    
    "Persona": f"""אתה מסווג פרסונה (סוג דמות) של מרצים בעברית.
{BASE_RULES_HE}
{CATEGORY_INSTRUCTIONS_HE["Persona"]}""",
    
    "Audience": f"""אתה מסווג קהל יעד להרצאות בעברית.
{BASE_RULES_HE}
{CATEGORY_INSTRUCTIONS_HE["Audience"]}""",
    
    "Tone": f"""אתה מסווג טון (סגנון ואווירה) של הרצאות בעברית.
{BASE_RULES_HE}
{CATEGORY_INSTRUCTIONS_HE["Tone"]}""",
    
    "Format": f"""אתה מסווג פורמט (מבנה וסגנון) של הרצאות בעברית.
{BASE_RULES_HE}
{CATEGORY_INSTRUCTIONS_HE["Format"]}"""
}


def build_user_prompt_he(
    lecture: Dict[str, Any],
    lecturer_profile: Optional[str],
    category: str,
    tags: List[Dict[str, Any]]
) -> str:
    """
    Build Hebrew user prompt for one category.
    
    Args:
        lecture: Lecture dict with title/description
        lecturer_profile: Optional lecturer bio text
        category: Category name (Topic, Persona, etc.)
        tags: List of tags in this category
        
    Returns:
        Hebrew prompt string
    """
    title = lecture.get("title") or lecture.get("lecture_title") or ""
    desc = lecture.get("description") or lecture.get("lecture_description") or ""
    
    lines = [
        f"קטגוריה: {category}",
        f"כותרת: {title}",
        f"תיאור: {desc}",
    ]
    
    if lecturer_profile:
        lines.append(f"ביוגרפיה של המרצה: {lecturer_profile}")
    
    lines.append("")
    lines.append("רשימת תגים זמינים (tag_id | name_he | [synonyms_he]):")
    
    for t in tags:
        syns = ", ".join(t.get("synonyms_he", [])) if t.get("synonyms_he") else ""
        lines.append(f"- {t['tag_id']} | {t.get('name_he', '')} | [{syns}]")
    
    lines.append("")
    lines.append("החזר JSON בלבד במבנה הבא:")
    lines.append("{")
    lines.append(f'  "category": "{category}",')
    lines.append('  "chosen_ids": ["<tag_id>", "..."],')
    lines.append('  "confidence": {"<tag_id>": 0.85},')
    lines.append('  "rationales": {"<tag_id>": "הסבר קצר בעברית"}')
    lines.append("}")
    
    return "\n".join(lines)


# Regex to find JSON blocks in model output
JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def parse_category_json(
    raw_text: str,
    allowed_ids: set[str],
    category: str
) -> CategoryResult:
    """
    Parse LLM response into CategoryResult.
    
    Args:
        raw_text: Raw text from LLM
        allowed_ids: Set of valid tag_ids for this category
        category: Category name
        
    Returns:
        CategoryResult with parsed data
    """
    text = raw_text.strip()
    
    # Try direct JSON parse first
    try:
        data = json.loads(text)
    except Exception:
        # Fallback: extract last JSON-looking block
        matches = list(JSON_BLOCK.finditer(text))
        if not matches:
            logger.warning(f"No JSON found in {category} response, returning empty")
            return CategoryResult(category, [], {}, {})
        
        try:
            data = json.loads(matches[-1].group(0))
        except Exception as e:
            logger.error(f"Failed to parse JSON for {category}: {e}")
            return CategoryResult(category, [], {}, {})
    
    # Extract and validate fields - deduplicate while preserving order
    seen = set()
    chosen = [tid for tid in data.get("chosen_ids", []) 
              if tid in allowed_ids and tid not in seen and not seen.add(tid)]
    conf = {k: float(v) for k, v in (data.get("confidence") or {}).items() if k in allowed_ids}
    rats = {k: str(v) for k, v in (data.get("rationales") or {}).items() if k in allowed_ids}
    
    return CategoryResult(
        category=category,
        chosen_ids=chosen,
        confidence=conf,
        rationales=rats
    )


def run_per_category_reasoning(
    lecture: Dict[str, Any],
    all_labels: List[Dict[str, Any]],
    lecturer_profile: Optional[str],
    llm_caller: Callable[[str, str, str], str],
    max_workers: int = 5,
    model: str = "gpt-4o"
) -> Dict[str, CategoryResult]:
    """
    Run reasoning for all categories in parallel.
    
    Args:
        lecture: Lecture dict with title/description
        all_labels: All available labels/tags
        lecturer_profile: Optional lecturer bio
        llm_caller: Function(system_text, user_text, category) -> str that calls LLM
        max_workers: Max parallel workers (default 5 for 5 categories)
        model: Model name for telemetry logging
        
    Returns:
        Dict mapping category name to CategoryResult
    """
    # Group labels by category
    by_cat = group_labels_by_category(all_labels)
    
    results: Dict[str, CategoryResult] = {}
    
    def process_category(category: str) -> tuple[str, CategoryResult]:
        """Process one category (for parallel execution)."""
        tags = by_cat.get(category, [])
        
        if not tags:
            logger.info(f"No tags for category {category}, skipping")
            return category, CategoryResult(category, [], {}, {})
        
        logger.info(f"Processing category {category} with {len(tags)} tags")
        
        sys_txt = SYSTEM_PROMPTS_HE[category]
        usr_txt = build_user_prompt_he(lecture, lecturer_profile, category, tags)
        
        # Call LLM
        raw = llm_caller(sys_txt, usr_txt, category)
        
        # Parse response
        allowed = {t["tag_id"] for t in tags}
        result = parse_category_json(raw, allowed, category)
        
        logger.info(f"Category {category} selected {len(result.chosen_ids)} tags")
        
        # Log telemetry for this category result
        lecture_id = lecture.get("id") or lecture.get("lecture_id")
        log_line({
            "kind": "category_reasoning_result",
            "lecture_id": lecture_id,
            "category": category,
            "model": model,
            "chosen_ids": result.chosen_ids,
            "confidence": result.confidence,
            "rationales_count": len(result.rationales or {})
        })
        
        return category, result
    
    # Execute all categories in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_category, cat): cat for cat in CATEGORIES}
        
        for future in as_completed(futures):
            category = futures[future]
            try:
                cat_name, result = future.result()
                results[cat_name] = result
            except Exception as e:
                logger.error(f"Error processing category {category}: {e}", exc_info=True)
                results[category] = CategoryResult(category, [], {}, {})
    
    return results


def category_results_to_suggestions(
    category_results: Dict[str, CategoryResult],
    tags_data: Dict[str, Dict]
) -> List[Dict]:
    """
    Convert category results to standard suggestion format.
    
    Args:
        category_results: Dict of CategoryResult per category
        tags_data: Dict mapping tag_id to tag metadata
        
    Returns:
        List of suggestion dicts compatible with ensemble scorer
    """
    suggestions = []
    
    for cat, result in category_results.items():
        for tid in result.chosen_ids:
            if tid not in tags_data:
                logger.warning(f"Tag {tid} from {cat} not found in tags_data, skipping")
                continue
            
            suggestions.append({
                "tag_id": tid,
                "tag_name_he": tags_data[tid].get("name_he", ""),
                "category": cat,
                "score": result.confidence.get(tid, 0.0),
                "confidence": result.confidence.get(tid, 0.0),
                "rationale": result.rationales.get(tid, ""),
                "reasons": ["per_category_reasoning"],
                "model": f"category_reasoning:{cat}"
            })
    
    return suggestions
