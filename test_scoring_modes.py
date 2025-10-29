#!/usr/bin/env python3
"""
Test all three scoring modes: fast, full_quality, and reasoning.
Compare results and verify the quality improvements.
"""

import requests
import json
import time

API_URL = "http://localhost:5000"

# Richer training data for better prototype quality
TRAINING_DATA = {
    "lectures": [
        {
            "id": "lec001",
            "title": "התמודדות עם חרדה בעידן המודרני",
            "description": "שיחה מעמיקה על דרכים להתמודד עם חרדה, מתח ולחץ נפשי בחיי היומיום. נכון לעומק הכלים הפסיכולוגיים והמעשיים",
            "label_ids": ["lab_topic_mental_health", "lab_tone_personal", "lab_format_talk", "lab_audience_general"]
        },
        {
            "id": "lec002",
            "title": "סלבריטאים חושפים: המסע האישי שלי",
            "description": "אישים ידועים משתפים לראשונה בסיפורים האישיים, האתגרים והתובנות מהדרך",
            "label_ids": ["lab_persona_celebs", "lab_tone_personal", "lab_format_interview"]
        },
        {
            "id": "lec003",
            "title": "חינוך ילדים בעידן הדיגיטלי",
            "description": "כנס חשוב לכולם - כיצד לחנך ילדים בעידן של רשתות חברתיות ומסכים",
            "label_ids": ["lab_topic_parenting", "lab_audience_general", "lab_format_talk"]
        },
        {
            "id": "lec004",
            "title": "שיחה אינטימית על אהבה ומערכות יחסים",
            "description": "שיחה פתוחה, אישית ואמיתית על יחסים, אהבה וזוגיות",
            "label_ids": ["lab_tone_personal", "lab_format_interview", "lab_topic_relationships"]
        },
        {
            "id": "lec005",
            "title": "פאנל: אישי ציבור דנים בבריאות הנפש",
            "description": "סלבריטאים ואנשי ציבור בשיחה גלויה על בריאות נפשית ומודעות",
            "label_ids": ["lab_persona_celebs", "lab_format_talk", "lab_topic_mental_health", "lab_audience_general"]
        },
        {
            "id": "lec006",
            "title": "מהפכת הבריאות הטבעית",
            "description": "הרצאה על בריאות, תזונה נכונה ואורח חיים בריא",
            "label_ids": ["lab_topic_health", "lab_format_talk", "lab_audience_general"]
        },
        {
            "id": "lec007",
            "title": "ריאיון עם מנכ\"ל חברת הייטק מצליחה",
            "description": "שיחה עם יזם מצליח על עסקים, חדשנות והצלחה",
            "label_ids": ["lab_format_interview", "lab_topic_business", "lab_audience_professional"]
        },
        {
            "id": "lec008",
            "title": "גידול ילדים מאושרים",
            "description": "כנס להורים - כיצד לגדל ילדים בריאים נפשית ורגשית",
            "label_ids": ["lab_topic_parenting", "lab_tone_personal", "lab_format_talk"]
        }
    ],
    "labels": [
        {"id": "lab_topic_mental_health", "name_he": "בריאות הנפש", "category": "Topic", "synonyms_he": "חרדה דיכאון נפש"},
        {"id": "lab_topic_parenting", "name_he": "הורות וחינוך", "category": "Topic", "synonyms_he": "חינוך ילדים גידול"},
        {"id": "lab_topic_health", "name_he": "בריאות כללית", "category": "Topic", "synonyms_he": "תזונה כושר בריאות"},
        {"id": "lab_topic_relationships", "name_he": "מערכות יחסים", "category": "Topic", "synonyms_he": "אהבה זוגיות"},
        {"id": "lab_topic_business", "name_he": "עסקים וקריירה", "category": "Topic", "synonyms_he": "הייטק יזמות"},
        {"id": "lab_persona_celebs", "name_he": "סלבריטאים", "category": "Persona", "synonyms_he": "מפורסמים"},
        {"id": "lab_tone_personal", "name_he": "אישי", "category": "Tone", "synonyms_he": "פרטי אינטימי"},
        {"id": "lab_format_talk", "name_he": "הרצאה", "category": "Format", "synonyms_he": "כנס פאנל"},
        {"id": "lab_format_interview", "name_he": "ריאיון", "category": "Format", "synonyms_he": "שיחה"},
        {"id": "lab_audience_general", "name_he": "קהל רחב", "category": "Audience", "synonyms_he": "כולם"},
        {"id": "lab_audience_professional", "name_he": "מקצועי", "category": "Audience", "synonyms_he": "עסקי"}
    ]
}

# Test lecture - should trigger mental health, personal tone
TEST_LECTURE = {
    "id": "test001",
    "title": "דיכאון ובדידות - המדריך המלא",
    "description": "שיחה עמוקה ואישית על התמודדות עם דיכאון, בדידות וחרדה. כולל כלים מעשיים וטיפים פסיכולוגיים",
    "related_lectures": [
        {"id": "lec001", "title": "התמודדות עם חרדה", "labels": ["lab_topic_mental_health", "lab_tone_personal"]},
        {"id": "lec005", "title": "פאנל בריאות נפש", "labels": ["lab_topic_mental_health"]}
    ]
}


def train_prototypes():
    """Train prototypes with rich data."""
    print("\n" + "="*60)
    print("TRAINING PROTOTYPES")
    print("="*60)
    
    response = requests.post(f"{API_URL}/train", json=TRAINING_DATA)
    if response.status_code != 200:
        print(f"❌ Training failed: {response.text}")
        return False
    
    result = response.json()
    print(f"✅ Training successful!")
    print(f"   Prototypes: {result['num_prototypes']}")
    print(f"   Lectures: {result['num_lectures']}")
    print(f"   Low-data tags: {result['low_data_tags']}")
    
    if result.get('validation', {}).get('warnings'):
        print(f"\n⚠️  Warnings:")
        for warning in result['validation']['warnings']:
            print(f"   - {warning}")
    
    # Reload prototypes
    reload_response = requests.post(f"{API_URL}/reload-prototypes")
    if reload_response.status_code == 200:
        print("✅ Prototypes reloaded")
    
    return True


def test_scoring_mode(mode_name, mode_value):
    """Test a specific scoring mode."""
    print(f"\n" + "="*60)
    print(f"TESTING: {mode_name.upper()} MODE")
    print("="*60)
    
    payload = {
        "request_id": f"test-{mode_value}-001",
        "model_version": "v1",
        "artifact_version": "test-2025-10-29",
        "scoring_mode": mode_value,
        "lecture": TEST_LECTURE,
        "labels": TRAINING_DATA["labels"]
    }
    
    start_time = time.time()
    response = requests.post(f"{API_URL}/suggest-tags", json=payload)
    elapsed = time.time() - start_time
    
    if response.status_code != 200:
        print(f"❌ Request failed: {response.text}")
        return None
    
    result = response.json()
    suggestions = result.get('suggestions', [])
    
    print(f"⏱️  Time: {elapsed:.2f}s")
    print(f"📊 Suggestions: {len(suggestions)}")
    
    if suggestions:
        print(f"\n{'Label':<30} {'Category':<12} {'Confidence':<10} {'Reasons'}")
        print("-" * 80)
        for sugg in suggestions:
            label = next((l for l in TRAINING_DATA["labels"] if l['id'] == sugg['label_id']), {})
            label_name = label.get('name_he', sugg['label_id'])
            reasons_str = ", ".join(sugg.get('reasons', []))
            
            print(f"{label_name:<30} {sugg['category']:<12} {sugg['confidence']:.3f}      {reasons_str}")
            
            # Show Hebrew rationale if present
            if 'rationale_he' in sugg:
                print(f"   └─ נימוק: {sugg['rationale_he']}")
    else:
        print("⚠️  No suggestions returned")
    
    return {
        'mode': mode_name,
        'elapsed': elapsed,
        'count': len(suggestions),
        'suggestions': suggestions
    }


def compare_modes(results):
    """Compare results across modes."""
    print("\n" + "="*60)
    print("COMPARISON ACROSS MODES")
    print("="*60)
    
    print(f"\n{'Mode':<15} {'Time (s)':<12} {'# Suggestions':<15} {'Avg Confidence'}")
    print("-" * 60)
    
    for result in results:
        if result is None:
            continue
        
        avg_conf = 0
        if result['suggestions']:
            avg_conf = sum(s['confidence'] for s in result['suggestions']) / len(result['suggestions'])
        
        print(f"{result['mode']:<15} {result['elapsed']:<12.2f} {result['count']:<15} {avg_conf:.3f}")
    
    # Check which labels appear in each mode
    print("\n📋 Label Distribution:")
    all_label_ids = set()
    for result in results:
        if result and result['suggestions']:
            all_label_ids.update(s['label_id'] for s in result['suggestions'])
    
    for label_id in all_label_ids:
        label = next((l for l in TRAINING_DATA["labels"] if l['id'] == label_id), {})
        label_name = label.get('name_he', label_id)
        
        modes_with_label = []
        for result in results:
            if result and any(s['label_id'] == label_id for s in result['suggestions']):
                modes_with_label.append(result['mode'])
        
        print(f"  {label_name:<30} → {', '.join(modes_with_label)}")


if __name__ == "__main__":
    try:
        # Train
        if not train_prototypes():
            exit(1)
        
        time.sleep(1)  # Let server settle
        
        # Test all three modes
        results = []
        
        # Mode 1: Fast (prototype only)
        results.append(test_scoring_mode("Fast (Prototype)", "fast"))
        time.sleep(0.5)
        
        # Mode 2: Full Quality (prototype + arbiter)
        results.append(test_scoring_mode("Full Quality (Arbiter)", "full_quality"))
        time.sleep(0.5)
        
        # Mode 3: Reasoning (pure LLM)
        results.append(test_scoring_mode("Reasoning (Pure LLM)", "reasoning"))
        
        # Compare
        compare_modes(results)
        
        print("\n" + "="*60)
        print("✅ ALL TESTS COMPLETE!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
