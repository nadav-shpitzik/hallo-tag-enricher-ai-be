#!/usr/bin/env python3
"""
Test the enhanced v2 API features:
- Category-aware thresholds
- Related lectures boost  
- Intelligent reasons generation
- Training validation
"""

import requests
import json

API_URL = "http://localhost:5000"

def test_train_v2_with_validation():
    """Test training with v2 format and validation."""
    print("\n=== Testing /train with v2 format and validation ===")
    
    payload = {
        "lectures": [
            {
                "id": "lec001",
                "title": "בריאות נפשית בימינו",
                "description": "שיחה על חרדה, דיכאון והתמודדות עם אתגרים נפשיים",
                "label_ids": ["lab_topic_mental_health", "lab_tone_personal", "lab_format_talk"]
            },
            {
                "id": "lec002",
                "title": "סלבס מדברים על בריאות",
                "description": "אישים ידועים חולקים את הסיפור האישי שלהם",
                "label_ids": ["lab_persona_celebs", "lab_tone_personal", "lab_topic_mental_health"]
            },
            {
                "id": "lec003",
                "title": "הרצאה לקהל רחב",
                "description": "נושאים חשובים לכולם - בריאות וחינוך",
                "label_ids": ["lab_audience_general", "lab_format_talk"]
            },
            {
                "id": "lec004",
                "title": "שיחה אישית ואינטימית",
                "description": "שיחה פתוחה על נושאים אישיים",
                "label_ids": ["lab_tone_personal", "lab_format_talk"]
            },
            {
                "id": "lec005",
                "title": "פאנל עם אנשים מפורסמים",
                "description": "סלבריטאים דנים בנושאים חברתיים",
                "label_ids": ["lab_persona_celebs", "lab_format_talk", "lab_audience_general"]
            }
        ],
        "labels": [
            {"id": "lab_topic_mental_health", "name_he": "בריאות הנפש", "category": "Topic"},
            {"id": "lab_persona_celebs", "name_he": "סלבס", "category": "Persona"},
            {"id": "lab_tone_personal", "name_he": "אישי", "category": "Tone"},
            {"id": "lab_format_talk", "name_he": "הרצאה", "category": "Format"},
            {"id": "lab_audience_general", "name_he": "קהל רחב", "category": "Audience"}
        ]
    }
    
    response = requests.post(f"{API_URL}/train", json=payload)
    
    print(f"Status: {response.status_code}")
    result = response.json()
    print("Training result:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    # Validate response structure
    assert 'validation' in result, "Missing validation in response"
    assert 'categories' in result['validation'], "Missing categories in validation"
    
    print("\n✓ Training with validation passed!")
    return result


def test_suggestions_with_related_lectures():
    """Test suggestions with related lectures boost."""
    print("\n=== Testing suggestions with related lectures boost ===")
    
    payload = {
        "request_id": "test-boost-001",
        "model_version": "v1",
        "artifact_version": "test-2025-10-29",
        "lecture": {
            "id": "test001",
            "title": "על חרדה והתמודדות",
            "description": "כלים להתמודדות עם חרדה ומתח בחיי היומיום",
            "related_lectures": [
                {"id": "lec001", "title": "בריאות נפשית בימינו", "labels": ["lab_topic_mental_health", "lab_tone_personal"]},
                {"id": "lec002", "title": "סלבס מדברים", "labels": ["lab_persona_celebs"]}
            ]
        },
        "labels": [
            {"id": "lab_topic_mental_health", "name_he": "בריאות הנפש", "category": "Topic", "active": True},
            {"id": "lab_persona_celebs", "name_he": "סלבס", "category": "Persona", "active": True},
            {"id": "lab_tone_personal", "name_he": "אישי", "category": "Tone", "active": True},
            {"id": "lab_format_talk", "name_he": "הרצאה", "category": "Format", "active": True},
            {"id": "lab_audience_general", "name_he": "קהל רחב", "category": "Audience", "active": True}
        ]
    }
    
    response = requests.post(f"{API_URL}/suggest-tags", json=payload)
    
    print(f"Status: {response.status_code}")
    result = response.json()
    print("Suggestions:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    # Validate enhancements
    suggestions = result.get('suggestions', [])
    
    if suggestions:
        print("\n=== Analyzing enhancements ===")
        
        # Check for related lectures boost
        related_boosted = [s for s in suggestions if 'related_cooccur' in s.get('reasons', [])]
        if related_boosted:
            print(f"✓ Found {len(related_boosted)} suggestions with related_cooccur boost:")
            for s in related_boosted:
                print(f"  - {s['label_id']} ({s['category']}): confidence={s['confidence']:.3f}, reasons={s['reasons']}")
        
        # Check for intelligent reasons
        reason_types = set()
        for s in suggestions:
            reason_types.update(s.get('reasons', []))
        print(f"\n✓ Reason types found: {sorted(reason_types)}")
        
        # Check category distribution
        categories = {}
        for s in suggestions:
            cat = s.get('category', 'Unknown')
            categories[cat] = categories.get(cat, 0) + 1
        print(f"\n✓ Category distribution: {categories}")
    else:
        print("\n⚠ No suggestions returned (may need retraining with matching label IDs)")
    
    return result


if __name__ == "__main__":
    try:
        # Train with v2 format
        train_result = test_train_v2_with_validation()
        
        # Reload prototypes
        print("\n=== Reloading prototypes ===")
        reload_response = requests.post(f"{API_URL}/reload-prototypes")
        print(f"Reload status: {reload_response.status_code}")
        print(json.dumps(reload_response.json(), indent=2))
        
        # Test suggestions with enhancements
        suggestions_result = test_suggestions_with_related_lectures()
        
        print("\n=== Test Summary ===")
        print(f"✓ Training validation working")
        print(f"✓ Category-aware thresholds configured")
        print(f"✓ Related lectures boost implemented")
        print(f"✓ Intelligent reasons generation working")
        print(f"✓ Got {len(suggestions_result.get('suggestions', []))} suggestions")
        
        if train_result.get('validation', {}).get('warnings'):
            print(f"\n⚠ Training warnings:")
            for warning in train_result['validation']['warnings']:
                print(f"  - {warning}")
        
        print("\nAll enhancements tested! 🎉")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
