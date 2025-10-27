#!/usr/bin/env python3
"""
Test script for the Tag Suggestions API

This script:
1. Trains prototypes using sample data
2. Tests tag suggestions endpoint
3. Tests reload endpoint
"""

import json
import requests

API_URL = "http://localhost:5000"

# Sample training data (lectures with existing tags)
training_data = {
    "lectures": [
        {
            "id": "lec_001",
            "lecture_title": "מבוא לתלמוד",
            "lecture_description": "שיעור מבוא בסיסי בתלמוד בבלי, מסכת ברכות",
            "lecture_tag_ids": ["talmud", "gemara", "berachot"]
        },
        {
            "id": "lec_002",
            "lecture_title": "קבלה ומיסטיקה יהודית",
            "lecture_description": "סודות הקבלה והזוהר, מקורות מיסטיים ביהדות",
            "lecture_tag_ids": ["kabbalah", "mysticism", "zohar"]
        },
        {
            "id": "lec_003",
            "lecture_title": "הלכות שבת",
            "lecture_description": "דיני שבת, איסורי מלאכה והיתרים",
            "lecture_tag_ids": ["halakha", "shabbat", "jewish_law"]
        },
        {
            "id": "lec_004",
            "lecture_title": "פרשת השבוע - בראשית",
            "lecture_description": "פרשת בראשית, מעשה בראשית וסיפור הבריאה",
            "lecture_tag_ids": ["torah", "beresheet", "weekly_portion"]
        },
        {
            "id": "lec_005",
            "lecture_title": "תלמוד בבלי - מסכת שבת",
            "lecture_description": "עיון במסכת שבת, סוגיות נבחרות",
            "lecture_tag_ids": ["talmud", "gemara", "shabbat"]
        },
    ],
    "tags": {
        "talmud": {"tag_id": "talmud", "name_he": "תלמוד", "synonyms_he": "גמרא תלמוד בבלי"},
        "gemara": {"tag_id": "gemara", "name_he": "גמרא", "synonyms_he": "תלמוד סוגיות"},
        "berachot": {"tag_id": "berachot", "name_he": "ברכות", "synonyms_he": "מסכת ברכות"},
        "kabbalah": {"tag_id": "kabbalah", "name_he": "קבלה", "synonyms_he": "סודות מיסטיקה"},
        "mysticism": {"tag_id": "mysticism", "name_he": "מיסטיקה", "synonyms_he": "קבלה רוחניות"},
        "zohar": {"tag_id": "zohar", "name_he": "זוהר", "synonyms_he": "ספר הזוהר קבלה"},
        "halakha": {"tag_id": "halakha", "name_he": "הלכה", "synonyms_he": "דינים משפטים"},
        "shabbat": {"tag_id": "shabbat", "name_he": "שבת", "synonyms_he": "שבת קודש יום מנוחה"},
        "jewish_law": {"tag_id": "jewish_law", "name_he": "דיני ישראל", "synonyms_he": "הלכה משפטים"},
        "torah": {"tag_id": "torah", "name_he": "תורה", "synonyms_he": "חומש תנ״ך"},
        "beresheet": {"tag_id": "beresheet", "name_he": "בראשית", "synonyms_he": "ספר בראשית"},
        "weekly_portion": {"tag_id": "weekly_portion", "name_he": "פרשת השבוע", "synonyms_he": "פרשה שבועית"},
    }
}

# Sample test data (new lectures to tag)
test_data = {
    "lectures": [
        {
            "id": "test_001",
            "lecture_title": "עיון בגמרא - מסכת ברכות",
            "lecture_description": "שיעור מעמיק בתלמוד בבלי מסכת ברכות עם ראשונים"
        },
        {
            "id": "test_002",
            "lecture_title": "סודות הזוהר",
            "lecture_description": "למידת קבלה מעשית וסודות הזוהר הקדוש"
        }
    ],
    "tags": training_data["tags"]
}


def test_train():
    """Test the training endpoint."""
    print("=" * 60)
    print("STEP 1: Training prototypes...")
    print("=" * 60)
    
    try:
        response = requests.post(
            f"{API_URL}/train",
            json=training_data,
            headers={"Content-Type": "application/json"},
            timeout=120  # Training can take time with embeddings
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
        
        if response.status_code == 200:
            print("\n✓ Training successful!")
            return True
        else:
            print("\n✗ Training failed!")
            return False
            
    except Exception as e:
        print(f"\n✗ Error during training: {e}")
        return False


def test_suggest():
    """Test the tag suggestion endpoint."""
    print("\n" + "=" * 60)
    print("STEP 2: Getting tag suggestions...")
    print("=" * 60)
    
    try:
        response = requests.post(
            f"{API_URL}/suggest-tags",
            json=test_data,
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        
        print(f"Status Code: {response.status_code}")
        result = response.json()
        print(f"Response: {json.dumps(result, indent=2, ensure_ascii=False)}")
        
        if response.status_code == 200:
            print(f"\n✓ Got {len(result.get('suggestions', []))} suggestions!")
            
            # Show suggestions grouped by lecture
            if 'suggestions' in result:
                print("\nSuggestions by lecture:")
                by_lecture = {}
                for sug in result['suggestions']:
                    lec_id = sug['lecture_id']
                    if lec_id not in by_lecture:
                        by_lecture[lec_id] = []
                    by_lecture[lec_id].append(sug)
                
                for lec_id, suggestions in by_lecture.items():
                    print(f"\n{lec_id}:")
                    for sug in suggestions:
                        print(f"  - {sug['tag_name_he']} ({sug['tag_id']}): {sug['score']:.3f}")
            
            return True
        else:
            print("\n✗ Suggestion request failed!")
            return False
            
    except Exception as e:
        print(f"\n✗ Error during suggestion: {e}")
        return False


def test_reload():
    """Test the reload prototypes endpoint."""
    print("\n" + "=" * 60)
    print("STEP 3: Testing reload endpoint...")
    print("=" * 60)
    
    try:
        response = requests.post(
            f"{API_URL}/reload-prototypes",
            timeout=10
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
        
        if response.status_code == 200:
            print("\n✓ Reload successful!")
            return True
        else:
            print("\n✗ Reload failed!")
            return False
            
    except Exception as e:
        print(f"\n✗ Error during reload: {e}")
        return False


def test_health():
    """Test the health check endpoint."""
    print("\n" + "=" * 60)
    print("STEP 0: Health check...")
    print("=" * 60)
    
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
        return response.status_code == 200
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return False


if __name__ == "__main__":
    print("Testing Tag Suggestions API")
    print("=" * 60)
    
    # Test health first
    if not test_health():
        print("\n✗ API server is not running!")
        exit(1)
    
    # Run tests
    train_ok = test_train()
    if not train_ok:
        print("\n✗ Training failed, stopping tests")
        exit(1)
    
    suggest_ok = test_suggest()
    reload_ok = test_reload()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Health check: ✓")
    print(f"Training: {'✓' if train_ok else '✗'}")
    print(f"Suggestions: {'✓' if suggest_ok else '✗'}")
    print(f"Reload: {'✓' if reload_ok else '✗'}")
    
    if train_ok and suggest_ok and reload_ok:
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed!")
