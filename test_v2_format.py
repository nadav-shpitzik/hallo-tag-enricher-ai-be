#!/usr/bin/env python3
"""
Test the v2 API format with new request/response structure.
"""

import requests
import json

API_URL = "http://localhost:5000"

def test_v2_suggest_tags():
    """Test the /suggest-tags endpoint with v2 format."""
    print("\n=== Testing /suggest-tags endpoint (v2 format) ===")
    
    payload = {
        "request_id": "c5f6f5f2-6c36-4b6f-9d6f-1d0b02b24a11",
        "model_version": "v1",
        "artifact_version": "labels-emb-2025-10-29",
        "lecture": {
            "id": "rec17SffStTL231k8",
            "title": "על חרדה והתמודדות",
            "description": "אסי עזר ומיכל עזר סטיין הם אחים המתמחים בבריאות הנפש והתמודדות עם חרדות. בהרצאה זו הם מציגים כלים יומיומיים להתמודדות עם חרדה ומתח.",
            "lecturer_id": "recasCfleyvOlrhkf",
            "lecturer_name": "אסי עזר ומיכל עזר סטיין",
            "lecturer_role": "אסי עזר, יוצר ומנחה; מיכל עזר סטיין, מאמנת אישית",
            "language": "he",
            "related_lectures": [
                {"id": "recR59LwPxi07sk6g", "title": "לחיות עם חרדה", "labels": ["lab_persona_celebs", "lab_tone_personal"]},
                {"id": "recJMrbAISxCxbr8m", "title": "בריאות נפשית לקהל הרחב", "labels": ["lab_topic_mental_health"]}
            ]
        },
        "labels": [
            {"id": "lab_persona_celebs", "name_he": "סלבס", "category": "Persona", "active": True},
            {"id": "lab_topic_mental_health", "name_he": "בריאות הנפש", "category": "Topic", "active": True},
            {"id": "lab_tone_personal", "name_he": "אישי", "category": "Tone", "active": True},
            {"id": "lab_format_talk", "name_he": "הרצאה", "category": "Format", "active": True},
            {"id": "lab_audience_general", "name_he": "קהל רחב", "category": "Audience", "active": True}
        ]
    }
    
    print("Request payload:")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    
    response = requests.post(f"{API_URL}/suggest-tags", json=payload)
    
    print(f"\nStatus: {response.status_code}")
    print("Response:")
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    
    # Validate response structure
    data = response.json()
    assert 'request_id' in data, "Missing request_id in response"
    assert 'model_version' in data, "Missing model_version in response"
    assert 'artifact_version' in data, "Missing artifact_version in response"
    assert 'suggestions' in data, "Missing suggestions in response"
    
    assert data['request_id'] == payload['request_id'], "Request ID mismatch"
    
    # Validate suggestion structure
    if data['suggestions']:
        suggestion = data['suggestions'][0]
        assert 'label_id' in suggestion, "Missing label_id in suggestion"
        assert 'category' in suggestion, "Missing category in suggestion"
        assert 'confidence' in suggestion, "Missing confidence in suggestion"
        assert 'reasons' in suggestion, "Missing reasons in suggestion"
        assert isinstance(suggestion['reasons'], list), "Reasons should be a list"
    
    print("\n✓ All v2 format validations passed!")
    return data


def test_health():
    """Test the /health endpoint."""
    print("\n=== Testing /health endpoint ===")
    response = requests.get(f"{API_URL}/health")
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    return response.json()


if __name__ == "__main__":
    try:
        # Test health
        health = test_health()
        
        if not health.get('prototypes_loaded'):
            print("\n⚠ No prototypes loaded. Please train first.")
            print("Run: python test_api.py (to train with sample data)")
        else:
            # Test v2 format
            suggestions = test_v2_suggest_tags()
            
            print("\n=== Test Summary ===")
            print(f"✓ Health check passed")
            print(f"✓ V2 format working correctly")
            print(f"✓ Got {len(suggestions['suggestions'])} suggestions")
            print("\nAll tests passed! 🎉")
            
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
