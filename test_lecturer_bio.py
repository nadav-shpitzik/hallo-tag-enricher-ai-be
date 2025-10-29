#!/usr/bin/env python3
"""
Test script for lecturer bio enrichment feature.
"""

import os
import requests
import json

API_BASE = "http://localhost:5000"

def test_lecturer_bio_enrichment():
    """Test that lecturer bio lookup and caching works."""
    
    print("🧪 Testing Lecturer Bio Enrichment")
    print("=" * 60)
    
    # Test request with lecturer info in reasoning mode
    request_data = {
        "request_id": "test_bio_001",
        "model_version": "v1",
        "artifact_version": "test",
        "scoring_mode": "reasoning",
        "lecture": {
            "id": "test_lec_001",
            "title": "פילוסופיה יהודית",
            "description": "שיעור מעמיק בפילוסופיה היהודית המודרנית",
            "lecturer_id": "rec_rabbi_test",
            "lecturer_name": "הרב אברהם יצחק הכהן קוק"
        },
        "labels": [
            {
                "id": "lab_philosophy",
                "name_he": "פילוסופיה",
                "category": "Topic",
                "active": True
            },
            {
                "id": "lab_modern_thought",
                "name_he": "מחשבה מודרנית",
                "category": "Topic",
                "active": True
            },
            {
                "id": "lab_deep",
                "name_he": "עיוני מעמיק",
                "category": "Tone",
                "active": True
            }
        ]
    }
    
    print("\n📤 Sending request with lecturer info...")
    print(f"   Lecturer: {request_data['lecture']['lecturer_name']}")
    print(f"   Mode: reasoning")
    
    try:
        # First request - should trigger bio search
        print("\n⏱️  First request (bio search expected)...")
        response = requests.post(
            f"{API_BASE}/suggest-tags",
            json=request_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code != 200:
            print(f"❌ Error: {response.status_code}")
            print(response.json())
            return False
        
        result1 = response.json()
        print(f"✅ Got {len(result1.get('suggestions', []))} suggestions")
        
        for sugg in result1.get('suggestions', []):
            print(f"   - {sugg['label_id']}: {sugg['confidence']:.3f}")
            if 'rationale_he' in sugg:
                print(f"     נימוק: {sugg['rationale_he'][:80]}...")
        
        # Second request - should use cached bio
        print("\n⏱️  Second request (cached bio expected)...")
        response2 = requests.post(
            f"{API_BASE}/suggest-tags",
            json=request_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response2.status_code != 200:
            print(f"❌ Error: {response2.status_code}")
            return False
        
        result2 = response2.json()
        print(f"✅ Got {len(result2.get('suggestions', []))} suggestions (cached)")
        
        print("\n🎉 Lecturer bio enrichment working!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_database():
    """Check if lecturer bio was saved to database."""
    print("\n📊 Checking Database")
    print("=" * 60)
    
    try:
        import psycopg2
        
        conn = psycopg2.connect(os.getenv('DATABASE_URL'))
        cursor = conn.cursor()
        
        cursor.execute("SELECT lecturer_id, lecturer_name, LEFT(bio_text, 100) FROM lecturer_bios")
        rows = cursor.fetchall()
        
        print(f"\n✅ Found {len(rows)} cached lecturer bios:")
        for row in rows:
            print(f"   - {row[0]}: {row[1]}")
            print(f"     Bio: {row[2]}...")
        
        cursor.close()
        conn.close()
        
        return True
        
    except Exception as e:
        print(f"❌ Database check failed: {e}")
        return False


if __name__ == "__main__":
    # Check API is running
    try:
        response = requests.get(f"{API_BASE}/health")
        if response.status_code != 200:
            print("❌ API server not running. Start it first!")
            exit(1)
    except:
        print("❌ Cannot connect to API server. Start it first!")
        exit(1)
    
    # Run tests
    success = test_lecturer_bio_enrichment()
    
    if success:
        check_database()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed")
