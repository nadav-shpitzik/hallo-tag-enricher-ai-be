#!/usr/bin/env python3
"""
Quick test script for ensemble scoring mode.
"""

import requests
import json

BASE_URL = "http://localhost:5000"

def test_ensemble_mode():
    """Test the ensemble scoring mode."""
    
    # First check if server is running
    try:
        response = requests.get(f"{BASE_URL}/health")
        health = response.json()
        print(f"✓ Server health: {health}")
        
        if not health.get('prototypes_loaded'):
            print("⚠ No prototypes loaded. Please train first with /train-ui or /train endpoint")
            return
    except Exception as e:
        print(f"✗ Server not responding: {e}")
        return
    
    # Test ensemble mode with a sample lecture
    test_request = {
        "request_id": "test_ensemble_001",
        "model_version": "v1",
        "artifact_version": "test",
        "scoring_mode": "ensemble",  # Test ensemble mode
        "lecture": {
            "id": "test_lec_001",
            "title": "קבלה ומיסטיקה יהודית",
            "description": "שיעור עמוק בסודות הזוהר והקבלה המעשית, עם דגש על ספירות והתפתחות רוחנית",
            "lecturer_name": "הרב משה כהן"
        },
        "labels": [
            {
                "id": "lab_kabbalah",
                "name_he": "קבלה",
                "category": "Topic",
                "active": True
            },
            {
                "id": "lab_mysticism",
                "name_he": "מיסטיקה",
                "category": "Topic",
                "active": True
            },
            {
                "id": "lab_zohar",
                "name_he": "זוהר",
                "category": "Topic",
                "active": True
            }
        ]
    }
    
    print("\n📝 Testing ensemble mode...")
    print(f"Request: {json.dumps(test_request, ensure_ascii=False, indent=2)}")
    
    try:
        response = requests.post(
            f"{BASE_URL}/suggest-tags",
            json=test_request,
            timeout=60  # Ensemble mode may take a few seconds
        )
        
        if response.status_code == 200:
            result = response.json()
            print("\n✓ Ensemble mode test successful!")
            print(f"\nResponse: {json.dumps(result, ensure_ascii=False, indent=2)}")
            
            suggestions = result.get('suggestions', [])
            print(f"\n📊 Got {len(suggestions)} suggestions:")
            for sugg in suggestions:
                print(f"  - {sugg['label_id']}: {sugg['confidence']:.3f} ({', '.join(sugg.get('reasons', []))})")
                if sugg.get('rationale_he'):
                    print(f"    Rationale: {sugg['rationale_he'][:100]}...")
        else:
            print(f"\n✗ Request failed with status {response.status_code}")
            print(f"Error: {response.text}")
    
    except Exception as e:
        print(f"\n✗ Error during test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Ensemble Scoring Mode")
    print("=" * 60)
    test_ensemble_mode()
    print("\n" + "=" * 60)
