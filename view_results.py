#!/usr/bin/env python3
"""
View tag suggestions per lecture with full details.
Shows lecture info, previous tags, and new suggestions.
"""
import os
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()

def get_lecture_details(conn, lecture_ids):
    """Fetch lecture details from database."""
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
            SELECT 
                id,
                lecture_external_id,
                lecture_title,
                lecture_description,
                lecturer_name,
                lecturer_role,
                lecture_tag_ids,
                final_price,
                is_active
            FROM enriched_lectures
            WHERE id = ANY(%s)
            ORDER BY id
        """, (lecture_ids,))
        return {row['id']: dict(row) for row in cur.fetchall()}

def main():
    # Load suggestions CSV
    csv_path = os.getenv('OUTPUT_CSV_PATH', 'output/tag_suggestions.csv')
    if not os.path.exists(csv_path):
        print(f"❌ No suggestions file found at {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    
    # Load tags for name mapping
    tags_csv = os.getenv('TAGS_CSV_PATH', 'data/tags.csv')
    tags_df = pd.read_csv(tags_csv)
    tag_map = dict(zip(tags_df['tag_id'], tags_df['name_he']))
    
    # Connect to database
    conn = psycopg2.connect(os.getenv('DATABASE_URL'))
    
    # Get unique lecture IDs
    lecture_ids = df['lecture_id'].unique().tolist()
    lectures = get_lecture_details(conn, lecture_ids)
    
    # Group suggestions by lecture
    suggestions_by_lecture = df.groupby('lecture_id')
    
    print("=" * 100)
    print("TAG SUGGESTIONS REPORT - Per Lecture View")
    print("=" * 100)
    print()
    
    for idx, (lecture_id, suggestions) in enumerate(suggestions_by_lecture, 1):
        lecture = lectures.get(lecture_id)
        if not lecture:
            continue
        
        print(f"\n{'─' * 100}")
        print(f"📚 LECTURE #{idx} (ID: {lecture_id}, External: {lecture['lecture_external_id']})")
        print(f"{'─' * 100}")
        
        # Basic details
        print(f"\n📖 Title: {lecture['lecture_title'] or '—'}")
        print(f"👨‍🏫 Lecturer: {lecture['lecturer_name'] or '—'}")
        if lecture.get('lecturer_role'):
            print(f"   Role: {lecture['lecturer_role']}")
        if lecture.get('final_price'):
            print(f"💰 Price: ₪{lecture['final_price']:,}")
        
        # Description
        desc = lecture['lecture_description'] or ''
        if desc:
            # Truncate long descriptions
            desc_display = desc[:300] + '...' if len(desc) > 300 else desc
            print(f"\n📝 Description:")
            print(f"   {desc_display}")
        
        # Previous tags
        previous_tags = lecture.get('lecture_tag_ids') or []
        print(f"\n🏷️  PREVIOUS TAGS ({len(previous_tags)}):")
        if previous_tags:
            prev_tag_names = [tag_map.get(tag_id, tag_id) for tag_id in previous_tags]
            for tag_name in prev_tag_names:
                print(f"   • {tag_name}")
        else:
            print("   (No previous tags)")
        
        # New suggestions
        print(f"\n✨ NEW SUGGESTIONS ({len(suggestions)}):")
        for _, sugg in suggestions.iterrows():
            score_pct = f"{sugg['score'] * 100:.1f}%"
            model_type = "🤖 LLM" if 'llm' in sugg['model'] else "📊 Prototype"
            print(f"   • {sugg['tag_name_he']} [{score_pct}] {model_type}")
            print(f"     └─ {sugg['rationale']}")
        
        print()
    
    print("=" * 100)
    print(f"SUMMARY: {len(lecture_ids)} lectures with suggestions")
    print("=" * 100)
    
    conn.close()

if __name__ == '__main__':
    main()
