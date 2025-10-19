#!/usr/bin/env python3
"""
Web-based viewer for tag suggestions with filters.
"""
import os
import json
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
from flask import Flask, render_template, request, jsonify, redirect, url_for
import subprocess
import threading
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

suggestions_df = None
lectures_cache = {}
tags_map = {}

def load_data():
    """Load suggestions, lectures, and tags on startup."""
    global suggestions_df, lectures_cache, tags_map
    
    csv_path = 'output/tag_suggestions.csv'
    if not os.path.exists(csv_path):
        print(f"Error: No suggestions file found at {csv_path}")
        return False
    
    suggestions_df = pd.read_csv(csv_path)
    
    tags_csv = 'data/tags.csv'
    tags_df = pd.read_csv(tags_csv)
    tags_map = dict(zip(tags_df['tag_id'], tags_df['name_he']))
    
    conn = psycopg2.connect(os.getenv('DATABASE_URL'))
    lecture_ids = suggestions_df['lecture_id'].unique().tolist()
    
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
                final_price
            FROM enriched_lectures
            WHERE id = ANY(%s)
        """, (lecture_ids,))
        
        for row in cur.fetchall():
            lectures_cache[row['id']] = dict(row)
    
    conn.close()
    print(f"Loaded {len(suggestions_df)} suggestions for {len(lectures_cache)} lectures")
    return True

@app.route('/')
def index():
    """Main page with all suggestions."""
    if suggestions_df is None:
        return "Error: Data not loaded. Check console for errors.", 500
    
    min_score = float(request.args.get('min_score', 0.0))
    max_score = float(request.args.get('max_score', 1.0))
    tag_filter = request.args.get('tag', '')
    model_filter = request.args.get('model', '')
    has_prev_tags = request.args.get('has_prev_tags', '')
    
    filtered_df = suggestions_df.copy()
    
    if min_score > 0:
        filtered_df = filtered_df[filtered_df['score'] >= min_score]
    if max_score < 1:
        filtered_df = filtered_df[filtered_df['score'] <= max_score]
    if tag_filter:
        filtered_df = filtered_df[filtered_df['tag_name_he'].str.contains(tag_filter, na=False)]
    if model_filter:
        filtered_df = filtered_df[filtered_df['model'].str.contains(model_filter, na=False)]
    
    grouped = filtered_df.groupby('lecture_id')
    
    lectures_data = []
    for lecture_id, suggestions in grouped:
        lecture = lectures_cache.get(lecture_id, {})
        previous_tags = lecture.get('lecture_tag_ids', []) or []
        
        if has_prev_tags == 'yes' and len(previous_tags) == 0:
            continue
        if has_prev_tags == 'no' and len(previous_tags) > 0:
            continue
        
        prev_tag_names = [tags_map.get(tag_id, tag_id) for tag_id in previous_tags]
        
        suggestions_list = []
        for _, sugg in suggestions.iterrows():
            suggestions_list.append({
                'tag_name': sugg['tag_name_he'],
                'tag_id': sugg['tag_id'],
                'score': float(sugg['score']),
                'rationale': sugg['rationale'],
                'model': sugg['model'],
                'is_llm': 'llm' in sugg['model'].lower()
            })
        
        lectures_data.append({
            'lecture_id': int(lecture_id),
            'external_id': lecture.get('lecture_external_id', ''),
            'title': lecture.get('lecture_title', ''),
            'lecturer_name': lecture.get('lecturer_name', ''),
            'lecturer_role': lecture.get('lecturer_role', ''),
            'description': lecture.get('lecture_description', ''),
            'price': lecture.get('final_price'),
            'previous_tags': prev_tag_names,
            'suggestions': suggestions_list,
            'suggestion_count': len(suggestions_list)
        })
    
    lectures_data.sort(key=lambda x: x['lecture_id'])
    
    unique_tags = sorted(suggestions_df['tag_name_he'].unique().tolist())
    
    stats = {
        'total_lectures': len(lectures_data),
        'total_suggestions': len(filtered_df),
        'avg_score': float(filtered_df['score'].mean()) if len(filtered_df) > 0 else 0,
        'llm_count': len(filtered_df[filtered_df['model'].str.contains('llm', case=False)]),
        'prototype_count': len(filtered_df[~filtered_df['model'].str.contains('llm', case=False)])
    }
    
    return render_template('index.html', 
                         lectures=lectures_data,
                         unique_tags=unique_tags,
                         stats=stats,
                         filters={
                             'min_score': min_score,
                             'max_score': max_score,
                             'tag': tag_filter,
                             'model': model_filter,
                             'has_prev_tags': has_prev_tags
                         })

@app.route('/api/lecture/<int:lecture_id>')
def get_lecture(lecture_id):
    """Get lecture details as JSON."""
    lecture = lectures_cache.get(lecture_id, {})
    suggestions = suggestions_df[suggestions_df['lecture_id'] == lecture_id]
    
    return jsonify({
        'lecture': lecture,
        'suggestions': suggestions.to_dict('records')
    })

@app.route('/rerun', methods=['POST'])
def rerun_batch():
    """Trigger batch processing in background."""
    def run_batch():
        try:
            result = subprocess.run(
                ['python', 'src/main.py'],
                capture_output=True,
                text=True,
                timeout=300
            )
            print(f"Batch completed with exit code: {result.returncode}")
            if result.returncode == 0:
                load_data()
        except subprocess.TimeoutExpired:
            print("Batch processing timed out after 5 minutes")
        except Exception as e:
            print(f"Error running batch: {e}")
    
    thread = threading.Thread(target=run_batch)
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'status': 'started',
        'message': 'Batch processing started in background. Refresh the page in a few minutes to see new results.'
    })

if __name__ == '__main__':
    if load_data():
        print("Starting web viewer on http://0.0.0.0:5000")
        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
        print("Failed to load data. Exiting.")
