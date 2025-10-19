#!/usr/bin/env python3
"""
Web-based viewer for tag suggestions with filters.
"""
import os
import sys
import json
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
from flask import Flask, render_template, request, jsonify, redirect, url_for
import subprocess
import threading
from dotenv import load_dotenv

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from database import DatabaseConnection

load_dotenv()

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

suggestions_df = None
lectures_cache = {}
tags_map = {}

def load_data():
    """Load suggestions, lectures, and tags from database."""
    global suggestions_df, lectures_cache, tags_map
    
    # Load tags mapping
    tags_csv = 'data/tags.csv'
    if not os.path.exists(tags_csv):
        print(f"Error: Tags file not found at {tags_csv}")
        return False
    
    tags_df = pd.read_csv(tags_csv)
    tags_map = dict(zip(tags_df['tag_id'], tags_df['name_he']))
    
    # Load suggestions from database
    with DatabaseConnection(os.getenv('DATABASE_URL')) as db:
        with db.connection.cursor(cursor_factory=RealDictCursor) as cur:
            # Get all suggestions with their status
            cur.execute("""
                SELECT 
                    suggestion_id,
                    lecture_id,
                    tag_id,
                    score,
                    rationale,
                    model,
                    status,
                    approved_by,
                    approved_at,
                    synced_at,
                    created_at
                FROM lecture_tag_suggestions
                ORDER BY created_at DESC
            """)
            suggestions = cur.fetchall()
            
            if not suggestions:
                print("No suggestions found in database")
                suggestions_df = pd.DataFrame()
                return True
            
            # Convert to DataFrame
            suggestions_df = pd.DataFrame([dict(s) for s in suggestions])
            # Map tag_id to tag_name_he, use tag_id itself if not found in CSV
            suggestions_df['tag_name_he'] = suggestions_df['tag_id'].apply(
                lambda x: tags_map.get(x, f"[Missing: {x}]")
            )
            
            # Get unique lecture IDs
            lecture_ids = suggestions_df['lecture_id'].unique().tolist()
            
            # Load lecture details
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
    
    print(f"Loaded {len(suggestions_df)} suggestions for {len(lectures_cache)} lectures from database")
    return True

@app.route('/lectures')
def lectures_view():
    """Lecture-centric view with all suggestions."""
    # Reload data on each request to show latest status
    load_data()
    
    # Handle empty database gracefully
    if suggestions_df is None or suggestions_df.empty:
        return render_template('index.html', 
                             lectures=[],
                             unique_tags=[],
                             stats={
                                 'total_lectures': 0,
                                 'total_suggestions': 0,
                                 'avg_score': 0,
                                 'llm_count': 0,
                                 'prototype_count': 0,
                                 'pending_count': 0,
                                 'approved_count': 0,
                                 'rejected_count': 0,
                                 'synced_count': 0,
                                 'failed_count': 0
                             },
                             filters={
                                 'min_score': 0.0,
                                 'max_score': 1.0,
                                 'tag': '',
                                 'model': '',
                                 'has_prev_tags': '',
                                 'status': ''
                             })
    
    min_score = float(request.args.get('min_score', 0.0))
    max_score = float(request.args.get('max_score', 1.0))
    tag_filter = request.args.get('tag', '')
    model_filter = request.args.get('model', '')
    has_prev_tags = request.args.get('has_prev_tags', '')
    status_filter = request.args.get('status', '')
    
    filtered_df = suggestions_df.copy()
    
    if min_score > 0:
        filtered_df = filtered_df[filtered_df['score'] >= min_score]
    if max_score < 1:
        filtered_df = filtered_df[filtered_df['score'] <= max_score]
    if tag_filter:
        filtered_df = filtered_df[filtered_df['tag_name_he'].str.contains(tag_filter, na=False)]
    if model_filter:
        filtered_df = filtered_df[filtered_df['model'].str.contains(model_filter, na=False)]
    if status_filter:
        filtered_df = filtered_df[filtered_df['status'] == status_filter]
    
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
                'suggestion_id': int(sugg['suggestion_id']),
                'tag_name': sugg['tag_name_he'],
                'tag_id': sugg['tag_id'],
                'score': float(sugg['score']),
                'rationale': sugg['rationale'],
                'model': sugg['model'],
                'is_llm': 'llm' in sugg['model'].lower(),
                'status': sugg['status'],
                'approved_by': sugg.get('approved_by'),
                'approved_at': sugg.get('approved_at'),
                'synced_at': sugg.get('synced_at')
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
    
    # Filter out NaN values before sorting
    unique_tags = sorted([tag for tag in suggestions_df['tag_name_he'].unique().tolist() if pd.notna(tag)])
    
    # Calculate status-based statistics
    status_counts = suggestions_df['status'].value_counts().to_dict()
    
    stats = {
        'total_lectures': len(lectures_data),
        'total_suggestions': len(filtered_df),
        'avg_score': float(filtered_df['score'].mean()) if len(filtered_df) > 0 else 0,
        'llm_count': len(filtered_df[filtered_df['model'].str.contains('llm', case=False)]),
        'prototype_count': len(filtered_df[~filtered_df['model'].str.contains('llm', case=False)]),
        'pending_count': status_counts.get('pending', 0),
        'approved_count': status_counts.get('approved', 0),
        'rejected_count': status_counts.get('rejected', 0),
        'synced_count': status_counts.get('synced', 0),
        'failed_count': status_counts.get('failed', 0)
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
                             'has_prev_tags': has_prev_tags,
                             'status': status_filter
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

@app.route('/api/approve/<int:suggestion_id>', methods=['POST'])
def approve_suggestion(suggestion_id):
    """Approve a single suggestion (pending -> approved only)."""
    try:
        actor = request.json.get('actor', 'web_user') if request.json else 'web_user'
        
        with DatabaseConnection(os.getenv('DATABASE_URL')) as db:
            # Attempt update with expected status check
            success, current_status = db.update_suggestion_status(
                suggestion_id, 'approved', actor, expected_status='pending'
            )
            
            if success:
                return jsonify({'status': 'approved', 'suggestion_id': suggestion_id})
            elif current_status is None:
                return jsonify({'error': 'Suggestion not found'}), 404
            else:
                return jsonify({
                    'error': f"Cannot approve: suggestion is '{current_status}', expected 'pending'",
                    'current_status': current_status
                }), 409
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/reject/<int:suggestion_id>', methods=['POST'])
def reject_suggestion(suggestion_id):
    """Reject a single suggestion (pending -> rejected only)."""
    try:
        actor = request.json.get('actor', 'web_user') if request.json else 'web_user'
        
        with DatabaseConnection(os.getenv('DATABASE_URL')) as db:
            # Attempt update with expected status check
            success, current_status = db.update_suggestion_status(
                suggestion_id, 'rejected', actor, expected_status='pending'
            )
            
            if success:
                return jsonify({'status': 'rejected', 'suggestion_id': suggestion_id})
            elif current_status is None:
                return jsonify({'error': 'Suggestion not found'}), 404
            else:
                return jsonify({
                    'error': f"Cannot reject: suggestion is '{current_status}', expected 'pending'",
                    'current_status': current_status
                }), 409
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/bulk_approve', methods=['POST'])
def bulk_approve():
    """Bulk approve suggestions for a lecture, optionally filtered by minimum score."""
    try:
        data = request.json
        lecture_id = data.get('lecture_id')
        min_score = data.get('min_score', 0.0)
        actor = data.get('actor', 'web_user')
        
        if not lecture_id:
            return jsonify({'error': 'lecture_id is required'}), 400
        
        with DatabaseConnection(os.getenv('DATABASE_URL')) as db:
            # Get all pending suggestions for this lecture
            with db.connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT suggestion_id, score 
                    FROM lecture_tag_suggestions
                    WHERE lecture_id = %s AND status = 'pending' AND score >= %s
                """, (lecture_id, min_score))
                
                suggestions_to_approve = cursor.fetchall()
            
            if not suggestions_to_approve:
                return jsonify({'message': 'No pending suggestions found matching criteria', 'approved_count': 0})
            
            # Approve each suggestion with expected status check
            approved_count = 0
            skipped_count = 0
            for suggestion in suggestions_to_approve:
                success, _ = db.update_suggestion_status(
                    suggestion['suggestion_id'], 'approved', actor, expected_status='pending'
                )
                if success:
                    approved_count += 1
                else:
                    skipped_count += 1
            
            return jsonify({
                'status': 'success',
                'approved_count': approved_count,
                'skipped_count': skipped_count,
                'lecture_id': lecture_id,
                'min_score': min_score
            })
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    """Main page: Lecturer-centric view with grouped suggestions."""
    load_data()
    
    if suggestions_df is None or suggestions_df.empty:
        return render_template('lecturers.html', lecturers=[], stats={})
    
    # Join suggestions with lecture data to get lecturer info
    with DatabaseConnection(os.getenv('DATABASE_URL')) as db:
        with db.connection.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT 
                    s.suggestion_id,
                    s.lecture_id,
                    s.tag_id,
                    s.score,
                    s.rationale,
                    s.model,
                    s.status,
                    l.lecturer_external_id,
                    l.lecturer_name,
                    l.lecture_external_id,
                    l.lecture_title
                FROM lecture_tag_suggestions s
                JOIN enriched_lectures l ON s.lecture_id = l.id
                WHERE l.lecturer_external_id IS NOT NULL
                ORDER BY l.lecturer_name, s.tag_id
            """)
            rows = cur.fetchall()
    
    # Group by lecturer
    from collections import defaultdict
    lecturer_data = defaultdict(lambda: {
        'lecturer_external_id': None,
        'lecturer_name': None,
        'lecture_count': 0,
        'lectures': set(),
        'tags': defaultdict(lambda: {
            'tag_id': None,
            'tag_name_he': None,
            'max_score': 0,
            'rationales': [],
            'source_lectures': [],
            'statuses': set(),
            'suggestion_ids': []
        })
    })
    
    for row in rows:
        lecturer_id = row['lecturer_external_id']
        tag_id = row['tag_id']
        
        # Update lecturer info
        if lecturer_data[lecturer_id]['lecturer_external_id'] is None:
            lecturer_data[lecturer_id]['lecturer_external_id'] = lecturer_id
            lecturer_data[lecturer_id]['lecturer_name'] = row['lecturer_name']
        
        # Track unique lectures for this lecturer
        lecturer_data[lecturer_id]['lectures'].add(row['lecture_id'])
        
        # Aggregate tag data
        tag_info = lecturer_data[lecturer_id]['tags'][tag_id]
        tag_info['tag_id'] = tag_id
        tag_info['tag_name_he'] = tags_map.get(tag_id, f"[Missing: {tag_id}]")
        tag_info['max_score'] = max(tag_info['max_score'], row['score'])
        tag_info['rationales'].append({
            'text': row['rationale'],
            'score': row['score'],
            'lecture_title': row['lecture_title']
        })
        tag_info['source_lectures'].append({
            'lecture_external_id': row['lecture_external_id'],
            'lecture_title': row['lecture_title']
        })
        tag_info['statuses'].add(row['status'])
        tag_info['suggestion_ids'].append(row['suggestion_id'])
    
    # Convert to list and clean up
    lecturers_list = []
    for lecturer_id, lecturer_info in lecturer_data.items():
        lecturer_info['lecture_count'] = len(lecturer_info['lectures'])
        
        # Convert tags dict to list
        tags_list = []
        for tag_id, tag_info in lecturer_info['tags'].items():
            # Deduplicate source lectures
            unique_lectures = []
            seen_ids = set()
            for lec in tag_info['source_lectures']:
                if lec['lecture_external_id'] not in seen_ids:
                    unique_lectures.append(lec)
                    seen_ids.add(lec['lecture_external_id'])
            
            # Determine overall status (pending if any pending, approved if all approved, etc.)
            if 'pending' in tag_info['statuses']:
                overall_status = 'pending'
            elif 'approved' in tag_info['statuses']:
                overall_status = 'approved'
            elif 'synced' in tag_info['statuses']:
                overall_status = 'synced'
            else:
                overall_status = list(tag_info['statuses'])[0] if tag_info['statuses'] else 'unknown'
            
            tags_list.append({
                'tag_id': tag_info['tag_id'],
                'tag_name_he': tag_info['tag_name_he'],
                'score': tag_info['max_score'],
                'source_count': len(unique_lectures),
                'source_lectures': unique_lectures,
                'rationales': tag_info['rationales'],
                'status': overall_status,
                'suggestion_ids': tag_info['suggestion_ids']
            })
        
        # Sort tags by score (descending)
        tags_list.sort(key=lambda x: x['score'], reverse=True)
        
        lecturers_list.append({
            'lecturer_external_id': lecturer_info['lecturer_external_id'],
            'lecturer_name': lecturer_info['lecturer_name'],
            'lecture_count': lecturer_info['lecture_count'],
            'tags': tags_list,
            'tag_count': len(tags_list)
        })
    
    # Sort by lecturer name
    lecturers_list.sort(key=lambda x: x['lecturer_name'])
    
    stats = {
        'total_lecturers': len(lecturers_list),
        'total_tags': sum(len(l['tags']) for l in lecturers_list),
        'avg_tags_per_lecturer': sum(len(l['tags']) for l in lecturers_list) / len(lecturers_list) if lecturers_list else 0
    }
    
    return render_template('lecturers.html', lecturers=lecturers_list, stats=stats)

@app.route('/api/sync_airtable', methods=['POST'])
def sync_airtable():
    """Trigger Airtable sync in background."""
    def run_sync():
        print("\n" + "="*80)
        print("🔄 AIRTABLE SYNC TRIGGERED FROM WEB INTERFACE")
        print("="*80)
        print(f"⏰ Started at: {pd.Timestamp.now()}")
        print("📊 Syncing approved suggestions to Airtable...")
        print("="*80 + "\n")
        
        try:
            process = subprocess.Popen(
                ['python', 'sync_to_airtable.py'],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            for line in process.stdout:
                print(line, end='', flush=True)
            
            process.wait(timeout=300)
            
            print("\n" + "="*80)
            print(f"✅ Sync completed with exit code: {process.returncode}")
            print(f"⏰ Finished at: {pd.Timestamp.now()}")
            print("="*80 + "\n")
            
            if process.returncode == 0:
                print("🔄 Reloading data into web viewer...")
                load_data()
                print("✅ Data reloaded successfully!")
            else:
                print(f"❌ Sync failed with exit code: {process.returncode}")
                
        except subprocess.TimeoutExpired:
            print("\n" + "="*80)
            print("⚠️ Sync timed out after 5 minutes")
            print("="*80 + "\n")
            process.kill()
        except Exception as e:
            print("\n" + "="*80)
            print(f"❌ Error running sync: {e}")
            print("="*80 + "\n")
    
    thread = threading.Thread(target=run_sync)
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'status': 'started',
        'message': 'Airtable sync started! Check the console logs to see progress.'
    })

@app.route('/rerun', methods=['POST'])
def rerun_batch():
    """Trigger batch processing in background."""
    def run_batch():
        print("\n" + "="*80)
        print("🔄 RERUN TRIGGERED FROM WEB INTERFACE")
        print("="*80)
        print(f"⏰ Started at: {pd.Timestamp.now()}")
        print("📊 Running batch processing with live logs below...")
        print("="*80 + "\n")
        
        try:
            process = subprocess.Popen(
                ['python', 'src/main.py'],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            for line in process.stdout:
                print(line, end='', flush=True)
            
            process.wait(timeout=300)
            
            print("\n" + "="*80)
            print(f"✅ Batch completed with exit code: {process.returncode}")
            print(f"⏰ Finished at: {pd.Timestamp.now()}")
            print("="*80 + "\n")
            
            if process.returncode == 0:
                print("🔄 Reloading data into web viewer...")
                load_data()
                print("✅ Data reloaded successfully!")
            else:
                print(f"❌ Batch failed with exit code: {process.returncode}")
                
        except subprocess.TimeoutExpired:
            print("\n" + "="*80)
            print("⚠️ Batch processing timed out after 5 minutes")
            print("="*80 + "\n")
            process.kill()
        except Exception as e:
            print("\n" + "="*80)
            print(f"❌ Error running batch: {e}")
            print("="*80 + "\n")
    
    thread = threading.Thread(target=run_batch)
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'status': 'started',
        'message': 'Batch processing started! Check the console logs to see progress in real-time.'
    })

if __name__ == '__main__':
    if load_data():
        print("Starting web viewer on http://0.0.0.0:5000")
        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
        print("Failed to load data. Exiting.")
