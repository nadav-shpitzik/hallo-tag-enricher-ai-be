import pandas as pd
import os
import json
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class OutputGenerator:
    def __init__(self, config):
        self.config = config
    
    def save_to_csv(self, suggestions: Dict[int, List[Dict]]) -> None:
        all_suggestions = []
        for lecture_suggestions in suggestions.values():
            all_suggestions.extend(lecture_suggestions)
        
        if not all_suggestions:
            logger.warning("No suggestions to save to CSV")
            return
        
        df = pd.DataFrame(all_suggestions)
        
        df = df[[
            'lecture_id', 
            'lecture_external_id', 
            'tag_id', 
            'tag_name_he', 
            'score', 
            'rationale', 
            'model'
        ]]
        
        dirpath = os.path.dirname(self.config.output_csv_path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        
        df.to_csv(self.config.output_csv_path, index=False, encoding='utf-8-sig')
        
        logger.info(f"Saved {len(all_suggestions)} suggestions to {self.config.output_csv_path}")
    
    def save_to_database(self, suggestions: Dict[int, List[Dict]], db_connection) -> None:
        all_suggestions = []
        for lecture_suggestions in suggestions.values():
            for suggestion in lecture_suggestions:
                db_record = {
                    'lecture_id': suggestion['lecture_id'],
                    'lecture_external_id': suggestion['lecture_external_id'],
                    'tag_id': suggestion['tag_id'],
                    'tag_name_he': suggestion['tag_name_he'],
                    'score': suggestion['score'],
                    'rationale': suggestion['rationale'],
                    'model': suggestion['model'],
                    'sources': json.dumps(['title', 'description'])
                }
                all_suggestions.append(db_record)
        
        if not all_suggestions:
            logger.warning("No suggestions to save to database")
            return
        
        db_connection.create_suggestions_table()
        db_connection.upsert_suggestions(all_suggestions)
    
    def generate_qa_report(
        self, 
        suggestions: Dict[int, List[Dict]], 
        lectures: List[Dict]
    ) -> str:
        total_lectures = len(lectures)
        lectures_with_suggestions = len([s for s in suggestions.values() if s])
        total_suggestions = sum(len(s) for s in suggestions.values())
        
        coverage_pct = (lectures_with_suggestions / total_lectures * 100) if total_lectures > 0 else 0
        avg_suggestions = (total_suggestions / lectures_with_suggestions) if lectures_with_suggestions > 0 else 0
        
        suggestions_per_lecture = [len(s) for s in suggestions.values() if s]
        
        report = f"""
QA Report
=========

Total Lectures: {total_lectures}
Lectures with Suggestions: {lectures_with_suggestions} ({coverage_pct:.1f}%)
Total Suggestions: {total_suggestions}
Average Suggestions per Lecture: {avg_suggestions:.2f}

Suggestions Distribution:
"""
        
        if suggestions_per_lecture:
            from collections import Counter
            dist = Counter(suggestions_per_lecture)
            for num_tags in sorted(dist.keys()):
                count = dist[num_tags]
                pct = count / lectures_with_suggestions * 100
                report += f"  {num_tags} tags: {count} lectures ({pct:.1f}%)\n"
        
        all_scores = [s['score'] for suggestions_list in suggestions.values() 
                     for s in suggestions_list]
        
        if all_scores:
            report += f"""
Score Statistics:
  Min: {min(all_scores):.3f}
  Max: {max(all_scores):.3f}
  Mean: {sum(all_scores) / len(all_scores):.3f}
  Median: {sorted(all_scores)[len(all_scores) // 2]:.3f}
"""
        
        model_counts = {}
        for suggestions_list in suggestions.values():
            for s in suggestions_list:
                model = s['model']
                model_counts[model] = model_counts.get(model, 0) + 1
        
        if model_counts:
            report += "\nModel Distribution:\n"
            for model, count in sorted(model_counts.items(), key=lambda x: x[1], reverse=True):
                pct = count / total_suggestions * 100
                report += f"  {model}: {count} ({pct:.1f}%)\n"
        
        return report
