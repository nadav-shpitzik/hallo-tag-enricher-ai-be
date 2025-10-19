import numpy as np
from typing import List, Dict, Tuple
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class LectureScorer:
    def __init__(self, config, prototype_knn, tags_data):
        self.config = config
        self.prototype_knn = prototype_knn
        self.tags_data = tags_data
    
    def score_all_lectures(
        self,
        lectures: List[Dict],
        lecture_embeddings: Dict[int, np.ndarray],
        tag_embeddings: Dict[str, np.ndarray],
        llm_arbiter=None
    ) -> Dict[int, List[Dict]]:
        results = {}
        
        for i, lecture in enumerate(lectures):
            lecture_id = lecture['id']
            
            if lecture_id not in lecture_embeddings:
                logger.warning(f"No embedding for lecture {lecture_id}, skipping")
                continue
            
            existing_tag_ids = set()
            raw_tags = lecture.get('lecture_tag_ids', [])
            if raw_tags:
                if isinstance(raw_tags, str):
                    existing_tag_ids = {t.strip() for t in raw_tags.split(',') if t.strip()}
                elif isinstance(raw_tags, list):
                    existing_tag_ids = {str(t).strip() for t in raw_tags if t}
                else:
                    logger.warning(f"Unexpected type for lecture_tag_ids: {type(raw_tags)}")
            
            if existing_tag_ids:
                continue
            
            embedding = lecture_embeddings[lecture_id]
            
            scores = self.prototype_knn.score_lecture(embedding, tag_embeddings)
            
            if self.config.use_llm and llm_arbiter and scores:
                llm_selected = llm_arbiter.refine_suggestions(
                    lecture.get('lecture_title', ''),
                    lecture.get('lecture_description', ''),
                    self.tags_data,
                    scores
                )
                
                for tag_id in list(scores.keys()):
                    score = scores[tag_id]
                    if (self.config.llm_borderline_lower <= score < self.config.llm_borderline_upper):
                        if tag_id not in llm_selected:
                            del scores[tag_id]
            
            top_tags = self._select_top_k(scores)
            
            suggestions = []
            for tag_id, score in top_tags:
                rationale = self._create_rationale(
                    lecture_id, 
                    tag_id, 
                    score, 
                    lectures, 
                    lecture_embeddings
                )
                
                model_name = self._get_model_name(score)
                
                suggestions.append({
                    'lecture_id': lecture_id,
                    'lecture_external_id': lecture.get('lecture_external_id', ''),
                    'tag_id': tag_id,
                    'tag_name_he': self.tags_data[tag_id]['name_he'],
                    'score': score,
                    'rationale': rationale,
                    'model': model_name
                })
            
            results[lecture_id] = suggestions
            
            if (i + 1) % 100 == 0:
                logger.info(f"Scored {i + 1}/{len(lectures)} lectures")
        
        total_suggestions = sum(len(s) for s in results.values())
        logger.info(f"Generated {total_suggestions} total suggestions for {len(results)} lectures")
        
        return results
    
    def _select_top_k(self, scores: Dict[str, float]) -> List[Tuple[str, float]]:
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        filtered = [
            (tag_id, score) for tag_id, score in sorted_scores 
            if score >= self.config.min_confidence_threshold
        ]
        
        return filtered[:self.config.top_k_tags]
    
    def _create_rationale(
        self,
        lecture_id: int,
        tag_id: str,
        score: float,
        lectures: List[Dict],
        lecture_embeddings: Dict[int, np.ndarray]
    ) -> str:
        stats = self.prototype_knn.tag_stats.get(tag_id, {})
        num_examples = stats.get('num_examples', 0)
        is_low_data = stats.get('is_low_data', False)
        
        tag_name = self.tags_data[tag_id]['name_he']
        
        if is_low_data:
            return f"דמיון גבוה לתיאור התגית '{tag_name}' (דוגמאות מעטות: {num_examples})"
        
        neighbors = self._find_nearest_neighbors(
            lecture_id, 
            tag_id, 
            lectures, 
            lecture_embeddings,
            k=3
        )
        
        neighbor_ids_str = ", ".join(str(nid) for nid in neighbors)
        return f"דמיון גבוה להרצאות שתוייגו ב־'{tag_name}' (שכנים: {neighbor_ids_str})"
    
    def _find_nearest_neighbors(
        self,
        lecture_id: int,
        tag_id: str,
        lectures: List[Dict],
        lecture_embeddings: Dict[int, np.ndarray],
        k: int = 3
    ) -> List[int]:
        if lecture_id not in lecture_embeddings:
            return []
        
        target_embedding = lecture_embeddings[lecture_id]
        
        tagged_with_tag = []
        for lecture in lectures:
            lid = lecture['id']
            if lid == lecture_id or lid not in lecture_embeddings:
                continue
            
            tag_ids = lecture.get('lecture_tag_ids', [])
            if isinstance(tag_ids, str):
                tag_ids = [t.strip() for t in tag_ids.split(',') if t.strip()]
            
            tag_ids = [str(t) for t in tag_ids]
            
            if tag_id in tag_ids:
                embedding = lecture_embeddings[lid]
                similarity = self._cosine_similarity(target_embedding, embedding)
                tagged_with_tag.append((lid, similarity))
        
        tagged_with_tag.sort(key=lambda x: x[1], reverse=True)
        
        return [lid for lid, _ in tagged_with_tag[:k]]
    
    def _get_model_name(self, score: float) -> str:
        if score >= self.config.high_confidence_threshold:
            return "prototype"
        elif self.config.use_llm and score >= self.config.llm_borderline_lower:
            return f"prototype+llm:{self.config.llm_model}"
        else:
            return "prototype"
    
    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-10)
        vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-10)
        return float(np.dot(vec1_norm, vec2_norm))
