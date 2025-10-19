import numpy as np
from typing import List, Dict, Tuple, Set
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class PrototypeKNN:
    def __init__(self, config):
        self.config = config
        self.tag_prototypes = {}
        self.tag_thresholds = {}
        self.tag_stats = {}
    
    def build_prototypes(
        self, 
        lectures: List[Dict], 
        lecture_embeddings: Dict[int, np.ndarray],
        tags_data: Dict[str, Dict]
    ) -> None:
        tagged_lectures = self._extract_tagged_lectures(lectures)
        
        logger.info(f"Building prototypes from {len(tagged_lectures)} tagged lectures")
        
        tag_vectors = defaultdict(list)
        
        for lecture in tagged_lectures:
            lecture_id = lecture['id']
            if lecture_id not in lecture_embeddings:
                continue
            
            embedding = lecture_embeddings[lecture_id]
            tag_ids = lecture.get('lecture_tag_ids', [])
            
            if isinstance(tag_ids, str):
                tag_ids = [t.strip() for t in tag_ids.split(',') if t.strip()]
            
            for tag_id in tag_ids:
                tag_id = str(tag_id).strip()
                if tag_id in tags_data:
                    tag_vectors[tag_id].append(embedding)
        
        for tag_id, vectors in tag_vectors.items():
            if len(vectors) >= 1:
                centroid = np.mean(vectors, axis=0)
                self.tag_prototypes[tag_id] = centroid
                self.tag_stats[tag_id] = {
                    'num_examples': len(vectors),
                    'is_low_data': len(vectors) < self.config.low_data_tag_threshold
                }
        
        logger.info(f"Built {len(self.tag_prototypes)} tag prototypes")
        low_data_count = sum(1 for s in self.tag_stats.values() if s['is_low_data'])
        logger.info(f"Low-data tags (<{self.config.low_data_tag_threshold} examples): {low_data_count}")
    
    def calibrate_thresholds(
        self,
        lectures: List[Dict],
        lecture_embeddings: Dict[int, np.ndarray],
        tag_embeddings: Dict[str, np.ndarray]
    ) -> None:
        tagged_lectures = self._extract_tagged_lectures(lectures)
        
        if not tagged_lectures:
            logger.warning("No tagged lectures for threshold calibration, using default thresholds")
            for tag_id in self.tag_prototypes.keys():
                self.tag_thresholds[tag_id] = self.config.min_confidence_threshold
            return
        
        split_idx = int(len(tagged_lectures) * self.config.train_holdout_split)
        holdout_lectures = tagged_lectures[split_idx:]
        
        logger.info(f"Calibrating thresholds on {len(holdout_lectures)} holdout lectures")
        
        tag_scores_positive = defaultdict(list)
        tag_scores_negative = defaultdict(list)
        
        for lecture in holdout_lectures:
            lecture_id = lecture['id']
            if lecture_id not in lecture_embeddings:
                continue
            
            embedding = lecture_embeddings[lecture_id]
            tag_ids_raw = lecture.get('lecture_tag_ids', [])
            
            if isinstance(tag_ids_raw, str):
                tag_ids = {t.strip() for t in tag_ids_raw.split(',') if t.strip()}
            else:
                tag_ids = set(tag_ids_raw)
            
            tag_ids = {str(t) for t in tag_ids}
            
            for tag_id in self.tag_prototypes.keys():
                score = self._compute_score(embedding, tag_id, tag_embeddings)
                
                if tag_id in tag_ids:
                    tag_scores_positive[tag_id].append(score)
                else:
                    tag_scores_negative[tag_id].append(score)
        
        for tag_id in self.tag_prototypes.keys():
            positives = sorted(tag_scores_positive.get(tag_id, []), reverse=True)
            negatives = sorted(tag_scores_negative.get(tag_id, []), reverse=True)
            
            if not positives:
                self.tag_thresholds[tag_id] = self.config.min_confidence_threshold
                continue
            
            threshold = self._find_precision_threshold(
                positives, 
                negatives, 
                self.config.target_precision
            )
            
            threshold = max(threshold, self.config.min_confidence_threshold)
            self.tag_thresholds[tag_id] = threshold
        
        logger.info(f"Calibrated thresholds for {len(self.tag_thresholds)} tags")
    
    def _find_precision_threshold(
        self, 
        positives: List[float], 
        negatives: List[float], 
        target_precision: float
    ) -> float:
        candidates = sorted(set(positives + negatives), reverse=True)
        
        for threshold in candidates:
            tp = sum(1 for s in positives if s >= threshold)
            fp = sum(1 for s in negatives if s >= threshold)
            
            if tp + fp == 0:
                continue
            
            precision = tp / (tp + fp)
            if precision >= target_precision:
                return threshold
        
        return max(candidates) if candidates else 0.5
    
    def _compute_score(
        self, 
        lecture_embedding: np.ndarray, 
        tag_id: str, 
        tag_embeddings: Dict[str, np.ndarray]
    ) -> float:
        if tag_id not in self.tag_prototypes:
            return 0.0
        
        prototype = self.tag_prototypes[tag_id]
        
        proto_sim = self._cosine_similarity(lecture_embedding, prototype)
        
        stats = self.tag_stats.get(tag_id, {})
        if stats.get('is_low_data', False) and tag_id in tag_embeddings:
            label_sim = self._cosine_similarity(lecture_embedding, tag_embeddings[tag_id])
            score = (self.config.prototype_weight * proto_sim + 
                    self.config.label_weight * label_sim)
        else:
            score = proto_sim
        
        return float(score)
    
    def score_lecture(
        self, 
        lecture_embedding: np.ndarray, 
        tag_embeddings: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        scores = {}
        
        for tag_id in self.tag_prototypes.keys():
            score = self._compute_score(lecture_embedding, tag_id, tag_embeddings)
            threshold = self.tag_thresholds.get(tag_id, self.config.min_confidence_threshold)
            
            if score >= threshold:
                scores[tag_id] = score
        
        return scores
    
    def _extract_tagged_lectures(self, lectures: List[Dict]) -> List[Dict]:
        tagged = []
        for lecture in lectures:
            tag_ids = lecture.get('lecture_tag_ids')
            if tag_ids:
                if isinstance(tag_ids, str):
                    tag_ids = [t.strip() for t in tag_ids.split(',') if t.strip()]
                if tag_ids:
                    tagged.append(lecture)
        return tagged
    
    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-10)
        vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-10)
        return float(np.dot(vec1_norm, vec2_norm))
