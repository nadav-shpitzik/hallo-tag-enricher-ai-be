import logging
from typing import List, Dict, Optional
import numpy as np

logger = logging.getLogger(__name__)


class EnsembleScorer:
    def __init__(
        self,
        reasoning_scorer,
        prototype_knn,
        tags_data: Dict[str, Dict],
        config
    ):
        self.reasoning_scorer = reasoning_scorer
        self.prototype_knn = prototype_knn
        self.tags_data = tags_data
        self.config = config
        
        self.reasoning_weight = config.ensemble_reasoning_weight
        self.prototype_weight = config.ensemble_prototype_weight
        self.agreement_bonus = config.ensemble_agreement_bonus
        
        logger.info(
            f"Ensemble scorer initialized: "
            f"reasoning={self.reasoning_weight:.0%}, "
            f"prototype={self.prototype_weight:.0%}, "
            f"agreement_bonus={self.agreement_bonus:.0%}"
        )
    
    def score_lecture(
        self,
        lecture: Dict,
        all_tags: List[Dict],
        lecture_embedding: np.ndarray,
        tag_embeddings: Dict[str, np.ndarray],
        lecturer_profile: Optional[str] = None
    ) -> List[Dict]:
        lecture_id = lecture.get('id')
        
        reasoning_suggestions = self.reasoning_scorer.score_lecture(
            lecture,
            all_tags,
            lecturer_profile,
            None
        )
        
        prototype_scores = self.prototype_knn.score_lecture(
            lecture_embedding,
            tag_embeddings
        )
        
        reasoning_map = {s['tag_id']: s for s in reasoning_suggestions}
        
        combined_suggestions = {}
        
        for tag_id, prototype_score in prototype_scores.items():
            reasoning_sugg = reasoning_map.get(tag_id)
            
            if reasoning_sugg:
                reasoning_score = reasoning_sugg['score']
                
                base_score = (
                    self.reasoning_weight * reasoning_score +
                    self.prototype_weight * prototype_score
                )
                
                final_score = min(1.0, base_score + self.agreement_bonus)
                
                rationale_parts = []
                rationale_parts.append(f"ניקוד משולב: {final_score:.3f}")
                rationale_parts.append(f"מודל חשיבה: {reasoning_score:.3f}")
                rationale_parts.append(f"פרוטוטייפ: {prototype_score:.3f}")
                
                if reasoning_sugg.get('rationale'):
                    rationale_parts.append(f"נימוק: {reasoning_sugg['rationale']}")
                
                combined_suggestions[tag_id] = {
                    'tag_id': tag_id,
                    'tag_name_he': self.tags_data[tag_id]['name_he'],
                    'score': final_score,
                    'rationale': " | ".join(rationale_parts),
                    'model': 'ensemble:reasoning+prototype',
                    'reasoning_score': reasoning_score,
                    'prototype_score': prototype_score,
                    'agreement_bonus_applied': True
                }
            else:
                if prototype_score >= self.config.min_confidence_threshold:
                    weighted_score = self.prototype_weight * prototype_score
                    
                    combined_suggestions[tag_id] = {
                        'tag_id': tag_id,
                        'tag_name_he': self.tags_data[tag_id]['name_he'],
                        'score': weighted_score,
                        'rationale': f"פרוטוטייפ בלבד: {prototype_score:.3f} (משוקלל: {weighted_score:.3f})",
                        'model': 'ensemble:prototype_only',
                        'prototype_score': prototype_score,
                        'agreement_bonus_applied': False
                    }
        
        for tag_id, reasoning_sugg in reasoning_map.items():
            if tag_id not in combined_suggestions:
                reasoning_score = reasoning_sugg['score']
                
                if reasoning_score >= self.config.min_confidence_threshold:
                    weighted_score = self.reasoning_weight * reasoning_score
                    
                    rationale = f"חשיבה בלבד: {reasoning_score:.3f} (משוקלל: {weighted_score:.3f})"
                    if reasoning_sugg.get('rationale'):
                        rationale += f" | {reasoning_sugg['rationale']}"
                    
                    combined_suggestions[tag_id] = {
                        'tag_id': tag_id,
                        'tag_name_he': self.tags_data[tag_id]['name_he'],
                        'score': weighted_score,
                        'rationale': rationale,
                        'model': 'ensemble:reasoning_only',
                        'reasoning_score': reasoning_score,
                        'agreement_bonus_applied': False
                    }
        
        final_suggestions = sorted(
            combined_suggestions.values(),
            key=lambda x: x['score'],
            reverse=True
        )[:self.config.top_k_tags]
        
        agreement_count = sum(1 for s in final_suggestions if s.get('agreement_bonus_applied'))
        logger.debug(
            f"Lecture {lecture_id}: {len(final_suggestions)} suggestions "
            f"({agreement_count} with agreement bonus)"
        )
        
        return final_suggestions
    
    def score_batch(
        self,
        lectures: List[Dict],
        all_tags: List[Dict],
        lecture_embeddings: Dict[int, np.ndarray],
        tag_embeddings: Dict[str, np.ndarray],
        lecturer_profiles: Dict[str, Optional[str]]
    ) -> Dict[int, List[Dict]]:
        all_suggestions = {}
        
        for i, lecture in enumerate(lectures):
            lecture_id = lecture['id']
            
            existing_tags = lecture.get('lecture_tag_ids') or []
            if existing_tags and len(existing_tags) > 0:
                logger.debug(f"Skipping lecture {lecture_id} - already has {len(existing_tags)} tags")
                continue
            
            if lecture_id not in lecture_embeddings:
                logger.warning(f"No embedding for lecture {lecture_id}, skipping")
                continue
            
            lecturer_profile = None
            if lecture.get('lecturer_name'):
                lecturer_profile = lecturer_profiles.get(lecture['lecturer_name'])
            
            suggestions = self.score_lecture(
                lecture,
                all_tags,
                lecture_embeddings[lecture_id],
                tag_embeddings,
                lecturer_profile
            )
            
            if suggestions:
                all_suggestions[lecture_id] = suggestions
            
            if (i + 1) % 10 == 0:
                logger.info(f"Scored {i + 1}/{len(lectures)} lectures with ensemble model")
        
        logger.info(
            f"Generated {sum(len(s) for s in all_suggestions.values())} "
            f"suggestions for {len(all_suggestions)} lectures using ensemble model"
        )
        
        return all_suggestions
