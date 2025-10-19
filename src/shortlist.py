"""
Shortlist generator for candidate tag selection.
Reduces LLM token usage by ~4x while maintaining high recall.
"""
import re
import numpy as np
from typing import Dict, List, Set, Any, Optional, Tuple
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)


class ShortlistGenerator:
    """Generates a recall-first shortlist of candidate tags per lecture."""
    
    def __init__(
        self,
        k_proto: int = 12,
        k_label: int = 10, 
        k_prior: int = 5,
        max_candidates: int = 25,
        hard_lecture_max: int = 40,
        proto_threshold: float = 0.42,
        label_threshold: float = 0.46,
        hard_lecture_threshold: float = 0.55
    ):
        self.k_proto = k_proto
        self.k_label = k_label
        self.k_prior = k_prior
        self.max_candidates = max_candidates
        self.hard_lecture_max = hard_lecture_max
        self.proto_threshold = proto_threshold
        self.label_threshold = label_threshold
        self.hard_lecture_threshold = hard_lecture_threshold
        
    def find_keyword_hits(
        self, 
        lecture: Dict[str, Any], 
        tags: List[Dict[str, Any]]
    ) -> Set[str]:
        """Find tags whose name or synonyms appear in lecture title/description."""
        hits = set()
        
        # Combine lecture text
        lecture_text = f"{lecture.get('lecture_title', '')} {lecture.get('lecture_description', '')}".lower()
        
        # Normalize Hebrew text for better matching
        lecture_text = self._normalize_hebrew(lecture_text)
        
        for tag in tags:
            tag_id = tag['tag_id']
            
            # Check tag name
            name = self._normalize_hebrew(tag.get('name_he', '').lower())
            if name and name in lecture_text:
                hits.add(tag_id)
                continue
                
            # Check synonyms
            synonyms = tag.get('synonyms_he', '')
            if synonyms:
                for synonym in synonyms.split(','):
                    synonym = self._normalize_hebrew(synonym.strip().lower())
                    if synonym and synonym in lecture_text:
                        hits.add(tag_id)
                        break
                        
        return hits
    
    def _normalize_hebrew(self, text: str) -> str:
        """Normalize Hebrew text for matching."""
        # Remove niqqud (Hebrew diacritics)
        text = re.sub(r'[\u0591-\u05C7]', '', text)
        # Normalize quotes and punctuation
        text = re.sub(r'[״""\'`]', '', text)
        return text
    
    def compute_label_similarities(
        self,
        lecture_embedding: np.ndarray,
        tag_embeddings: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """Compute cosine similarity between lecture and tag label embeddings."""
        scores = {}
        
        for tag_id, tag_embedding in tag_embeddings.items():
            if tag_embedding is not None and lecture_embedding is not None:
                sim = cosine_similarity(
                    lecture_embedding.reshape(1, -1),
                    tag_embedding.reshape(1, -1)
                )[0, 0]
                scores[tag_id] = float(sim)
            else:
                scores[tag_id] = 0.0
                
        return scores
    
    def compute_prototype_similarities(
        self,
        lecture_embedding: np.ndarray,
        prototype_embeddings: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """Compute cosine similarity between lecture and tag prototype embeddings."""
        scores = {}
        
        for tag_id, proto_embedding in prototype_embeddings.items():
            if proto_embedding is not None and lecture_embedding is not None:
                sim = cosine_similarity(
                    lecture_embedding.reshape(1, -1),
                    proto_embedding.reshape(1, -1)
                )[0, 0]
                scores[tag_id] = float(sim)
            else:
                scores[tag_id] = 0.0
                
        return scores
    
    def get_lecturer_priors(
        self,
        lecturer_name: str,
        lecturer_tag_history: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """Get tag co-occurrence priors for a lecturer."""
        if not lecturer_name or lecturer_name not in lecturer_tag_history:
            return {}
        return lecturer_tag_history.get(lecturer_name, {})
    
    def generate_shortlist(
        self,
        lecture: Dict[str, Any],
        tags: List[Dict[str, Any]],
        lecture_embedding: Optional[np.ndarray] = None,
        tag_embeddings: Optional[Dict[str, np.ndarray]] = None,
        prototype_embeddings: Optional[Dict[str, np.ndarray]] = None,
        lecturer_tag_history: Optional[Dict[str, Dict[str, float]]] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Generate a recall-first shortlist of candidate tags.
        
        Returns:
            - List of candidate tag dictionaries
            - Debug info dict with signal breakdowns
        """
        debug_info = {
            'keyword_hits': [],
            'top_proto': [],
            'top_label': [],
            'top_prior': [],
            'must_proto': [],
            'must_label': [],
            'is_hard_lecture': False,
            'final_count': 0
        }
        
        # 1. Keyword hits (no cap)
        kw_hits = self.find_keyword_hits(lecture, tags)
        debug_info['keyword_hits'] = list(kw_hits)
        
        # 2. Prototype similarities (if available)
        top_proto = set()
        must_proto = set()
        if lecture_embedding is not None and prototype_embeddings:
            proto_scores = self.compute_prototype_similarities(
                lecture_embedding, prototype_embeddings
            )
            
            # Top-K by prototype
            sorted_proto = sorted(proto_scores.items(), key=lambda x: x[1], reverse=True)
            top_proto = {tid for tid, _ in sorted_proto[:self.k_proto]}
            debug_info['top_proto'] = [(tid, score) for tid, score in sorted_proto[:self.k_proto]]
            
            # Must-include by threshold
            must_proto = {tid for tid, score in proto_scores.items() if score >= self.proto_threshold}
            debug_info['must_proto'] = [tid for tid in must_proto]
        
        # 3. Label similarities  
        top_label = set()
        must_label = set()
        if lecture_embedding is not None and tag_embeddings:
            label_scores = self.compute_label_similarities(
                lecture_embedding, tag_embeddings
            )
            
            # Top-K by label
            sorted_label = sorted(label_scores.items(), key=lambda x: x[1], reverse=True)
            top_label = {tid for tid, _ in sorted_label[:self.k_label]}
            debug_info['top_label'] = [(tid, score) for tid, score in sorted_label[:self.k_label]]
            
            # Must-include by threshold
            must_label = {tid for tid, score in label_scores.items() if score >= self.label_threshold}
            debug_info['must_label'] = [tid for tid in must_label]
        
        # 4. Lecturer priors
        top_prior = set()
        if lecturer_tag_history:
            lecturer_name = lecture.get('lecturer_name')
            priors = self.get_lecturer_priors(lecturer_name, lecturer_tag_history)
            if priors:
                sorted_priors = sorted(priors.items(), key=lambda x: x[1], reverse=True)
                top_prior = {tid for tid, _ in sorted_priors[:self.k_prior]}
                debug_info['top_prior'] = [(tid, score) for tid, score in sorted_priors[:self.k_prior]]
        
        # 5. Union all signals
        candidates = kw_hits | top_proto | top_label | top_prior | must_proto | must_label
        
        # 6. Hard lecture expansion
        is_hard_lecture = False
        if not kw_hits:
            max_proto = max(proto_scores.values()) if proto_scores else 0
            max_label = max(label_scores.values()) if label_scores else 0
            
            if max_proto < self.hard_lecture_threshold and max_label < self.hard_lecture_threshold:
                is_hard_lecture = True
                debug_info['is_hard_lecture'] = True
                
                # Expand top-K for hard lectures
                if proto_scores:
                    sorted_proto = sorted(proto_scores.items(), key=lambda x: x[1], reverse=True)
                    expanded_proto = {tid for tid, _ in sorted_proto[:20]}
                    candidates |= expanded_proto
                    
                if label_scores:
                    sorted_label = sorted(label_scores.items(), key=lambda x: x[1], reverse=True)
                    expanded_label = {tid for tid, _ in sorted_label[:20]}
                    candidates |= expanded_label
        
        # 7. Final trimming by blended score
        max_candidates = self.hard_lecture_max if is_hard_lecture else self.max_candidates
        
        # Create tag lookup
        tag_dict = {tag['tag_id']: tag for tag in tags}
        
        # Compute blended scores for ranking
        def blended_score(tag_id):
            p_score = proto_scores.get(tag_id, 0) if proto_scores else 0
            l_score = label_scores.get(tag_id, 0) if label_scores else 0
            prior = priors.get(tag_id, 0) if priors else 0
            
            # Higher weight for keyword hits
            kw_boost = 1.0 if tag_id in kw_hits else 0
            
            return 0.7 * p_score + 0.2 * l_score + 0.05 * prior + 0.05 * kw_boost
        
        # Sort and trim
        ranked_candidates = sorted(candidates, key=blended_score, reverse=True)
        final_candidates = ranked_candidates[:max_candidates]
        
        # Return tag objects
        candidate_tags = [tag_dict[tid] for tid in final_candidates if tid in tag_dict]
        
        debug_info['final_count'] = len(candidate_tags)
        debug_info['candidate_ids'] = final_candidates
        
        return candidate_tags, debug_info
    
    def build_lecturer_tag_history(
        self,
        lectures: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, float]]:
        """Build lecturer->tag co-occurrence history from tagged lectures."""
        history = {}
        
        for lecture in lectures:
            lecturer = lecture.get('lecturer_name')
            tag_ids = lecture.get('lecture_tag_ids', [])
            
            if not lecturer or not tag_ids:
                continue
                
            if lecturer not in history:
                history[lecturer] = {}
                
            # Count occurrences
            for tag_id in tag_ids:
                if tag_id not in history[lecturer]:
                    history[lecturer][tag_id] = 0
                history[lecturer][tag_id] += 1
        
        # Normalize to frequencies
        for lecturer in history:
            total = sum(history[lecturer].values())
            if total > 0:
                for tag_id in history[lecturer]:
                    history[lecturer][tag_id] /= total
                    
        return history