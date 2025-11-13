import logging
from typing import List, Dict, Optional, Any
import numpy as np

from src.telemetry import log_line

logger = logging.getLogger(__name__)

# Margin for guarding agreement bonus - only apply if prototype score is close to threshold
PROTOTYPE_THRESHOLD_MARGIN = 0.03


def _index_by_label(items: List[Dict[str, Any]], score_key: str) -> Dict[str, Dict[str, Any]]:
    """Index items by label_id/tag_id, normalizing score field."""
    out = {}
    for it in items:
        lid = it.get("label_id") or it.get("tag_id")
        if not lid:
            continue
        it = dict(it)
        it["label_id"] = lid
        it["tag_id"] = lid  # Keep both for compatibility
        it["score"] = float(it.get(score_key, it.get("score", 0.0)))
        out[lid] = it
    return out


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
            f"agreement_bonus={self.agreement_bonus:.0%}, "
            f"threshold_margin={PROTOTYPE_THRESHOLD_MARGIN}"
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
        
        # Get reasoning suggestions from per-category reasoning
        reasoning_suggestions = self.reasoning_scorer.score_lecture(
            lecture,
            all_tags,
            lecturer_profile,
            None
        )
        
        # Get prototype scores
        prototype_scores = self.prototype_knn.score_lecture(
            lecture_embedding,
            tag_embeddings
        )
        
        # Build prototype suggestions with thresholds from tags_data
        prototype_suggestions = []
        for tag_id, score in prototype_scores.items():
            if tag_id in self.tags_data:
                tag_data = self.tags_data[tag_id]
                category = tag_data.get('category', 'Unknown')
                
                # Get category-specific threshold
                threshold = self.config.category_thresholds.get(
                    category,
                    self.config.category_thresholds.get('default', 0.60)
                )
                
                prototype_suggestions.append({
                    'tag_id': tag_id,
                    'score': score,
                    'threshold': threshold,
                    'category': category,
                    'tag_name_he': tag_data.get('name_he', '')
                })
        
        # Combine using new logic with backfill and guarded bonus
        combined_suggestions = self._combine_reasoning_and_prototype(
            reasoning_suggestions,
            prototype_suggestions,
            lecture_id
        )
        
        # Apply top-K limit and return
        final_suggestions = combined_suggestions[:self.config.top_k_tags]
        
        return final_suggestions
    
    def _combine_reasoning_and_prototype(
        self,
        reasoning_suggestions: List[Dict[str, Any]],
        prototype_suggestions: List[Dict[str, Any]],
        lecture_id: Any
    ) -> List[Dict[str, Any]]:
        """
        Combine reasoning and prototype suggestions with:
        1. Backfill: Add top-1 prototype if everything is empty
        2. Guarded bonus: Only apply agreement bonus if prototype score >= (threshold - margin)
        
        Returns list sorted desc by combined_score (no K cap here).
        """
        
        # Index inputs
        rmap = _index_by_label(reasoning_suggestions, score_key="score")
        pmap = _index_by_label(prototype_suggestions, score_key="score")
        
        # Backfill: If reasoning is empty → add ONLY top-1 prototype above its threshold
        backfill_triggered = False
        if not reasoning_suggestions:
            # Pick highest proto that is >= its own threshold
            eligible = [
                p for p in prototype_suggestions 
                if float(p.get("score", 0.0)) >= float(p.get("threshold", 1.0))
            ]
            if eligible:
                best = max(eligible, key=lambda x: float(x["score"]))
                best_id = best.get("tag_id") or best.get("label_id")
                
                # Synthesize a minimal reasoning entry with zero reasoning score
                rmap[best_id] = {
                    "label_id": best_id,
                    "tag_id": best_id,
                    "score": 0.0,  # reasoning_score
                    "category": best.get("category"),
                    "rationale": "backfill: prototype top-1",
                    "reasons": ["backfill_prototype"]
                }
                
                # Keep only this one prototype
                pmap = {
                    best_id: {
                        **best,
                        "label_id": best_id,
                        "tag_id": best_id,
                        "score": float(best["score"]),
                        "threshold": float(best.get("threshold", 1.0))
                    }
                }
                backfill_triggered = True
                
                logger.info(
                    f"Lecture {lecture_id}: Backfill triggered - added top prototype tag {best_id} "
                    f"(score={best['score']:.3f}, threshold={best.get('threshold', 1.0):.3f})"
                )
        
        # Union of all labels mentioned (after potential backfill)
        labels = set(rmap.keys()) | set(pmap.keys())
        
        # Combine all labels
        combined: List[Dict[str, Any]] = []
        agreement_bonus_count = 0
        
        for lid in labels:
            r = rmap.get(lid)
            p = pmap.get(lid)
            
            reasoning_score = float(r["score"]) if r else 0.0
            prototype_score = float(p["score"]) if p else 0.0
            proto_threshold = float(p.get("threshold", 1.0)) if p else 1.0
            
            # Base score from weighted average
            base = (
                self.reasoning_weight * reasoning_score +
                self.prototype_weight * prototype_score
            )
            
            # Guarded agreement bonus - only if prototype is actually good
            agreement_bonus_applied = False
            if r and p:
                # Only apply bonus if prototype score >= (threshold - margin)
                if prototype_score >= (proto_threshold - PROTOTYPE_THRESHOLD_MARGIN):
                    base += self.agreement_bonus
                    agreement_bonus_applied = True
                    agreement_bonus_count += 1
            
            # Prototype-only filtering: Must meet category threshold
            if not r and p:
                if prototype_score < proto_threshold:
                    # Skip this prototype-only suggestion - doesn't meet threshold
                    continue
            
            # Clamp to [0, 1]
            combined_score = min(1.0, max(0.0, base))
            
            # Build rationale
            rationale_parts = []
            rationale_parts.append(f"ניקוד משולב: {combined_score:.3f}")
            
            if r and p:
                rationale_parts.append(f"מודל חשיבה: {reasoning_score:.3f}")
                rationale_parts.append(f"פרוטוטייפ: {prototype_score:.3f}")
                if agreement_bonus_applied:
                    rationale_parts.append(f"בונוס הסכמה: +{self.agreement_bonus:.2f}")
                model = 'ensemble:reasoning+prototype'
            elif r:
                rationale_parts.append(f"חשיבה בלבד: {reasoning_score:.3f}")
                model = 'ensemble:reasoning_only'
            else:
                rationale_parts.append(f"פרוטוטייפ בלבד: {prototype_score:.3f}")
                model = 'ensemble:prototype_only'
            
            if r and r.get('rationale'):
                rationale_parts.append(f"נימוק: {r['rationale']}")
            
            # Get tag metadata
            tag_data = self.tags_data.get(lid, {})
            
            combined.append({
                "tag_id": lid,
                "tag_name_he": tag_data.get('name_he', ''),
                "category": r.get("category") if r else p.get("category") if p else None,
                "score": combined_score,
                "reasoning_score": reasoning_score,
                "prototype_score": prototype_score,
                "prototype_threshold": proto_threshold,
                "agreement_bonus_applied": agreement_bonus_applied,
                "rationale": " | ".join(rationale_parts),
                "model": model,
                "reasons": list(set(
                    (r.get("reasons", []) if r else []) + 
                    (p.get("reasons", []) if p else [])
                ))
            })
        
        # Sort descending by combined score
        combined.sort(key=lambda x: x["score"], reverse=True)
        
        # Log summary
        top3_summary = [(s['tag_id'], f"{s['score']:.3f}") for s in combined[:3]]
        logger.info(
            f"Lecture {lecture_id}: Combined {len(combined)} suggestions - "
            f"backfill={backfill_triggered}, agreement_bonus={agreement_bonus_count}, "
            f"top3={top3_summary}"
        )
        
        # Log telemetry for ensemble result
        log_line({
            "kind": "ensemble_result",
            "lecture_id": lecture_id,
            "backfill_triggered": backfill_triggered,
            "agreement_count": agreement_bonus_count,
            "top3": [
                {
                    "label_id": s["tag_id"],
                    "combined": s["score"],
                    "r": s["reasoning_score"],
                    "p": s["prototype_score"],
                    "pth": s["prototype_threshold"]
                }
                for s in combined[:3]
            ],
            "total_returned": len(combined)
        })
        
        return combined
    
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
