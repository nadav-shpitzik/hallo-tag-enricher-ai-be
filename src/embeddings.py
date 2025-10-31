import numpy as np
from openai import OpenAI
from typing import List, Dict
import logging
import time
from src.logging_utils import StructuredLogger, track_operation

logger = StructuredLogger(__name__)


class EmbeddingsGenerator:
    def __init__(self, api_key: str, model: str = "text-embedding-3-large", batch_size: int = 512):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.batch_size = batch_size
    
    def _estimate_tokens(self, texts: List[str]) -> int:
        """Estimate tokens from text (rough approximation: 1 token ~ 4 chars)."""
        total_chars = sum(len(text) for text in texts)
        return total_chars // 4
    
    def generate_embeddings(self, texts: List[str], desc: str = "items") -> np.ndarray:
        all_embeddings = []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        total_tokens = 0
        
        logger.info(
            f"Generating embeddings",
            num_texts=len(texts),
            description=desc,
            total_batches=total_batches,
            model=self.model
        )
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1
            batch_start_time = time.time()
            
            try:
                with track_operation(f"embedding_batch_{batch_num}", logger, batch_size=len(batch)):
                    response = self.client.embeddings.create(
                        input=batch,
                        model=self.model
                    )
                    
                    batch_embeddings = [item.embedding for item in response.data]
                    all_embeddings.extend(batch_embeddings)
                    
                    # Track token usage (with fallback estimation if usage not provided)
                    if hasattr(response, 'usage') and response.usage:
                        tokens_used = response.usage.total_tokens
                    else:
                        # Estimate tokens when API doesn't provide usage data
                        tokens_used = self._estimate_tokens(batch)
                    
                    total_tokens += tokens_used
                    
                    # Estimate cost (text-embedding-3-large: ~$0.13 per 1M tokens)
                    cost_per_million = 0.13 if 'large' in self.model else 0.02
                    batch_cost = (tokens_used / 1_000_000) * cost_per_million
                    batch_duration = time.time() - batch_start_time
                    
                    logger.info(
                        f"Embedding batch completed",
                        batch_num=batch_num,
                        total_batches=total_batches,
                        num_embeddings=len(batch_embeddings),
                        tokens=tokens_used,
                        estimated_cost_usd=round(batch_cost, 6),
                        duration_ms=round(batch_duration * 1000, 2)
                    )
                    
                if i + self.batch_size < len(texts):
                    time.sleep(0.1)
                    
            except Exception as e:
                logger.error(
                    f"Error generating embeddings",
                    batch_num=batch_num,
                    error_type=type(e).__name__,
                    error_message=str(e)
                )
                raise
        
        embeddings_array = np.array(all_embeddings, dtype=np.float32)
        total_cost = (total_tokens / 1_000_000) * (0.13 if 'large' in self.model else 0.02)
        
        logger.info(
            f"Embeddings generation completed",
            num_embeddings=embeddings_array.shape[0],
            dimensions=embeddings_array.shape[1],
            total_tokens=total_tokens,
            total_cost_usd=round(total_cost, 6)
        )
        
        return embeddings_array
    
    def create_lecture_text(self, title: str, description: str) -> str:
        title = title or ""
        description = description or ""
        return f"[כותרת] {title}\n[תיאור] {description}"
    
    def generate_lecture_embeddings(self, lectures: List[Dict]) -> Dict[int, np.ndarray]:
        lecture_texts = []
        lecture_ids = []
        
        for lecture in lectures:
            text = self.create_lecture_text(
                lecture.get('lecture_title', ''),
                lecture.get('lecture_description', '')
            )
            lecture_texts.append(text)
            lecture_ids.append(lecture['id'])
        
        embeddings = self.generate_embeddings(lecture_texts, "lectures")
        
        return {lecture_id: embeddings[i] for i, lecture_id in enumerate(lecture_ids)}
    
    def generate_tag_embeddings(self, tag_label_texts: Dict[str, str]) -> Dict[str, np.ndarray]:
        tag_ids = list(tag_label_texts.keys())
        tag_texts = [tag_label_texts[tag_id] for tag_id in tag_ids]
        
        embeddings = self.generate_embeddings(tag_texts, "tag labels")
        
        return {tag_id: embeddings[i] for i, tag_id in enumerate(tag_ids)}


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-10)
    vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-10)
    return float(np.dot(vec1_norm, vec2_norm))
