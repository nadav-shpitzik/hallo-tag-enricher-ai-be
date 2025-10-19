import numpy as np
from openai import OpenAI
from typing import List, Dict
import logging
import time

logger = logging.getLogger(__name__)


class EmbeddingsGenerator:
    def __init__(self, api_key: str, model: str = "text-embedding-3-large", batch_size: int = 512):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.batch_size = batch_size
    
    def generate_embeddings(self, texts: List[str], desc: str = "items") -> np.ndarray:
        all_embeddings = []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        
        logger.info(f"Generating embeddings for {len(texts)} {desc} in {total_batches} batches")
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1
            
            try:
                response = self.client.embeddings.create(
                    input=batch,
                    model=self.model
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                
                logger.info(f"Batch {batch_num}/{total_batches} complete ({len(batch_embeddings)} embeddings)")
                
                if i + self.batch_size < len(texts):
                    time.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"Error generating embeddings for batch {batch_num}: {e}")
                raise
        
        embeddings_array = np.array(all_embeddings, dtype=np.float32)
        logger.info(f"Generated {embeddings_array.shape[0]} embeddings of dimension {embeddings_array.shape[1]}")
        
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
