import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    database_url: str
    openai_api_key: str
    tags_csv_path: str
    output_csv_path: str
    
    use_llm: bool = True
    write_to_db: bool = False
    
    embedding_model: str = "text-embedding-3-large"
    embedding_dimensions: int = 3072
    
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 500
    
    min_examples_for_prototype: int = 5
    low_data_tag_threshold: int = 5
    prototype_weight: float = 0.8
    label_weight: float = 0.2
    
    target_precision: float = 0.90
    min_confidence_threshold: float = 0.60
    high_confidence_threshold: float = 0.80
    llm_borderline_lower: float = 0.50
    llm_borderline_upper: float = 0.80
    
    top_k_tags: int = 7
    min_k_tags: int = 3
    max_llm_candidates: int = 20
    
    batch_size_embeddings: int = 512
    batch_size_llm: int = 1
    
    train_holdout_split: float = 0.8
    
    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            database_url=os.getenv("DATABASE_URL", ""),
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            tags_csv_path=os.getenv("TAGS_CSV_PATH", "data/tags.csv"),
            output_csv_path=os.getenv("OUTPUT_CSV_PATH", "output/tag_suggestions.csv"),
            use_llm=os.getenv("USE_LLM", "true").lower() == "true",
            write_to_db=os.getenv("WRITE_TO_DB", "false").lower() == "true",
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-large"),
            llm_model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        )
    
    def validate(self) -> None:
        if not self.database_url:
            raise ValueError("DATABASE_URL is required")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required")
        if not self.tags_csv_path:
            raise ValueError("TAGS_CSV_PATH is required")
        if not self.output_csv_path:
            raise ValueError("OUTPUT_CSV_PATH is required")
