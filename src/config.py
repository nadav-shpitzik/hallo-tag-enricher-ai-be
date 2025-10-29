import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


class Config:
    def __init__(self, **kwargs):
        # Core settings
        self.openai_api_key = kwargs.get('openai_api_key') or os.getenv("OPENAI_API_KEY", "")
        
        # Feature flags
        self.use_llm = kwargs.get('use_llm', os.getenv("USE_LLM", "true").lower() == "true")
        
        # Scoring mode: "full_quality" (prototype + arbiter), "reasoning" (pure LLM), "fast" (prototype only)
        self.scoring_mode = kwargs.get('scoring_mode', os.getenv("SCORING_MODE", "full_quality"))
        
        self.use_shortlist = kwargs.get('use_shortlist', os.getenv("USE_SHORTLIST", "true").lower() == "true")
        self.shortlist_fallback = kwargs.get('shortlist_fallback', os.getenv("SHORTLIST_FALLBACK", "true").lower() == "true")
        self.test_mode = kwargs.get('test_mode', os.getenv("TEST_MODE", "false").lower() == "true")
        self.test_mode_limit = kwargs.get('test_mode_limit', int(os.getenv("TEST_MODE_LIMIT", "30")))
        
        # Model settings
        self.embedding_model = kwargs.get('embedding_model', os.getenv("EMBEDDING_MODEL", "text-embedding-3-large"))
        self.embedding_dimensions = 3072
        self.llm_model = kwargs.get('llm_model', os.getenv("LLM_MODEL", "gpt-4o-mini"))
        self.llm_temperature = 0.0
        self.llm_max_tokens = 500
        
        # Prototype settings
        self.min_examples_for_prototype = 5
        self.low_data_tag_threshold = 5
        self.prototype_weight = 0.8
        self.label_weight = 0.2
        
        # Threshold settings
        self.target_precision = 0.90
        self.min_confidence_threshold = 0.60
        self.high_confidence_threshold = 0.80
        self.llm_borderline_lower = 0.50
        self.llm_borderline_upper = 0.80
        
        # Reasoning mode calibration (LLMs tend to be over-confident)
        self.reasoning_confidence_scale = float(kwargs.get('reasoning_confidence_scale', 
                                                          os.getenv("REASONING_CONFIDENCE_SCALE", "0.85")))
        
        # Category-aware thresholds (v2)
        self.category_thresholds = {
            'Topic': 0.65,
            'Persona': 0.60,
            'Tone': 0.55,
            'Format': 0.50,
            'Audience': 0.60,
            'default': 0.60
        }
        
        # Related lectures boost settings
        self.related_lecture_boost = 0.10
        self.related_lecture_min_overlap = 1
        
        # Tag selection settings
        self.top_k_tags = 7
        self.min_k_tags = 3
        self.max_llm_candidates = 30
        
        # Batch settings
        self.batch_size_embeddings = 512
        self.batch_size_llm = 1
        
        # Training settings
        self.train_holdout_split = 0.8
    
    @classmethod
    def from_env(cls) -> "Config":
        return cls()
    
    def validate(self) -> None:
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required")
