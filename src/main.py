import logging
import sys
import os
from config import Config
from database import DatabaseConnection
from tags_loader import TagsLoader
from embeddings import EmbeddingsGenerator
from prototype_knn import PrototypeKNN
from llm_arbiter import LLMArbiter
from scorer import LectureScorer
from output import OutputGenerator
from lecturer_search import LecturerSearchService
from reasoning_scorer import ReasoningScorer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('batch_run.log', encoding='utf-8')
    ]
)

logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 80)
    logger.info("AI Tag Enrichment Batch - Starting")
    logger.info("=" * 80)
    
    config = Config.from_env()
    
    try:
        config.validate()
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        logger.info("Please set required environment variables (see .env.example)")
        return 1
    
    logger.info(f"Configuration: scoring_mode={config.scoring_mode}, only_untagged={config.only_untagged}, test_mode={config.test_mode}")
    
    logger.info("Step 1: Loading tags from CSV")
    tags_loader = TagsLoader(config.tags_csv_path)
    tags_data = tags_loader.load()
    tag_label_texts = tags_loader.get_tag_label_texts()
    
    logger.info("Step 2: Fetching lectures from database")
    with DatabaseConnection(config.database_url) as db:
        lectures = db.fetch_lectures(only_untagged=config.only_untagged)
    
    # Test mode: limit lectures for testing
    if config.test_mode:
        lectures = lectures[:config.test_mode_limit]
        logger.info(f"Test mode enabled: limited to {config.test_mode_limit} lectures")
    
    if not lectures:
        logger.error("No lectures found in database")
        return 1
    
    if config.scoring_mode == "reasoning":
        logger.info("Step 3: Gathering lecturer profiles (for reasoning mode)")
        lecturer_service = LecturerSearchService()
        lecturer_names = [lec.get('lecturer_name') for lec in lectures if lec.get('lecturer_name')]
        lecturer_profiles = lecturer_service.get_bulk_profiles(lecturer_names)
        
        # Generate embeddings for shortlist (if enabled)
        tags_list = list(tags_data.values())
        
        if config.use_shortlist:
            logger.info("Step 3b: Generating embeddings for shortlist")
            from shortlist import ShortlistGenerator
            
            shortlist_gen = ShortlistGenerator(
                max_candidates=config.max_llm_candidates,
                k_label=10,
                k_proto=12,
                k_prior=5
            )
            embeddings_gen = EmbeddingsGenerator(config.openai_api_key, config.embedding_model)
            
            # Generate lecture embeddings
            lecture_texts = [
                f"כותרת: {lec.get('lecture_title', '')} תיאור: {lec.get('lecture_description', '')}"
                for lec in lectures
            ]
            lecture_embeddings = embeddings_gen.generate_embeddings(lecture_texts, desc="lectures")
            
            # Generate tag label embeddings
            tag_label_texts_list = list(tag_label_texts.values())
            tag_ids_list = list(tag_label_texts.keys())
            tag_embeddings = embeddings_gen.generate_embeddings(tag_label_texts_list, desc="tag labels")
            tag_embeddings_dict = {
                tag_id: tag_embeddings[i] 
                for i, tag_id in enumerate(tag_ids_list)
                if i < len(tag_embeddings)
            }
            
            # Build lecturer tag history
            all_lectures = []
            with DatabaseConnection(config.database_url) as db:
                all_lectures = db.fetch_lectures(only_untagged=False)
            lecturer_history = shortlist_gen.build_lecturer_tag_history(all_lectures)
            
            logger.info(f"Generated embeddings for {len(lecture_embeddings)} lectures and {len(tag_embeddings_dict)} tags")
        
        logger.info("Step 4: Scoring with reasoning model")
        reasoning_scorer = ReasoningScorer(config.llm_model)
        
        if config.use_shortlist:
            # Score with shortlists
            suggestions_dict = {}
            total_tokens_saved = 0
            
            for i, lecture in enumerate(lectures):
                # Generate shortlist for this lecture
                lecture_embedding = lecture_embeddings[i] if i < len(lecture_embeddings) else None
                candidate_tags, debug_info = shortlist_gen.generate_shortlist(
                    lecture,
                    tags_list,
                    lecture_embedding=lecture_embedding,
                    tag_embeddings=tag_embeddings_dict,
                    prototype_embeddings=None,  # Could add if we have prototypes
                    lecturer_tag_history=lecturer_history
                )
                
                # Log shortlist stats periodically
                if i % 10 == 0:
                    logger.info(f"Lecture {i+1}/{len(lectures)}: shortlist has {len(candidate_tags)}/{len(tags_list)} tags")
                    tokens_saved = (len(tags_list) - len(candidate_tags)) * 100  # Rough estimate
                    total_tokens_saved += tokens_saved
                
                # Score with shortlist
                lecturer_profile = lecturer_profiles.get(lecture.get('lecturer_name'))
                suggestions = reasoning_scorer.score_lecture(
                    lecture, 
                    candidate_tags,  # Pass shortlist instead of all tags
                    lecturer_profile
                )
                
                if suggestions:
                    suggestions_dict[lecture['id']] = suggestions
                    
                # Fallback: if no tags selected, retry with full list
                if not suggestions and config.shortlist_fallback:
                    logger.info(f"No tags found for lecture {lecture['id']}, retrying with full tag list")
                    suggestions = reasoning_scorer.score_lecture(
                        lecture,
                        tags_list,  # Full list
                        lecturer_profile
                    )
                    if suggestions:
                        suggestions_dict[lecture['id']] = suggestions
            
            logger.info(f"Estimated tokens saved with shortlist: ~{total_tokens_saved:,}")
        else:
            # Original: score with all tags
            suggestions_dict = reasoning_scorer.score_batch(
                lectures,
                tags_list,
                lecturer_profiles
            )
        
        suggestions = {}
        for lecture in lectures:
            lecture_id = lecture['id']
            if lecture_id in suggestions_dict:
                formatted_suggestions = []
                for sugg in suggestions_dict[lecture_id]:
                    formatted_suggestions.append({
                        'lecture_id': lecture_id,
                        'lecture_external_id': lecture.get('lecture_external_id', ''),
                        'tag_id': sugg['tag_id'],
                        'tag_name_he': sugg['tag_name_he'],
                        'score': sugg['score'],
                        'rationale': sugg['rationale'],
                        'model': sugg['model']
                    })
                suggestions[lecture_id] = formatted_suggestions
    
    elif config.scoring_mode == "prototype":
        logger.info("Step 3: Generating embeddings")
        embeddings_gen = EmbeddingsGenerator(
            config.openai_api_key, 
            config.embedding_model, 
            config.batch_size_embeddings
        )
        
        lecture_embeddings = embeddings_gen.generate_lecture_embeddings(lectures)
        tag_embeddings = embeddings_gen.generate_tag_embeddings(tag_label_texts)
        
        logger.info("Step 4: Building tag prototypes from existing tagged lectures")
        prototype_knn = PrototypeKNN(config)
        prototype_knn.build_prototypes(lectures, lecture_embeddings, tags_data)
        
        if not prototype_knn.tag_prototypes:
            logger.error("No tag prototypes could be built. Need existing tagged lectures.")
            return 1
        
        logger.info("Step 5: Calibrating per-tag thresholds")
        prototype_knn.calibrate_thresholds(lectures, lecture_embeddings, tag_embeddings)
        
        llm_arbiter = None
        if config.use_llm:
            logger.info("Step 6: Initializing LLM arbiter")
            llm_arbiter = LLMArbiter(config.openai_api_key, config)
        else:
            logger.info("Step 6: LLM arbiter disabled (skipping)")
        
        logger.info("Step 7: Scoring all lectures with prototype model")
        scorer = LectureScorer(config, prototype_knn, tags_data)
        suggestions = scorer.score_all_lectures(
            lectures, 
            lecture_embeddings, 
            tag_embeddings,
            llm_arbiter
        )
    
    else:
        logger.error(f"Unknown scoring mode: {config.scoring_mode}. Use 'reasoning' or 'prototype'")
        return 1
    
    logger.info("Step 8: Saving outputs")
    output_gen = OutputGenerator(config)
    
    output_gen.save_to_csv(suggestions)
    
    if config.write_to_db:
        logger.info("Writing suggestions to database")
        with DatabaseConnection(config.database_url) as db:
            output_gen.save_to_database(suggestions, db)
    
    logger.info("Step 9: Generating QA report")
    report = output_gen.generate_qa_report(suggestions, lectures)
    print(report)
    logger.info(report)
    
    logger.info("=" * 80)
    logger.info("AI Tag Enrichment Batch - Complete")
    logger.info("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
