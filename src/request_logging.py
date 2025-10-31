import time
import uuid
from flask import request, g
from functools import wraps
from typing import Callable, Tuple
from src.logging_utils import StructuredLogger, set_request_context, clear_request_context, sanitize_for_logging

logger = StructuredLogger(__name__)


def log_request_middleware(app):
    """Flask middleware to log all requests and responses."""
    
    @app.before_request
    def before_request():
        """Log incoming request and set context."""
        g.start_time = time.time()
        
        request_id = request.headers.get('X-Request-ID') or str(uuid.uuid4())
        g.request_id = request_id
        
        set_request_context(request_id)
        
        logger.info(
            f"Incoming {request.method} request to {request.path}",
            method=request.method,
            path=request.path,
            endpoint=request.endpoint,
            remote_addr=request.remote_addr,
            user_agent=request.headers.get('User-Agent', 'unknown')[:100]
        )
    
    @app.after_request
    def after_request(response):
        """Log response details."""
        if hasattr(g, 'start_time'):
            duration = time.time() - g.start_time
            
            logger.info(
                f"{request.method} {request.path} completed",
                method=request.method,
                path=request.path,
                status_code=response.status_code,
                duration_ms=round(duration * 1000, 2),
                response_size=response.content_length
            )
        
        if hasattr(g, 'request_id'):
            response.headers['X-Request-ID'] = g.request_id
        
        clear_request_context()
        
        return response
    
    @app.teardown_request
    def teardown_request(exception=None):
        """Clean up request context on error."""
        if exception:
            logger.error(
                f"Request failed with exception",
                error_type=type(exception).__name__,
                error_message=str(exception)
            )
        clear_request_context()


def log_endpoint(endpoint_name: str):
    """Decorator to add detailed logging to specific endpoints."""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger.info(
                f"{endpoint_name} endpoint called",
                endpoint=endpoint_name
            )
            
            try:
                result = func(*args, **kwargs)
                return result
                
            except Exception as e:
                logger.error(
                    f"{endpoint_name} endpoint error",
                    endpoint=endpoint_name,
                    error_type=type(e).__name__,
                    error_message=str(e)
                )
                raise
        
        return wrapper
    return decorator


def log_api_call_details(data: dict, endpoint: str):
    """Log sanitized API call details for debugging.
    
    WARNING: Only use this for debugging. Never log full request payloads in production
    as they may contain sensitive data even after sanitization.
    """
    sanitized = sanitize_for_logging(data)
    
    logger.info(
        f"API call details for {endpoint}",
        endpoint=endpoint,
        request_data_preview=sanitized
    )


def log_scoring_metrics(
    num_labels: int,
    num_suggestions: int,
    scoring_mode: str,
    confidence_stats: dict
):
    """Log business metrics for scoring operations."""
    logger.info(
        f"Scoring completed",
        num_labels=num_labels,
        num_suggestions=num_suggestions,
        scoring_mode=scoring_mode,
        avg_confidence=confidence_stats.get('avg', 0),
        max_confidence=confidence_stats.get('max', 0),
        min_confidence=confidence_stats.get('min', 0)
    )
