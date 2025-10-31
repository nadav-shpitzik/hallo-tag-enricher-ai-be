import logging
import json
import time
import uuid
from typing import Dict, Any, Optional, Callable
from functools import wraps
from contextlib import contextmanager
from threading import local

_request_context = local()


class StructuredLogger:
    """Enhanced logger with structured logging support."""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        
    def _build_message(self, message: str, extra: Optional[Dict[str, Any]] = None) -> str:
        """Build structured log message with context."""
        data = {
            'message': message,
            'timestamp': time.time()
        }
        
        if hasattr(_request_context, 'request_id'):
            data['request_id'] = _request_context.request_id
        
        if hasattr(_request_context, 'scoring_mode'):
            data['scoring_mode'] = _request_context.scoring_mode
            
        if extra:
            data.update(extra)
        
        return json.dumps(data)
    
    def info(self, message: str, **extra):
        """Log info with structured data."""
        self.logger.info(self._build_message(message, extra))
    
    def warning(self, message: str, **extra):
        """Log warning with structured data."""
        self.logger.warning(self._build_message(message, extra))
    
    def error(self, message: str, **extra):
        """Log error with structured data."""
        self.logger.error(self._build_message(message, extra))
    
    def debug(self, message: str, **extra):
        """Log debug with structured data."""
        self.logger.debug(self._build_message(message, extra))


def set_request_context(request_id: str, **kwargs):
    """Set context for the current request."""
    _request_context.request_id = request_id
    for key, value in kwargs.items():
        setattr(_request_context, key, value)


def clear_request_context():
    """Clear request context."""
    if hasattr(_request_context, 'request_id'):
        delattr(_request_context, 'request_id')
    if hasattr(_request_context, 'scoring_mode'):
        delattr(_request_context, 'scoring_mode')


def get_request_id() -> Optional[str]:
    """Get current request_id from context."""
    return getattr(_request_context, 'request_id', None)


def track_performance(operation_name: str):
    """Decorator to track performance of functions."""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            logger = StructuredLogger(func.__module__)
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                logger.info(
                    f"{operation_name} completed",
                    operation=operation_name,
                    duration_ms=round(duration * 1000, 2),
                    status='success'
                )
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                
                logger.error(
                    f"{operation_name} failed",
                    operation=operation_name,
                    duration_ms=round(duration * 1000, 2),
                    status='error',
                    error_type=type(e).__name__,
                    error_message=str(e)
                )
                raise
        
        return wrapper
    return decorator


@contextmanager
def track_operation(operation_name: str, logger: StructuredLogger, **extra):
    """Context manager to track operation performance."""
    start_time = time.time()
    
    logger.info(
        f"{operation_name} started",
        operation=operation_name,
        **extra
    )
    
    try:
        yield
        duration = time.time() - start_time
        
        logger.info(
            f"{operation_name} completed",
            operation=operation_name,
            duration_ms=round(duration * 1000, 2),
            status='success',
            **extra
        )
        
    except Exception as e:
        duration = time.time() - start_time
        
        logger.error(
            f"{operation_name} failed",
            operation=operation_name,
            duration_ms=round(duration * 1000, 2),
            status='error',
            error_type=type(e).__name__,
            error_message=str(e),
            **extra
        )
        raise


def log_llm_call(provider: str, model: str, tokens: Optional[int] = None, cost: Optional[float] = None):
    """Log LLM API call with cost tracking."""
    logger = StructuredLogger('llm_tracking')
    
    logger.info(
        f"LLM API call to {provider}",
        provider=provider,
        model=model,
        tokens=tokens,
        estimated_cost_usd=cost
    )


def sanitize_for_logging(data: Any, max_length: int = 200) -> Any:
    """Sanitize data for safe logging (truncate long strings, remove sensitive info)."""
    # List of sensitive field names to redact
    sensitive_fields = {
        'api_key', 'apikey', 'password', 'secret', 'token', 
        'authorization', 'auth', 'credential', 'openai_api_key'
    }
    
    if isinstance(data, str):
        if len(data) > max_length:
            return data[:max_length] + f"... (truncated, {len(data)} chars total)"
        return data
    elif isinstance(data, dict):
        sanitized = {}
        for k, v in data.items():
            # Redact sensitive fields
            if any(sensitive in k.lower() for sensitive in sensitive_fields):
                sanitized[k] = "[REDACTED]"
            else:
                sanitized[k] = sanitize_for_logging(v, max_length)
        return sanitized
    elif isinstance(data, list):
        if len(data) > 10:
            return [sanitize_for_logging(item, max_length) for item in data[:10]] + [f"... ({len(data) - 10} more items)"]
        return [sanitize_for_logging(item, max_length) for item in data]
    return data
