"""
Rate limiting utilities for API providers.

This module provides rate limiting functionality to prevent API
throttling and manage costs during experiments.

Example:
    >>> from synthetic_experiments.rate_limiting import (
    ...     RateLimiter,
    ...     TokenBucket,
    ...     RateLimitedProvider
    ... )
    >>> 
    >>> # Wrap provider with rate limiting
    >>> limiter = RateLimiter(requests_per_minute=60, tokens_per_minute=100000)
    >>> provider = RateLimitedProvider(base_provider, limiter)
"""

from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import time
import threading
import logging
from datetime import datetime, timedelta
from collections import deque

logger = logging.getLogger(__name__)


class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded and blocking is disabled."""
    pass


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    requests_per_minute: int = 60
    requests_per_day: Optional[int] = None
    tokens_per_minute: Optional[int] = None
    tokens_per_day: Optional[int] = None
    concurrent_requests: int = 1
    retry_on_limit: bool = True
    max_retries: int = 3
    backoff_factor: float = 2.0


class TokenBucket:
    """
    Token bucket rate limiter implementation.
    
    Allows bursting while maintaining average rate.
    """
    
    def __init__(
        self,
        capacity: int,
        refill_rate: float,
        refill_interval: float = 1.0
    ):
        """
        Args:
            capacity: Maximum tokens in bucket
            refill_rate: Tokens added per refill interval
            refill_interval: Seconds between refills
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.refill_interval = refill_interval
        
        self._tokens = float(capacity)
        self._last_refill = time.time()
        self._lock = threading.Lock()
    
    def _refill(self):
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self._last_refill
        intervals = elapsed / self.refill_interval
        
        tokens_to_add = intervals * self.refill_rate
        self._tokens = min(self.capacity, self._tokens + tokens_to_add)
        self._last_refill = now
    
    def acquire(self, tokens: int = 1, blocking: bool = True, timeout: float = None) -> bool:
        """
        Acquire tokens from the bucket.
        
        Args:
            tokens: Number of tokens to acquire
            blocking: Wait if not enough tokens
            timeout: Maximum wait time (None = forever)
            
        Returns:
            True if acquired, False if timeout/non-blocking
        """
        start_time = time.time()
        
        while True:
            with self._lock:
                self._refill()
                
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return True
            
            if not blocking:
                return False
            
            if timeout and (time.time() - start_time) >= timeout:
                return False
            
            # Calculate wait time for enough tokens
            needed = tokens - self._tokens
            wait_time = (needed / self.refill_rate) * self.refill_interval
            time.sleep(min(wait_time, 0.1))  # Don't wait too long between checks
    
    @property
    def available_tokens(self) -> float:
        """Get current available tokens."""
        with self._lock:
            self._refill()
            return self._tokens


class SlidingWindowLimiter:
    """
    Sliding window rate limiter.
    
    More accurate than fixed window for bursty traffic.
    """
    
    def __init__(self, limit: int, window_seconds: float = 60.0):
        """
        Args:
            limit: Maximum requests in window
            window_seconds: Window duration in seconds
        """
        self.limit = limit
        self.window_seconds = window_seconds
        self._timestamps: deque = deque()
        self._lock = threading.Lock()
    
    def _cleanup(self):
        """Remove expired timestamps."""
        cutoff = time.time() - self.window_seconds
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()
    
    def acquire(self, blocking: bool = True, timeout: float = None) -> bool:
        """
        Acquire permission for a request.
        
        Args:
            blocking: Wait if limit reached
            timeout: Maximum wait time
            
        Returns:
            True if acquired, False otherwise
        """
        start_time = time.time()
        
        while True:
            with self._lock:
                self._cleanup()
                
                if len(self._timestamps) < self.limit:
                    self._timestamps.append(time.time())
                    return True
            
            if not blocking:
                return False
            
            if timeout and (time.time() - start_time) >= timeout:
                return False
            
            # Wait until oldest request expires
            with self._lock:
                if self._timestamps:
                    wait_time = self._timestamps[0] + self.window_seconds - time.time()
                    if wait_time > 0:
                        time.sleep(min(wait_time, 0.1))
    
    @property
    def current_count(self) -> int:
        """Get current request count in window."""
        with self._lock:
            self._cleanup()
            return len(self._timestamps)
    
    @property
    def available(self) -> int:
        """Get available slots."""
        return max(0, self.limit - self.current_count)


class RateLimiter:
    """
    Composite rate limiter with multiple constraints.
    
    Combines requests per minute, tokens per minute, and daily limits.
    
    Example:
        >>> limiter = RateLimiter(
        ...     requests_per_minute=60,
        ...     tokens_per_minute=100000,
        ...     requests_per_day=10000
        ... )
        >>> 
        >>> # Before each API call
        >>> limiter.acquire(estimated_tokens=500)
        >>> response = api.call(...)
        >>> limiter.record_usage(actual_tokens=response.tokens)
    """
    
    def __init__(
        self,
        requests_per_minute: int = 60,
        tokens_per_minute: Optional[int] = None,
        requests_per_day: Optional[int] = None,
        tokens_per_day: Optional[int] = None,
        concurrent_requests: int = 1
    ):
        """
        Args:
            requests_per_minute: Max requests per minute
            tokens_per_minute: Max tokens per minute
            requests_per_day: Max requests per day
            tokens_per_day: Max tokens per day
            concurrent_requests: Max concurrent requests
        """
        self.config = RateLimitConfig(
            requests_per_minute=requests_per_minute,
            tokens_per_minute=tokens_per_minute,
            requests_per_day=requests_per_day,
            tokens_per_day=tokens_per_day,
            concurrent_requests=concurrent_requests
        )
        
        # Request limiters
        self._rpm_limiter = SlidingWindowLimiter(requests_per_minute, 60.0)
        self._rpd_limiter = SlidingWindowLimiter(requests_per_day, 86400.0) if requests_per_day else None
        
        # Token limiters (using token bucket)
        self._tpm_limiter = TokenBucket(
            capacity=tokens_per_minute,
            refill_rate=tokens_per_minute / 60.0,
            refill_interval=1.0
        ) if tokens_per_minute else None
        
        self._tpd_limiter = TokenBucket(
            capacity=tokens_per_day,
            refill_rate=tokens_per_day / 86400.0,
            refill_interval=1.0
        ) if tokens_per_day else None
        
        # Concurrency limiter
        self._semaphore = threading.Semaphore(concurrent_requests)
        
        # Usage tracking
        self._total_requests = 0
        self._total_tokens = 0
        self._start_time = time.time()
    
    def acquire(
        self,
        estimated_tokens: int = 0,
        blocking: bool = True,
        timeout: float = None
    ) -> bool:
        """
        Acquire permission for an API call.
        
        Args:
            estimated_tokens: Estimated tokens for this request
            blocking: Wait if limits reached
            timeout: Maximum wait time
            
        Returns:
            True if acquired, False otherwise
        """
        # Acquire concurrency slot
        if not self._semaphore.acquire(blocking=blocking, timeout=timeout):
            if not blocking:
                raise RateLimitExceeded("Concurrent request limit reached")
            return False
        
        try:
            # Check RPM
            if not self._rpm_limiter.acquire(blocking=blocking, timeout=timeout):
                self._semaphore.release()
                if not blocking:
                    raise RateLimitExceeded("Requests per minute limit reached")
                return False
            
            # Check RPD
            if self._rpd_limiter and not self._rpd_limiter.acquire(blocking=blocking, timeout=timeout):
                self._semaphore.release()
                if not blocking:
                    raise RateLimitExceeded("Requests per day limit reached")
                return False
            
            # Check TPM
            if self._tpm_limiter and estimated_tokens > 0:
                if not self._tpm_limiter.acquire(estimated_tokens, blocking=blocking, timeout=timeout):
                    self._semaphore.release()
                    if not blocking:
                        raise RateLimitExceeded("Tokens per minute limit reached")
                    return False
            
            # Check TPD
            if self._tpd_limiter and estimated_tokens > 0:
                if not self._tpd_limiter.acquire(estimated_tokens, blocking=blocking, timeout=timeout):
                    self._semaphore.release()
                    if not blocking:
                        raise RateLimitExceeded("Tokens per day limit reached")
                    return False
            
            self._total_requests += 1
            return True
            
        except Exception:
            self._semaphore.release()
            raise
    
    def release(self):
        """Release concurrency slot after request completes."""
        self._semaphore.release()
    
    def record_usage(self, tokens: int):
        """Record actual token usage after request completes."""
        self._total_tokens += tokens
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        elapsed = time.time() - self._start_time
        return {
            'total_requests': self._total_requests,
            'total_tokens': self._total_tokens,
            'elapsed_seconds': elapsed,
            'avg_requests_per_minute': (self._total_requests / elapsed) * 60 if elapsed > 0 else 0,
            'avg_tokens_per_minute': (self._total_tokens / elapsed) * 60 if elapsed > 0 else 0,
            'current_rpm': self._rpm_limiter.current_count,
            'rpm_available': self._rpm_limiter.available
        }


class RateLimitedProvider:
    """
    Wrapper that adds rate limiting to any provider.
    
    Example:
        >>> from synthetic_experiments.providers import OpenAIProvider
        >>> 
        >>> base_provider = OpenAIProvider(model_name="gpt-4")
        >>> limiter = RateLimiter(requests_per_minute=20, tokens_per_minute=40000)
        >>> provider = RateLimitedProvider(base_provider, limiter)
        >>> 
        >>> # Use like normal - rate limiting is automatic
        >>> response = provider.generate(messages)
    """
    
    def __init__(
        self,
        provider,
        rate_limiter: RateLimiter,
        estimate_tokens: Callable[[list], int] = None
    ):
        """
        Args:
            provider: Base provider instance
            rate_limiter: RateLimiter instance
            estimate_tokens: Function to estimate tokens from messages
        """
        self._provider = provider
        self._limiter = rate_limiter
        self._estimate_tokens = estimate_tokens or self._default_estimate
    
    def _default_estimate(self, messages: list) -> int:
        """Simple token estimation based on character count."""
        total_chars = sum(len(str(m.get('content', ''))) for m in messages)
        # Rough estimate: ~4 chars per token
        return int(total_chars / 4) + 100  # Add buffer for overhead
    
    @property
    def model_name(self):
        return self._provider.model_name
    
    def generate(self, messages: list, **kwargs):
        """
        Generate response with rate limiting.
        
        Args:
            messages: Conversation messages
            **kwargs: Additional provider arguments
            
        Returns:
            Provider response
        """
        estimated_tokens = self._estimate_tokens(messages)
        
        # Acquire rate limit permission
        self._limiter.acquire(estimated_tokens=estimated_tokens)
        
        try:
            response = self._provider.generate(messages, **kwargs)
            
            # Record actual usage if available
            if hasattr(response, 'usage'):
                actual_tokens = getattr(response.usage, 'total_tokens', estimated_tokens)
                self._limiter.record_usage(actual_tokens)
            else:
                self._limiter.record_usage(estimated_tokens)
            
            return response
            
        finally:
            self._limiter.release()
    
    def __getattr__(self, name):
        """Proxy other attributes to base provider."""
        return getattr(self._provider, name)


# Pre-configured limiters for common providers
def create_openai_limiter(tier: str = "tier1") -> RateLimiter:
    """
    Create rate limiter for OpenAI API.
    
    Args:
        tier: API tier ('free', 'tier1', 'tier2', etc.)
        
    Returns:
        Configured RateLimiter
    """
    configs = {
        'free': RateLimiter(
            requests_per_minute=3,
            tokens_per_minute=40000,
            requests_per_day=200
        ),
        'tier1': RateLimiter(
            requests_per_minute=60,
            tokens_per_minute=60000
        ),
        'tier2': RateLimiter(
            requests_per_minute=500,
            tokens_per_minute=80000
        ),
        'tier3': RateLimiter(
            requests_per_minute=5000,
            tokens_per_minute=160000
        ),
        'tier4': RateLimiter(
            requests_per_minute=10000,
            tokens_per_minute=800000
        ),
        'tier5': RateLimiter(
            requests_per_minute=10000,
            tokens_per_minute=10000000
        )
    }
    return configs.get(tier, configs['tier1'])


def create_claude_limiter(tier: str = "standard") -> RateLimiter:
    """
    Create rate limiter for Anthropic Claude API.
    
    Args:
        tier: API tier ('standard', 'scale')
        
    Returns:
        Configured RateLimiter
    """
    configs = {
        'standard': RateLimiter(
            requests_per_minute=60,
            tokens_per_minute=100000
        ),
        'scale': RateLimiter(
            requests_per_minute=1000,
            tokens_per_minute=400000
        )
    }
    return configs.get(tier, configs['standard'])


def create_ollama_limiter() -> RateLimiter:
    """
    Create rate limiter for Ollama (local, typically no limits needed).
    
    Returns:
        Permissive RateLimiter
    """
    return RateLimiter(
        requests_per_minute=1000,  # Essentially unlimited
        concurrent_requests=4  # Limit concurrency to not overload system
    )


class AdaptiveRateLimiter(RateLimiter):
    """
    Rate limiter that adapts based on API responses.
    
    Automatically backs off when rate limit errors are detected.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._backoff_until = 0
        self._consecutive_errors = 0
        self._backoff_factor = 2.0
        self._max_backoff = 300.0  # 5 minutes
    
    def report_error(self, error_type: str = "rate_limit"):
        """Report an error to trigger backoff."""
        if error_type == "rate_limit":
            self._consecutive_errors += 1
            backoff_time = min(
                self._backoff_factor ** self._consecutive_errors,
                self._max_backoff
            )
            self._backoff_until = time.time() + backoff_time
            logger.warning(f"Rate limit hit, backing off for {backoff_time:.1f}s")
    
    def report_success(self):
        """Report success to reset backoff."""
        self._consecutive_errors = 0
    
    def acquire(self, *args, **kwargs) -> bool:
        # Check backoff
        if time.time() < self._backoff_until:
            wait_time = self._backoff_until - time.time()
            if kwargs.get('blocking', True):
                logger.debug(f"Waiting for backoff: {wait_time:.1f}s")
                time.sleep(wait_time)
            else:
                raise RateLimitExceeded(f"In backoff period for {wait_time:.1f}s")
        
        return super().acquire(*args, **kwargs)
