"""Normalized provider errors. SDK exceptions never escape an adapter."""


class LLMError(Exception):
    pass


class RetryableError(LLMError):
    """429, 5xx, timeouts, connection errors."""


class AuthError(LLMError):
    """Invalid or missing credentials, or no permission for the model."""


class ContextLengthError(LLMError):
    """The request exceeded the model's context window."""


class ProviderError(LLMError):
    """Any other provider failure."""
