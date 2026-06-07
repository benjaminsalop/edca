from __future__ import annotations


class EDCAError(Exception):
    """Base exception for the rewritten EDCA core."""


class DomainError(EDCAError):
    """Raised when source data cannot be normalized into the domain model."""


class ValidationError(EDCAError):
    """Raised when normalized data fails integrity validation."""


class RepositoryError(EDCAError):
    """Raised for repository loading, lookup, or cross-reference issues."""


class CompatibilityError(EDCAError):
    """Raised when an assembly violates a hard compatibility rule."""


class LoadResolutionError(EDCAError):
    """Raised when load values or combinations cannot be resolved."""


class AssemblyEvaluationError(EDCAError):
    """Raised when a candidate assembly cannot be evaluated."""


class UnsupportedConfigurationError(EDCAError):
    """Raised when the user requests a configuration outside current scope."""
