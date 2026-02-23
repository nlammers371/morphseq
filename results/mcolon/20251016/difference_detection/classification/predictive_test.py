"""
Compatibility wrapper for predictive_signal_test within difference detection.

Re-exports the core implementation from the primary classification package so
that downstream imports continue to function without maintaining duplicate code.
"""

from ...classification.predictive_test import predictive_signal_test

__all__ = ['predictive_signal_test']

