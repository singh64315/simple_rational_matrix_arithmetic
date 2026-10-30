"""Rational matrix arithmetic with exact Fraction arithmetic.

This package exposes Matrix and a module-level eye helper.  All matrix
entries are stored as :class:`fractions.Fraction`, so addition,
multiplication, determinants, and inverses are exact -- no floating point
rounding, ever.
"""

from .core import Matrix, eye

__all__ = ["Matrix", "eye"]
