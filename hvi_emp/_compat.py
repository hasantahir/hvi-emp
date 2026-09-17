"""Compatibility shims for the NumPy versions this package claims to support.

``pyproject.toml`` declares ``numpy>=1.24``.  That declaration has to be true,
which means not reaching for APIs that only exist in NumPy 2.x without a
fallback.

The one that caught us: **NumPy 2.0 renamed ``np.trapz`` to ``np.trapezoid``
and removed the old name.**  Using ``np.trapezoid`` unguarded made the package
silently require NumPy >= 2.0, and on 1.24-1.26 the whole chain died with

    AttributeError: module 'numpy' has no attribute 'trapezoid'

which is a confusing way to learn about a dependency floor.  Resolving the
name once here, at import, costs nothing per call and keeps both versions
working.
"""

from __future__ import annotations

import numpy as np

__all__ = ["trapezoid", "NUMPY_MAJOR"]

NUMPY_MAJOR = int(np.__version__.split(".")[0])

if hasattr(np, "trapezoid"):          # NumPy >= 2.0
    trapezoid = np.trapezoid
elif hasattr(np, "trapz"):            # NumPy < 2.0
    trapezoid = np.trapz
else:                                 # pragma: no cover
    raise ImportError(
        f"numpy {np.__version__} provides neither trapezoid nor trapz.\n"
        "This package needs one of them for the shell integrals in "
        "impact.py and the spectral energy in emp.py.\n"
        "Install a supported NumPy:  pip install -U 'numpy>=1.24'")
