"""Do not run this file directly -- it is not the installer.

    python setup.py            <- does nothing ("error: no commands supplied")
    pip install -e .           <- editable install (what you probably want)
    pip install .              <- regular install

All packaging metadata lives in `pyproject.toml` (PEP 621). This shim exists
only so setuptools has a `setup.py` to fall back on in build environments that
predate PEP 660 editable installs; modern toolchains ignore it and read
`pyproject.toml` directly.

The package also works with no installation at all -- it is a plain directory,
so running from the project root (or putting it on PYTHONPATH) is enough.
"""

import sys

from setuptools import setup

if len(sys.argv) == 1:
    sys.exit(__doc__)

# --- guard against a build backend too old to read [project] ---------------
# setuptools < 61 silently ignores pyproject.toml's [project] table and builds
# a distribution called "UNKNOWN 0.0.0" -- installed, importable, and wrong.
# A loud failure is far better than a silent mis-install.
try:
    from setuptools import __version__ as _sv

    _major = int(_sv.split(".")[0])
except Exception:                                   # pragma: no cover
    _major = 61                                     # unknown: assume fine

if _major < 61:
    sys.exit(
        f"\nERROR: setuptools {_sv} is too old to read [project] from "
        "pyproject.toml.\n"
        "It would silently build this package as 'UNKNOWN 0.0.0'.\n\n"
        "Fix, in order of preference:\n"
        "  pip install -U 'setuptools>=61' pip\n"
        "  pip install --no-build-isolation .   "
        "(if your env already has a newer setuptools)\n"
        "  # or skip installation entirely -- hvi_emp is a plain directory,\n"
        "  # so running from the project root works without installing.\n"
    )

setup()
