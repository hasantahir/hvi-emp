"""Optional-plotting helper shared by the examples.

matplotlib is an *optional* dependency: it is needed to draw the figures,
never to compute the physics. Examples therefore obtain pyplot through
`get_pyplot()`, which returns either the real module or a do-nothing stand-in
-- so a missing matplotlib costs you the PNG and nothing else. All the printed
results, tables and diagnostics still appear.

The stand-in is a Null Object: every attribute access, call and subscript
returns another stand-in, and unpacking (`fig, ax = plt.subplots(...)`) yields
two of them. That lets the example code stay linear and readable instead of
being wrapped in `if plotting_available:` blocks, which is worth more than
avoiding a little metaprogramming in one helper file.
"""

from __future__ import annotations

from pathlib import Path

_WARNED = False


class _NoPlot:
    """Absorbs any plotting call and does nothing."""

    __slots__ = ()

    def __getattr__(self, _name):
        return _NoPlot()

    def __call__(self, *args, **kwargs):
        return _NoPlot()

    def __getitem__(self, _key):
        return _NoPlot()

    def __iter__(self):
        # supports `fig, ax = plt.subplots(...)` and `for a in axes`
        return iter((_NoPlot(), _NoPlot()))

    def __bool__(self):
        return False

    def __repr__(self):                      # pragma: no cover - cosmetic
        return "<plotting disabled: matplotlib not installed>"


def plotting_available() -> bool:
    """True if matplotlib can be imported."""
    try:
        import matplotlib  # noqa: F401
        return True
    except ModuleNotFoundError:
        return False


def get_pyplot(figure_name: str = "the figure"):
    """Headless `matplotlib.pyplot`, or a no-op stand-in if it is missing.

    Parameters
    ----------
    figure_name : str
        Named in the message so it is obvious which figure is skipped.
    """
    global _WARNED
    try:
        import matplotlib
    except ModuleNotFoundError:
        if not _WARNED:
            _WARNED = True
            print(
                "\n" + "-" * 74
                + f"\nmatplotlib is not installed, so {figure_name} will be "
                  "skipped.\n"
                  "Everything else below is unaffected: matplotlib is used "
                  "only for plotting,\nnever for the physics.\n\n"
                  "    pip install matplotlib\n"
                  '    pip install -e ".[plots]"     # same thing, via the '
                  "package extra\n"
                + "-" * 74 + "\n")
        return _NoPlot()

    matplotlib.use("Agg")        # headless: write files, never open a window
    import matplotlib.pyplot as plt
    return plt


def figures_dir() -> Path:
    """`figures/` at the project root, created if absent."""
    out = Path(__file__).resolve().parents[1] / "figures"
    out.mkdir(exist_ok=True)
    return out


__all__ = ["get_pyplot", "figures_dir", "plotting_available"]
