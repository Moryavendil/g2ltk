"""
Checks for the optional dependencies of the subpackages of g2ltk (the extras of pyproject.toml).
"""
from importlib.util import find_spec
from typing import Dict


def require_extra(subpackage: str, extra: str, dependencies: Dict[str, str]) -> None:
    """
    Raises an error naming the extra to install if some optional dependencies of a subpackage are missing.

    Without it, importing the subpackage fails on the first missing module (No module named 'cv2'), which does not say
    how to install it.

    Parameters
    ----------
    subpackage : str
        name of the subpackage, e.g. 'g2ltk.videoreading'
    extra : str
        extra of g2ltk that installs the dependencies, e.g. 'video'
    dependencies : dict
        modules the subpackage imports when it is imported, as {module name: name of its package on PyPI}

    Raises
    ------
    ModuleNotFoundError
        if some of the modules are not installed
    """
    missing = [package for module, package in dependencies.items() if find_spec(module) is None]
    if missing:
        raise ModuleNotFoundError(f'{subpackage} needs {", ".join(missing)}, which g2ltk does not install by default. '
                                  f'Install the optional dependencies of {subpackage} with:  pip install "g2ltk[{extra}]"')
