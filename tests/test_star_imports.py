"""
Tests for the star imports of g2ltk.

A package gets the functions of its submodules with ``from .submodule import *``, which copies
the names listed in the ``__all__`` of the submodule (or, without ``__all__``, every name of the
submodule, including what it imports). Every module imported this way must therefore have an
``__all__`` listing exactly the public names it defines: its imports (numpy, typing, ...) do not
leak into the package, and a new function cannot be forgotten.

The modules are read, not imported, so these tests also run without the optional dependencies.
"""
import ast
import importlib.util
from pathlib import Path

import pytest

PACKAGE_DIR = Path(importlib.util.find_spec('g2ltk').origin).parent

# Public names deliberately left out of the __all__ of their module
NOT_EXPORTED = {
    # usable as g2ltk.customlog.<name>, but kept out of the top-level g2ltk namespace
    'g2ltk.customlog': {'G2LFormatter', 'default_verbose'},
}


def module_name(path: Path) -> str:
    parts = path.relative_to(PACKAGE_DIR.parent).with_suffix('').parts
    return '.'.join(parts[:-1] if parts[-1] == '__init__' else parts)


def module_tree(name: str) -> ast.Module:
    path = PACKAGE_DIR.parent.joinpath(*name.split('.'))
    path = path / '__init__.py' if path.is_dir() else path.with_suffix('.py')
    return ast.parse(path.read_text(encoding='utf-8'))


def find_star_imports():
    """(importing module, imported module) for every `from ... import *` of a g2ltk module in g2ltk."""
    found = []
    for path in sorted(PACKAGE_DIR.rglob('*.py')):
        importer = module_name(path)
        package = importer if path.name == '__init__.py' else importer.rpartition('.')[0]
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
            if not (isinstance(node, ast.ImportFrom) and any(alias.name == '*' for alias in node.names)):
                continue
            if node.level:
                base = package.split('.')[:len(package.split('.')) - node.level + 1]
                imported = '.'.join(base + ([node.module] if node.module else []))
            else:
                imported = node.module
            if imported.split('.')[0] == 'g2ltk':
                found.append((importer, imported))
    return found


def declared_all(tree: ast.Module):
    """The value of the __all__ of a module, or None if it has none."""
    for statement in tree.body:
        if isinstance(statement, ast.Assign) and any(isinstance(target, ast.Name) and target.id == '__all__'
                                                     for target in statement.targets):
            return ast.literal_eval(statement.value)
    return None


def defined_names(tree: ast.Module) -> set:
    """The names a module binds at its top level, other than by importing them."""
    names = set()

    def collect(statements):
        for statement in statements:
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(statement.name)
            elif isinstance(statement, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.For)):
                targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
                for target in targets:
                    names.update(node.id for node in ast.walk(target)
                                 if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store))
            if isinstance(statement, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
                for field in ('body', 'orelse', 'finalbody'):
                    collect(getattr(statement, field, []))
                for handler in getattr(statement, 'handlers', []):
                    collect(handler.body)

    collect(tree.body)
    return names


STAR_IMPORTS = find_star_imports()
STAR_IMPORTED_MODULES = sorted({imported for _, imported in STAR_IMPORTS})


def test_star_imports_are_found():
    # guards against the search silently finding nothing
    assert ('g2ltk.videoreading', 'g2ltk.videoreading.reading_gcv') in STAR_IMPORTS
    assert ('g2ltk.fourier', 'g2ltk.fourier.FFT1D') in STAR_IMPORTS


@pytest.mark.parametrize('module', STAR_IMPORTED_MODULES)
def test_star_imported_module_has_all(module):
    exported = declared_all(module_tree(module))
    importers = [importer for importer, imported in STAR_IMPORTS if imported == module]
    assert exported is not None, (f'{module} is star imported by {importers} but has no __all__, '
                                  f'so everything it imports is copied into them')
    duplicates = sorted({name for name in exported if exported.count(name) > 1})
    assert not duplicates, f'{module}.__all__ lists some names twice: {duplicates}'


@pytest.mark.parametrize('module', STAR_IMPORTED_MODULES)
def test_all_lists_only_names_defined_in_module(module):
    tree = module_tree(module)
    exported = declared_all(tree)
    if exported is None:
        pytest.skip(f'{module} has no __all__')
    not_defined = sorted(set(exported) - defined_names(tree))
    assert not not_defined, (f'{module}.__all__ lists names that {module} does not define: {not_defined}. '
                             f'If they are imported, import them explicitly where they are needed instead')


@pytest.mark.parametrize('module', STAR_IMPORTED_MODULES)
def test_all_lists_every_public_name_of_module(module):
    tree = module_tree(module)
    exported = declared_all(tree)
    if exported is None:
        pytest.skip(f'{module} has no __all__')
    public = {name for name in defined_names(tree) if not name.startswith('_')}
    missing = sorted(public - set(exported) - NOT_EXPORTED.get(module, set()))
    assert not missing, (f'{module} defines public names that are not in its __all__, so the packages that '
                         f'star import it do not get them: {missing}. Add them to __all__, or start their '
                         f'name with _ if they are private')
