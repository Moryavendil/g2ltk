#!/usr/bin/env python
"""Strip the outputs of Jupyter notebooks, using nbstripout with the options of this repository.

Usage:
    cleanipynb.py < in.ipynb > out.ipynb    # filter mode (this is how git calls it)
    cleanipynb.py a.ipynb b.ipynb           # clean the given files in place
Any other nbstripout option works too (e.g. `cleanipynb.py --dry-run a.ipynb`).

Setting up the git clean filter -- to do once in each new clone of the repository:
    pip install nbstripout
    git config filter.cleanipynb.clean "python scripts/cleanipynb.py"
    git config filter.cleanipynb.smudge cat
    git config filter.cleanipynb.required true
(a required filter also needs a smudge command, used when git writes a notebook to disk:
without it, checking out or merging a notebook fails with "smudge filter cleanipynb failed")
`python` must be the environment where nbstripout is installed; if git is run from somewhere
that environment is not activated (e.g. an IDE), give the full path of its python instead.
.gitattributes, which is versioned, already sends *.ipynb through the `cleanipynb` filter.
Git then only stores cleaned notebooks, while the files on disk keep their outputs: running
a notebook again does not show up in `git status`, only changes to its cells do.
"""
import sys

from nbstripout import main

# on top of nbstripout's defaults (outputs, execution counts, ExecuteTime, ...)
NBSTRIPOUT_OPTIONS = [
    '--keep-id',  # keep the cell ids, otherwise they are renumbered and inserting a cell changes all the following ones
    '--extra-keys', 'metadata.language_info.version',  # Python version of whoever ran the notebook last
]

if __name__ == '__main__':
    sys.argv[1:1] = NBSTRIPOUT_OPTIONS
    main()
