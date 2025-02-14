#!/usr/bin/env python
import os
from nbformat import v3, v4

scriptsdir = 'scripts' # '.'

prefix = 'ipynb-'
ipynbs = [f[len(prefix):-3] for f in os.listdir(scriptsdir) if os.path.isfile(f) and f.endswith('.py') and f.startswith(prefix)]
ipynbs.sort()

for ipynb in ipynbs:
    infilename = os.path.join(scriptsdir, prefix + ipynb + '.py')
    outfilename = os.path.join('.', ipynb + '.ipynb')

    with open(infilename) as fin:
        text = fin.read()

    text += """
    # <markdowncell>
    
    # If you can read this, reads_py() is no longer broken! 
    """

    nbook = v3.reads_py(text)
    nbook = v4.upgrade(nbook)  # Upgrade v3 to v4

    # nbook.metadata.authors = [
    #     {
    #         "name": "G2L",
    #     },
    # ]

    nbook.metadata.kernelspec = {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        }

    nbook.metadata.language_info = {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        }

    nbook.metadata.file_extension = ".py"
    nbook.metadata.mimetype = "text/x-python"
    nbook.metadata.name = "python"
    nbook.metadata.nbconvert_exporter = "python"

    jsonform = v4.writes(nbook) + "\n"
    with open(outfilename, "w") as fout:
        fout.write(jsonform)