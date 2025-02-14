#!/usr/bin/env python
import os, json

scriptsdir = 'scripts' # '.'

prefix = 'ipynb-'
ipynbs = [f[:-6] for f in os.listdir('.') if os.path.isfile(f) and f.endswith('.ipynb')]
ipynbs.sort()

for ipynb in ipynbs:
    infilename = os.path.join('.', ipynb + '.ipynb')
    outfilename = os.path.join(scriptsdir, prefix + ipynb + '.py')

    fin = open(infilename, 'r') #input.ipynb
    j = json.load(fin)
    fout = open(outfilename, 'w') #output.py
    fout.write("# -*- coding: utf-8 -*-\n")
    fout.write("# <nbformat>3.0</nbformat>")
    if j["nbformat"] >= 4:
        for i,cell in enumerate(j["cells"]):
            if cell["cell_type"] == "markdown":
                fout.write("\n# <markdowncell>\n\n")
            elif cell["cell_type"] == "code":
                fout.write("\n# <codecell>\n\n")
            # of.write("#cell "+str(i)+"\n")
            for line in cell["source"]:
                if cell["cell_type"] == "markdown":
                    fout.write("# ")
                fout.write(line)
            fout.write('\n\n')
    else:
        for i,cell in enumerate(j["worksheets"][0]["cells"]):
            fout.write("#cell "+str(i)+"\n")
            for line in cell["input"]:
                fout.write(line)
            fout.write('\n\n')

    fin.close()
    fout.close()