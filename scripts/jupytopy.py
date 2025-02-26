#!/usr/bin/env python
import os, sys, json

import argparse

def generate_parser():
    program_name = 'jupytopy'
    parser = argparse.ArgumentParser(
        prog='jupytopy',
        description=f'{program_name} - jupyter notebooks to python scripts',
        epilog=f'', add_help=True)
    # parser.add_argument('-n', metavar='SCRIPTNAME', help='scriptname', type=argparse.FileType('r'))
    parser.add_argument('-d', metavar='DIRNAME', help='Directory name', type=str)
    # parser.add_argument('-p', metavar='PXL_DENSITY', help='Pixel density (mm/px)', type=float)
    # parser.add_argument('-g', metavar='GRAVITY', help='Acceleration of gravity (typically 9.81)', type=float)
    # parser.add_argument('-d', metavar='DELTARHO', help='Density contrast, in kg/L (typically 1.00 for water/air)', type=float)
    # parser.add_argument('-o', metavar='OUTPUTFILE', help='Generate graphs [optional:file prefix]', type=str, nargs='?', const='drop', default = None)
    # parser.add_argument('-v', help='Verbosity (-v: info, -vv: logger.debug, -vvv: trace)', action="count", default=0)
    return parser

parser = generate_parser()

args = parser.parse_args()
dirname = args.d


maindir = '.'
rootscriptsdir = os.path.join(maindir, 'scripts')
if not os.path.isdir(rootscriptsdir):
    os.mkdir(rootscriptsdir)

scriptsdir = rootscriptsdir
if dirname is not None:
    scriptsdir = os.path.join(rootscriptsdir, dirname)

    if not os.path.isdir(scriptsdir):
        os.mkdir(scriptsdir)


prefix = 'ipynb-'
ipynbs = [f[:-6] for f in os.listdir(maindir) if os.path.isfile(os.path.join(maindir, f)) and f.endswith('.ipynb')]
ipynbs.sort()

for ipynb in ipynbs:
    infilename = os.path.join(maindir, ipynb + '.ipynb')
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
                # hide the %matplotlib magic command so that the script can be runnable by itself
                line = line.replace('%matplotlib ', '# %matplotlib ')
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