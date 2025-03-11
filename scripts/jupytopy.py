#!/usr/bin/env python
import os, sys, json

import argparse
import pathlib

def generate_parser():
    program_name = 'jupytopy'
    parser = argparse.ArgumentParser(
        prog='jupytopy',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=f'{program_name} - Translate jupyter notebooks to python scripts for easy versioning',
        epilog=f'Examples:' '\n'
               f' * To save the script a.ipynb to the directory `scripts`, call' '\n'
               f'    jupytopy.py a.ipynb' '\n'
               f' * To save the all scripts of type foo in a dedicated folder, call' '\n'
               f'    jupytopy.py foo-* -d foo' '\n'
        ,
        add_help=True)
    parser.add_argument('n', metavar='SCRIPTNAME', help='scriptname', nargs='*', type=pathlib.Path, action='extend')
    parser.add_argument('-d', metavar='DIRNAME', help='Directory name', type=str)
    # parser.add_argument('-o', metavar='OUTPUTFILE', help='Generate graphs [optional:file prefix]', type=str, nargs='?', const='drop', default = None)
    # parser.add_argument('-v', help='Verbosity (-v: info, -vv: debug, -vvv: trace)', action="count", default=0)
    return parser

parser = generate_parser()

args = parser.parse_args()
notebook_paths = args.n
dirname = args.d

prefix = 'ipynb-'

### SELECTING WHERE WE WILL SAVE THE SCRIPTS
maindir = '.'
rootscriptsdir = os.path.join(maindir, 'scripts')
if not os.path.isdir(rootscriptsdir):
    os.mkdir(rootscriptsdir)

scriptsdir = rootscriptsdir
if dirname is not None:
    scriptsdir = os.path.join(rootscriptsdir, dirname)

    if not os.path.isdir(scriptsdir):
        os.mkdir(scriptsdir)

### SELECTING WHICH NOTEBOOKS WE WILL CONVERT TO SCRIPTS
ipynbs = []

if notebook_paths is not None: # case : a particular or several notebooks was specified
    for notebook_path in notebook_paths:
        if os.path.isfile(notebook_path) and notebook_path.name.endswith('.ipynb') and str(notebook_path.parent)== '.':
            ipynbs.append(str(notebook_path.name)[:-6])
        else:
            pass
            #print(f'ERROR - NOTEBOOK IS NOT CORRECT ? {notebook_path}')
else:
    ipynbs = [f[:-6] for f in os.listdir(maindir) if os.path.isfile(os.path.join(maindir, f)) and f.endswith('.ipynb')]
ipynbs.sort()

#print(f'-d: {scriptsdir}')
#print(f'n: {notebook_paths}')
#print(f'ipynbs: {ipynbs}')
# sys.exit(0) # debug exit

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
