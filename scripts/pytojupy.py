#!/usr/bin/env python
import os, sys
from nbformat import v3, v4

import argparse
import pathlib

def generate_parser():
    program_name = 'pytojupy'
    parser = argparse.ArgumentParser(
        prog='pytojupy',
        description=f'{program_name} - python scriptsa to jupyter notebooks',
        epilog=f'', add_help=True)
    # parser.add_argument('-n', metavar='SCRIPTNAME', help='scriptname', type=argparse.FileType('r'))
    parser.add_argument('-n', metavar='SCRIPTPATH', help='script or scipt dir path', type=pathlib.Path)
    # parser.add_argument('-p', metavar='PXL_DENSITY', help='Pixel density (mm/px)', type=float)
    # parser.add_argument('-g', metavar='GRAVITY', help='Acceleration of gravity (typically 9.81)', type=float)
    # parser.add_argument('-d', metavar='DELTARHO', help='Density contrast, in kg/L (typically 1.00 for water/air)', type=float)
    # parser.add_argument('-o', metavar='OUTPUTFILE', help='Generate graphs [optional:file prefix]', type=str, nargs='?', const='drop', default = None)
    # parser.add_argument('-v', help='Verbosity (-v: info, -vv: logger.debug, -vvv: trace)', action="count", default=0)
    return parser


parser = generate_parser()

args = parser.parse_args()
scriptpath = args.n

prefix = 'ipynb-'
maindir = '.'
scriptsdir = os.path.join(maindir, 'scripts')

ipynb_names = []
if scriptpath is None:
    print('No file selected, converting default scripts')
    ipynb_names = [f[len(prefix):-3] for f in os.listdir(scriptsdir) if os.path.isfile(os.path.join(scriptsdir, f)) and f.endswith('.py') and f.startswith(prefix)]
else:
    if os.path.isdir(scriptpath):
        # print(f'Directory selected: {scriptpath}')
        scriptsdir = scriptpath
        ipynb_names = [f[len(prefix):-3] for f in os.listdir(scriptsdir) if os.path.isfile(os.path.join(scriptsdir, f)) and f.endswith('.py') and f.startswith(prefix)]
    elif os.path.isfile(scriptpath):
        scriptsdir = scriptpath.parent
        scriptname = scriptpath.name
        if not (scriptname.endswith('.py') & scriptname.startswith(prefix)):
            print(f'Invalid script name: {scriptname} (path: {scriptpath})')
            sys.exit(-101)
        # print(f'File selected: {scriptpath}')
        ipynb_names = [scriptname[len(prefix):-3]]
    else:
        print(f'ERROR - ISWHAT?: {scriptpath}')
        sys.exit(-100)

ipynb_names.sort()

print(f'Scripts directory is: {scriptsdir}')
print(f'Scripts to convert are: {ipynb_names}')


for ipynb_name in ipynb_names:
    infilename = os.path.join(scriptsdir, prefix + ipynb_name + '.py')

    outfilename = os.path.join(maindir, ipynb_name + '.ipynb')

    try:
        with open(infilename) as fin:
            text = fin.read()
    except:
        print(f'ERROR - CANNOT OPEN?: {infilename}')
        sys.exit(-201)

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

    try:
        with open(outfilename, "w") as fout:
            fout.write(jsonform)
    except:
        print(f'ERROR - CANNOT OPEN?: {outfilename}')
        sys.exit(-202)