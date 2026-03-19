#!/usr/bin/env python
import os, sys
from nbformat import v3, v4

import argparse
import pathlib

prefix = 'ipynb-'
maindir = '.'
rootscriptsdir = os.path.join(maindir, 'scripts')
def generate_parser():
    program_name = 'pytojupy'
    parser = argparse.ArgumentParser(
        prog='pytojupy',
        description=f'{program_name} - python scriptsa to jupyter notebooks',
        epilog=f'', add_help=True)
    parser.add_argument('n', metavar='PYNAMES', help=f'Scripts (.py) to convert (ex: `scripts/{prefix}foo.py`)', nargs='*', type=pathlib.Path, action='extend')
    # parser.add_argument('-o', metavar='OUTPUTFILE', help='Generate graphs [optional:file prefix]', type=str, nargs='?', const='drop', default = None)
    # parser.add_argument('-v', help='Verbosity (-v: info, -vv: logger.debug, -vvv: trace)', action="count", default=0)
    return parser


parser = generate_parser()

args = parser.parse_args()
scripts_paths = args.n

print(f'PyToJupy: Converting .py -> .ipynb')
if len(scripts_paths) == 0:
    print(f'No script specified. Aborting.')
    sys.exit(-11)

targets = {}
for script_path in scripts_paths:
    if os.path.isfile(script_path) and script_path.name.endswith('.py'):
        targets[str(script_path.name)[len(prefix):-3]] = script_path
    else:
        pass
        #print(f'ERROR - NOTEBOOK IS NOT CORRECT ? {notebook_path}')

# print(f'Scripts directory is: {scriptsdir}')
print(f'Scripts to convert are: {[target for target in targets]}')

for target in targets:
    infilename = targets[target]

    outfilename = os.path.join(maindir, target + '.ipynb')

    try:
        with open(infilename) as fin:
            text = fin.read()
    except:
        print(f'ERROR - CANNOT OPEN?: {infilename}')
        sys.exit(-201)

    # restore the magic matplotlib command
    text = text.replace('# %matplotlib', '%matplotlib')

    # text += """
    # # <markdowncell>
    #
    # # If you can read this, reads_py() is no longer broken!
    # """

    nbook = v3.reads_py(text) # we read in v3 because the formatting is more sympathic
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
