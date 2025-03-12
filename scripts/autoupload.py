#!/usr/bin/env python

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


import subprocess
import sys

cmd_getlatesttag = "git tag --sort=committerdate | grep -E '[0-9]' | tail -1 | cut -b 2-7"
cmd_rmcurrentags = "git tag -l | xargs git tag -d"
cmd_getgitstatus_machinereadable = "git status --porcelain"
cmd_getgitstatus_humanreadable = "git status"

gitversion = subprocess.check_output(cmd_getlatesttag, shell=True, text=True)[:-1]

print(bcolors.HEADER + f"Current tagged version: '{gitversion or '[None]'}'" + bcolors.ENDC)


# This is a dirty hack to avoid writing the proper "from .. import g2ltk" which ncessitates being in a package
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import g2ltk

toolsversion = g2ltk.__version__
print(bcolors.HEADER + f"Current g2ltk version: '{toolsversion}'" + bcolors.ENDC)

gitstatus = subprocess.check_output(cmd_getgitstatus_machinereadable, shell=True, text=True)[:-1]
if gitstatus != "":
    print(bcolors.FAIL + f"Cannot autotag: git status is unclean" + bcolors.ENDC)
    subprocess.run(cmd_getgitstatus_humanreadable, shell=True)
    sys.exit(100)


subprocess.run(f"git pull", shell=True)

if toolsversion == gitversion:
    print(bcolors.FAIL + "Cannot autotag: Current version is git's last version" + bcolors.ENDC)
    sys.exit(200)

print(bcolors.HEADER + 'Tagging the current version.' + bcolors.ENDC)

versiontext = f"v{toolsversion}"

subprocess.run(f"git tag {versiontext}", shell=True)

subprocess.run(f"git push", shell=True)

subprocess.run(f"git push origin tag {versiontext}", shell=True)

print(bcolors.HEADER + 'Incrementing __version__.' + bcolors.ENDC)

toolsversion_new = toolsversion

toolsversion_compo = toolsversion.split('.')
if len(toolsversion_compo) > 3 and 'dev' in toolsversion_compo[3]:
    try:
        devnbr = int(toolsversion_compo[3].split('dev')[1])
        toolsversion_compo[3] = 'dev' + str(devnbr + 1)
    except:
        print(bcolors.OKCYAN + f'Version number is {toolsversion} = {toolsversion_compo}' + bcolors.ENDC)
        print(bcolors.OKCYAN + f"And {toolsversion_compo[3]} = {toolsversion_compo[3].split('dev')}" + bcolors.ENDC)
        print(bcolors.OKCYAN + f"But {toolsversion_compo[3].split('dev')[1]} does not seem to be a number ?" + bcolors.ENDC)
elif len(toolsversion_compo) > 2:
    try:
        subvnbr = int(toolsversion_compo[2])
        toolsversion_compo[2] = str(subvnbr + 1)
    except:
        print(bcolors.OKCYAN + f'Version number is {toolsversion} = {toolsversion_compo}' + bcolors.ENDC)
        print(bcolors.OKCYAN + f'But {toolsversion_compo[2]} does not seem to be a number ?' + bcolors.ENDC)

toolsversion_new = '.'.join(toolsversion_compo)
if toolsversion_new == toolsversion:
    print(bcolors.OKGREEN + 'COULD NOT UPDATE VERSION NUMBER' + bcolors.ENDC)
    sys.exit(300)

with open('g2ltk/__init__.py', 'r') as f:
    txt = f.read()

txt = txt.replace(f"__version__ = '{toolsversion}'", f"__version__ = '{toolsversion_new}'")

with open('g2ltk/__init__.py', 'w') as f:
    f.write(txt)


subprocess.run(f"git add g2ltk/__init__.py", shell=True)
subprocess.run(f'git commit -m "Increment version number to {toolsversion_new}"', shell=True)
subprocess.run(f"git push", shell=True)
