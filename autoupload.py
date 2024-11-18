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

gitstatus = subprocess.check_output(cmd_getgitstatus_machinereadable, shell=True, text=True)[:-1]
if gitstatus != "":
    print(bcolors.FAIL + f"Cannot autotag: git status is unclean" + bcolors.ENDC)
    subprocess.run(cmd_getgitstatus_humanreadable, shell=True)
    sys.exit(100)


subprocess.run(f"git pull", shell=True)

gitversion = subprocess.check_output(cmd_getlatesttag, shell=True, text=True)[:-1]

print(bcolors.HEADER + f"Current tagged version: '{gitversion or '[None]'}'" + bcolors.ENDC)

import tools

toolsversion = tools.__version__
print(bcolors.HEADER + f"Current tools version: '{toolsversion}'" + bcolors.ENDC)

if toolsversion == gitversion:
    print(bcolors.WARNING + 'BOTH HAVE SAME VERSION !!' + bcolors.ENDC)
    print(bcolors.WARNING + 'ABORTING !!!' +bcolors.ENDC)
    sys.exit(200)

print(bcolors.HEADER + 'Tagging the current version.' + bcolors.ENDC)

versiontext = f"v{toolsversion}"

subprocess.run(f"git tag {versiontext}", shell=True)

subprocess.run(f"git push", shell=True)

subprocess.run(f"git push origin tag {versiontext}", shell=True)

print(bcolors.HEADER + 'Incrementing __version__.' + bcolors.ENDC)

toolsversion_compo = toolsversion.split('.')
if len(toolsversion_compo) > 2:
    toolsversion_compo[2] = str(int(toolsversion_compo[2]) + 1)
toolsversion_new = '.'.join(toolsversion_compo)

with open('tools/__init__.py', 'r') as f:
    txt = f.read()

txt = txt.replace(f"__version__ = '{toolsversion}'", f"__version__ = '{toolsversion_new}'")

with open('tools/__init__.py', 'w') as f:
    f.write(txt)
