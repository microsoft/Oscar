#!/usr/bin/python

from __future__ import print_function
import os
import sys
import re
import os.path as op
from setuptools import find_packages, setup

# change directory to this module path
try:
    this_file = __file__
except NameError:
    this_file = sys.argv[0]
this_file = os.path.abspath(this_file)
if op.dirname(this_file):
    os.chdir(op.dirname(this_file))
script_dir = os.getcwd()

def readme(fname):
    """Read text out of a file in the same directory as setup.py.
    """
    return open(op.join(script_dir, fname)).read()


def find_version(fname):
    version_file = readme(fname)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


setup(
    name="oscar",
    version=find_version("oscar/__init__.py"),
    url='https://github.com/xjli/Oscar',
    description="Oscar for vision and language tasks",
    long_description=readme('README.md'),
    packages=find_packages(),
    classifiers=[
        'Intended Audience :: Developers',
        "Programming Language :: Python",
        'Topic :: Software Development',
    ]
)
