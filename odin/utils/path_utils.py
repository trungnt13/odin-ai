from __future__ import print_function, division, absolute_import

import os
import sys
import shutil


def get_script_path():
    """Return the path of the script that calling this methods"""
    path = os.path.dirname(sys.argv[0])
    path = os.path.join('.', path)
    return path
