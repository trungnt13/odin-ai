# -*- coding: utf-8 -*-
# openSMILE LICENSE
# ===========
# Authors: Florian Eyben, Felix Weninger, Martin Woellmer, Bjoern Schuller
# Copyright (C) 2008-2013, Institute for Human-Machine Communication, TUM
# Copyright (C) 2013-2014, audEERING UG (haftungsbeschrÃ¤nkt)
# http://www.audeering.com/research-and-open-source/files/openSMILE-open-source-license.txt
from __future__ import print_function, division, absolute_import

import os
import re
from .base import Extractor

__all__ = [
    'SMILEpitch',
    'SMILEsad'
]


def verify_dependencies(args):
    """
    Make sure we can find external dependencies, the executables
    ffmpeg and openSMILE
    """
    if not os.path.exists(args.opensmile_home):
        raise Exception("Can't find openSMILE home {0}".format(args.opensmile_home))
    if not os.path.exists(args.opensmile_conf):
        raise Exception("Can't find openSMILE config {0}".format(args.opensmile_conf))

    try:
        command = "ffmpeg -version"
        output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError:
        raise Exception("Can't find ffmpeg executable")
    else:
        m = re.search('ffmpeg version (.*) Copyright', output, re.MULTILINE)
        if m:
            ffmpeg_version = m.group(1)
            print('ffmpeg version ', ffmpeg_version)
    try:
        command = "SMILExtract -h"
        output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError:
        raise Exception("Can't find SMILExtract executable")
    else:
        m = re.search('openSMILE version (.*)', output, re.MULTILINE)
        if m:
            opensmile_version = m.group(1)
            print('openSMILE version ', opensmile_version)


class SMILEpitch(Extractor):
    pass


class SMILEsad(object):
    """docstring for SMILEsad"""

    def __init__(self, arg):
        super(SMILEsad, self).__init__()
        self.arg = arg
