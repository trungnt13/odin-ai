from __future__ import print_function, division, absolute_import

import os
import pickle

import numpy as np

from odin.utils import Progbar, get_exppath
from odin import fuel as F

sre_file_list = F.load_sre_list()
print(sre_file_list)

# ===========================================================================
# Exp path
# ===========================================================================
EXP_DIR = get_exppath('sre', override=False)

# ===========================================================================
# FILE LIST PATH
# ===========================================================================
# path to directory contain following folder:
#  * voxceleb
#  * voxceleb2
#  * NIST1996_2008
#  * Switchboard
#  * fisher
PATH_RAW_DATA = '/mnt/sdb1'
# ====== check every files are exist ====== #
def validating_all_data():
  prog = Progbar(target=(len(sre_file_list['voxceleb']) +
                         len(sre_file_list['sre']) -
                         2),
                 print_summary=True, print_report=True,
                 name="Validating File List")
  for row in sre_file_list['voxceleb'][1:]:
    path = row[0]
    path = os.path.join(PATH_RAW_DATA, path)
    assert os.path.exists(path), path
    prog.add(1)
  for row in sre_file_list['sre'][1:]:
    path = row[0]
    path = os.path.join(PATH_RAW_DATA, path)
    assert os.path.exists(path), path
    prog.add(1)
# ===========================================================================
# DATA PATH
# ===========================================================================

# ===========================================================================
# PATH HELPER
# ===========================================================================

# ===========================================================================
# DATA HELPERI
# ===========================================================================
