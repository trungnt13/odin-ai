from __future__ import print_function, division, absolute_import
import os
os.environ['ODIN'] = 'float32,gpu'

import numpy as np

from odin import ml
from odin import fuel as F
from odin.utils import args_parse, ctext, stdio

from const import PATH_ACOUSTIC_FEAT, TRAIN_DATA, PATH_EXP
from utils import get_model_path, prepare_ivec_data

# ===========================================================================
# Configs
# ===========================================================================
args = args_parse([
    ('-nmix', "Number of GMM mixture", None, 2048),
    ('-tdim', "Dimension of t-matrix", None, 600),
    ('-feat', "Acoustic feature", ('mspec', 'mfcc'), 'mfcc'),
    ('--gmm', "Force re-run training GMM", None, False),
    ('--stat', "Force re-extraction of centered statistics", None, False),
    ('--tmat', "Force re-run training Tmatrix", None, False),
    ('--ivec', "Force re-run extraction of i-vector", None, False),
    ('--all', "Run all the system again, just a shortcut", None, False),
])
args.gmm |= args.all
args.stat |= args.all | args.gmm
args.tmat |= args.all | args.stat
args.ivec |= args.all | args.tmat
FEAT = args.feat
MODEL_PATH, LOG_PATH, TEST_PATH, NAME_PATH = get_model_path('ivec', args)
stdio(LOG_PATH)
# ===========================================================================
# Load dataset
# ===========================================================================
X, sad, train, test = prepare_ivec_data(FEAT)
# ===========================================================================
# Training I-vector model
# ===========================================================================
ivec = ml.Ivector(path=MODEL_PATH, nmix=args.nmix, tv_dim=args.tdim,
                  niter_gmm=16, niter_tmat=16,
                  downsample=2, stochastic_downsample=True,
                  device='gpu', name="VoxCelebIvec")
if not ivec.is_fitted:
  ivec.fit(X, sad=sad, indices=train,
           extract_ivec=True, keep_stats=False)
# ====== extract train i-vector ====== #
I_train = F.MmapData(ivec.ivec_path, read_only=True)
name_train = np.genfromtxt(ivec.name_path, dtype=str)
print("Train i-vectors:", ctext(I_train, 'cyan'))
# ====== extract test i-vector ====== #
test = sorted(test.items(), key=lambda x: x[0])
I_test = ivec.transform(X, sad=sad, indices=test,
                        save_ivecs=False, keep_stats=False)
with open(NAME_PATH, 'w') as f_name, open(TEST_PATH, 'w') as f_dat:
  for (name, (start, end)), z in zip(test, I_test):
    f_dat.write(' '.join([str(i) for i in z]) + '\n')
    f_name.write(name + '\n')
# ====== print the model ====== #
print(ivec)
