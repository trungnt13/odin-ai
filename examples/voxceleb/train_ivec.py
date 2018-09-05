from __future__ import print_function, division, absolute_import
import os
os.environ['ODIN'] = 'float32,gpu'

import numpy as np

from odin import ml
from odin import fuel as F
from odin.utils import args_parse, ctext

from const import PATH_ACOUSTIC_FEAT, TRAIN_DATA, PATH_EXP

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
NAME = '_'.join(['ivec', args.feat, str(args.nmix), str(args.tdim)])
SAVE_PATH = os.path.join(PATH_EXP, NAME)
print('Model name:', ctext(NAME, 'cyan'))
print("Save path:", ctext(SAVE_PATH, 'cyan'))
# ===========================================================================
# Load dataset
# ===========================================================================
ds = F.Dataset(PATH_ACOUSTIC_FEAT, read_only=True)
print(ds)
X = ds[FEAT]
train_indices = {name: ds['indices'][name]
                 for name in TRAIN_DATA.keys()}
test_indices = {name: start_end
                for name, start_end in ds['indices'].items()
                if name not in TRAIN_DATA}
print("#Train files:", ctext(len(train_indices), 'cyan'))
print("#Test files:", ctext(len(test_indices), 'cyan'))
# ===========================================================================
# Training I-vector model
# ===========================================================================
ivec = ml.Ivector(path=SAVE_PATH, nmix=args.nmix, tv_dim=args.tdim,
                  niter_gmm=16, niter_tmat=16,
                  downsample=2, stochastic_downsample=True,
                  device='gpu', name="VoxCelebIvec")
if not ivec.is_fitted:
  ivec.fit(X, sad=ds['sad'], indices=train_indices,
           extract_ivec=True, keep_stats=False)
# ====== extract train i-vector ====== #
I_train = F.MmapData(ivec.ivec_path, read_only=True)
name_train = np.genfromtxt(ivec.name_path,
                           dtype=str)
# ====== extract test i-vector ====== #
I_test = ivec.transform(X, sad=ds['sad'], indices=test_indices,
                        save_ivecs=True, name='test')
name_test = np.genfromtxt(ivec.get_name_path(name='test'),
                          dtype=str)
print("Train i-vectors:", ctext(I_train, 'cyan'))
print("Test i-vectors:", ctext(I_test, 'cyan'))
print(ivec)
