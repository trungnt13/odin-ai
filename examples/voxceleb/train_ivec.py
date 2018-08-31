from __future__ import print_function, division, absolute_import
import os
os.environ['ODIN'] = 'float32,gpu'

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
# ===========================================================================
# Training I-vector model
# ===========================================================================
# ivec = ml.Ivector(path=SAVE_PATH, nmix=8, tv_dim=8, niter_gmm=4, niter_tmat=4)
ivec = ml.Ivector(path=SAVE_PATH,
                  nmix=args.nmix,
                  tv_dim=args.tdim)
# from odin.ml import ivector
# ivector._input_data_2_md5(X=X, sad=ds['sad'], indices=train_indices)
ivec.fit(X, sad=ds['sad'], indices=train_indices)
