from __future__ import print_function, division, absolute_import
import os
os.environ['ODIN'] = 'float32,gpu'

import numpy as np

from odin import ml
from odin import fuel as F
from odin.utils import args_parse, ctext, stdio, Progbar

from utils import (get_model_path, prepare_ivec_data, csv2mat,
                   TRAIN_DATA)
# ===========================================================================
# Configs
# ===========================================================================
args = args_parse([
    ('recipe', 'the name of function defined in feature_recipes.py', None),
    ('-nmix', "Number of GMM mixture", None, 2048),
    ('-tdim', "Dimension of t-matrix", None, 600),
    ('-feat', "Acoustic feature", ('mspec', 'bnf'), 'bnf'),
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
EXP_DIR, MODEL_PATH, LOG_PATH, TRAIN_PATH, TEST_PATH = get_model_path('ivec', args)
stdio(LOG_PATH)
# ===========================================================================
# Load dataset
# ===========================================================================
X, train, test = prepare_ivec_data(args.recipe, FEAT)
# ===========================================================================
# Training I-vector model
# ===========================================================================
ivec = ml.Ivector(path=MODEL_PATH, nmix=args.nmix, tv_dim=args.tdim,
                  niter_gmm=16, niter_tmat=16,
                  downsample=2, stochastic_downsample=True,
                  device='gpu', name="VoxCelebIvec")
ivec.fit(X, indices=train,
         extract_ivecs=True, keep_stats=False)
# ====== extract train i-vector ====== #
I_train = F.MmapData(ivec.ivec_path, read_only=True)
name_train = np.genfromtxt(ivec.name_path, dtype=str)
print("Train i-vectors:", ctext(I_train, 'cyan'))
# save train i-vectors to csv
prog = Progbar(target=len(name_train),
               print_report=True, print_summary=True,
               name="Saving train i-vectors")
with open(TRAIN_PATH, 'w') as f_train:
  for i, name in enumerate(name_train):
    spk = TRAIN_DATA[name]
    vec = I_train[i]
    f_train.write('\t'.join([str(spk)] + [str(v) for v in vec]) + '\n')
    prog.add(1)
# ====== extract test i-vector ====== #
test = sorted(test.items(), key=lambda x: x[0])
I_test = ivec.transform(X, indices=test,
                        save_ivecs=False, keep_stats=False)
# save test i-vector to csv
with open(TEST_PATH, 'w') as f_test:
  for (name, (start, end)), z in zip(test, I_test):
    f_test.write('\t'.join([name] + [str(i) for i in z]) + '\n')
# ====== print the model ====== #
csv2mat(exp_dir=EXP_DIR)
print(ivec)
