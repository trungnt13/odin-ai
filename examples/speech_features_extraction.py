# ===========================================================================
# Without PCA:
#   ncpu=1:  16s
#   ncpu=2:  9.82
#   ncpu=4:  5.9s
#   ncpu=8:  4.3
#   ncpu=12: 4.0
# ===========================================================================
from __future__ import print_function, division, absolute_import
import matplotlib
matplotlib.use('Agg')

import numpy as np
import shutil
import os
from odin import fuel as F, utils
from collections import defaultdict

datapath = F.load_digit_wav()
output_path = utils.get_datasetpath(name='digit', override=True)
feat = F.SpeechProcessor(datapath, output_path, audio_ext='wav', sr_new=16000,
                         win=0.02, shift=0.01, nb_melfilters=40, nb_ceps=13,
                         get_delta=2, get_energy=True, get_phase=True,
                         get_spec=True, get_mspec=True, get_mfcc=True,
                         get_pitch=True, get_vad=True, get_qspec=True,
                         pitch_threshold=0.8, cqt_bins=96,
                         vad_smooth=12, vad_minlen=0.1,
                         pca=True, pca_whiten=False, center=True,
                         save_stats=True, substitute_nan=None,
                         dtype='float32', datatype='memmap',
                         ncache=0.12, ncpu=8)
with utils.UnitTimer():
    feat.run()
shutil.copy(os.path.join(datapath, 'README.md'),
            os.path.join(output_path, 'README.md'))
# ====== check the preprocessed dataset ====== #
ds = F.Dataset(output_path, read_only=True)
print('Output path:', output_path)
print(ds)

for n in ds.keys():
    if '_pca' in n:
        pca = ds[n]
        if pca.components_ is None:
            print(n, 'is None !')
        elif np.any(np.isnan(pca.components_)):
            print(n, 'contains NaN !')
        else:
            print(n, ':', ' '.join(['%.2f' % i + '-' + '%.2f' % j
                for i, j in zip(pca.explained_variance_ratio_[:8],
                                pca.explained_variance_[:8])]))
x = ds['vadids'][:, 1] - ds['vadids'][:, 0]
print("Check vadids:", np.all(x > 0))
