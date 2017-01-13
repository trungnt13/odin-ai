from __future__ import print_function, division, absolute_import

import shutil
import os
from odin import fuel as F, utils

datapath = F.load_digit_wav()
output_path = utils.get_datasetpath(name='digit', override=True)
feat = F.SpeechProcessor(datapath, output_path, audio_ext='wav', fs=8000,
                         win=0.02, shift=0.01, n_filters=40, n_ceps=13,
                         delta_order=2, energy=True, pitch_threshold=0.5,
                         get_spec=True, get_mspec=True, get_mfcc=True,
                         get_pitch=False, get_vad=True,
                         save_stats=True, substitute_nan=None,
                         dtype='float32', datatype='memmap',
                         ncache=0.12, ncpu=4)
feat.run()
shutil.copy(os.path.join(datapath, 'README.md'),
            os.path.join(output_path, 'README.md'))
ds = F.Dataset(output_path, read_only=True)
print(ds)
