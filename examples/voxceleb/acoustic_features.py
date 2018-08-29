# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import matplotlib
matplotlib.use('Agg')

import os
os.environ['ODIN'] = 'float32,gpu'

from odin import backend as K, nnet as N, visual as V
from odin import preprocessing as pp
from odin.utils import args_parse

from const import (WAV_FILES, SAMPLED_WAV_FILE,
                   PATH_ACOUSTIC_FEAT, PATH_EXP)
# ===========================================================================
# Config
# ===========================================================================
args = args_parse(descriptions=[
    ('--debug', 'enable debug or not', None, False)
])
DEBUG = args.debug
# ===========================================================================
# Create the recipes
# ===========================================================================
recipe = pp.make_pipeline(steps=[
    pp.speech.AudioReader(sr=16000),
    pp.speech.PreEmphasis(coeff=0.97),
    pp.base.Converter(converter=WAV_FILES,
                      input_name='path', output_name='name'),
    # ====== STFT ====== #
    pp.speech.STFTExtractor(frame_length=0.025, step_length=0.005,
                            n_fft=512),
    pp.base.RenameFeatures(input_name='stft_energy',
                           output_name='energy'),
    # ====== spectrogram ====== #
    pp.speech.PowerSpecExtractor(power=2.0, output_name='spec'),
    pp.speech.MelsSpecExtractor(n_mels=24, fmin=64, fmax=None,
                                output_name='mspec'),
    pp.speech.MFCCsExtractor(n_ceps=20, remove_first_coef=True,
                             output_name='mfcc'),
    # ====== SAD ====== #
    pp.speech.SADextractor(nb_mixture=3, smooth_window=3,
                           output_name='sad'),
    pp.speech.AcousticNorm(input_name=('mfcc', 'mspec'), mean_var_norm=True,
                           windowed_mean_var_norm=True, win_length=121),
    pp.base.DeltaExtractor(input_name='mfcc', width=9, order=(0, 1, 2)),
    # ====== cleaning ====== #
    pp.base.DeleteFeatures(input_name=('stft', 'raw', 'spec',
                                       'sad_threshold', 'energy')),
    pp.base.AsType(dtype='float16')
], debug=DEBUG)
if DEBUG:
  for path, name in SAMPLED_WAV_FILE:
    feat = recipe.transform(path)
    V.plot_multiple_features(feat, title=feat['name'])
  V.plot_save()
  exit()
# ===========================================================================
# Prepare the processor
# ===========================================================================
jobs = list(WAV_FILES.keys())
processor = pp.FeatureProcessor(jobs=jobs, path=PATH_ACOUSTIC_FEAT,
                                extractor=recipe, n_cache=0.08,
                                ncpu=None, override=True)
processor.run()
pp.validate_features(processor, path=os.path.join(PATH_EXP, 'acoustic'),
                     override=True)
