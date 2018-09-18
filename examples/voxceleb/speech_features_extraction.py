# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import matplotlib
matplotlib.use('Agg')

import os
os.environ['ODIN'] = 'float32,gpu'

import numpy as np

from odin import backend as K, nnet as N, visual as V
from odin import preprocessing as pp
from odin.utils import args_parse, stdio
from odin.utils.mpi import cpu_count

from const import (WAV_FILES, SAMPLED_WAV_FILE,
                   PATH_ACOUSTIC_FEAT, PATH_EXP)
# ===========================================================================
# Config
# ===========================================================================
stdio(os.path.join(PATH_EXP, 'features_extraction.log'))
args = args_parse(descriptions=[
    ('--debug', 'enable debug or not', None, False)
])
DEBUG = args.debug
# ===========================================================================
# Create the recipes
# ===========================================================================
bnf_network = N.models.BNF_2048_MFCC40()
recipe = pp.make_pipeline(steps=[
    pp.speech.AudioReader(sr=16000, sr_new=8000),
    pp.speech.PreEmphasis(coeff=0.97),
    pp.base.Converter(converter=WAV_FILES,
                      input_name='path', output_name='name'),
    # ====== STFT ====== #
    pp.speech.STFTExtractor(frame_length=0.025, step_length=0.01,
                            window='hamm', n_fft=512),
    pp.base.RenameFeatures(input_name='stft_energy',
                           output_name='energy'),
    # ====== SAD ====== #
    pp.speech.SADextractor(nb_mixture=3, smooth_window=3,
                           input_name='energy', output_name='sad'),
    # ====== spectrogram ====== #
    pp.speech.PowerSpecExtractor(power=2.0, output_name='spec'),
    pp.speech.MelsSpecExtractor(n_mels=24, fmin=64, fmax=None,
                                input_name='spec', output_name='mspec'),
    pp.speech.ApplyingSAD(input_name='mspec'),
    # ====== BNF ====== #
    pp.speech.MelsSpecExtractor(n_mels=40, fmin=100, fmax=4000,
                                input_name='spec', output_name='mspec_bnf'),
    pp.speech.MFCCsExtractor(n_ceps=40, remove_first_coef=False,
                             input_name='mspec_bnf', output_name='mfcc'),
    pp.base.AsType(dtype='float32', input_name='mfcc'),
    pp.speech.BNFExtractor(input_name='mfcc', network=bnf_network,
                           output_name='bnf', sad_name='sad',
                           remove_non_speech=True,
                           stack_context=10, pre_mvn=True,
                           batch_size=5218),
    # ====== normalization ====== #
    pp.speech.AcousticNorm(input_name=('mfcc', 'bnf'), mean_var_norm=True,
                           windowed_mean_var_norm=True, win_length=301),
    # ====== cleaning ====== #
    pp.base.DeleteFeatures(input_name=('stft', 'raw', 'spec', 'mspec_bnf', 'mfcc',
                                       'sad_threshold', 'energy', 'sad')),
    pp.base.AsType(dtype='float16')
], debug=DEBUG)
if DEBUG:
  for path, name in SAMPLED_WAV_FILE:
    feat = recipe.transform(path)
    assert feat['bnf'].shape[0] == feat['mspec'].shape[0]
    V.plot_multiple_features(feat, title=feat['name'])
  V.plot_save(os.path.join(PATH_EXP, 'features_debug.pdf'))
  exit()
# ===========================================================================
# Prepare the processor
# ===========================================================================
with np.warnings.catch_warnings():
  np.warnings.filterwarnings('ignore')
  jobs = list(WAV_FILES.keys())
  processor = pp.FeatureProcessor(jobs=jobs, path=PATH_ACOUSTIC_FEAT,
                                  extractor=recipe, n_cache=0.12,
                                  ncpu=min(18, cpu_count() - 1),
                                  override=True)
  processor.run()
  pp.validate_features(processor, path=os.path.join(PATH_EXP, 'acoustic'),
                       override=True)
