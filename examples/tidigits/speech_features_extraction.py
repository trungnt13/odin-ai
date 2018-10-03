from __future__ import print_function, division, absolute_import
import matplotlib
matplotlib.use('Agg')

import os
os.environ['ODIN'] = 'float32,cpu=1,thread=1,gpu=1'
import sys
import shutil
import pickle

import numpy as np

from odin import visual as V, nnet as N
from odin.utils import ctext, unique_labels, Progbar, UnitTimer, args_parse
from odin.utils.mpi import cpu_count
from odin import fuel as F, preprocessing as pp
from utils import (PATH_ACOUSTIC, PATH_EXP, FeatureConfigs)

args = args_parse(descriptions=[
    ('--debug', 'enable debugging', None, False),
    ('-ncpu', 'if smaller than 1, auto select all possible CPU', None, 0)
])
audio = F.TIDIGITS.load()
print(audio)
all_files = sorted(list(audio['indices'].keys()))
# ===========================================================================
# Extractor
# ===========================================================================
bnf_network = N.models.BNF_2048_MFCC40()
extractors = pp.make_pipeline(steps=[
    pp.speech.AudioReader(sr=FeatureConfigs.sr, dataset=audio),
    pp.speech.PreEmphasis(coeff=0.97),
    pp.speech.Dithering(),
    # ====== STFT ====== #
    pp.speech.STFTExtractor(frame_length=FeatureConfigs.frame_length,
                            step_length=FeatureConfigs.step_length,
                            n_fft=FeatureConfigs.n_fft,
                            window=FeatureConfigs.window),
    # ====== SAD ====== #
    pp.base.RenameFeatures(input_name='stft_energy', output_name='energy'),
    pp.speech.SADgmm(nb_mixture=3, nb_train_it=25,
                           input_name='energy', output_name='sad'),
    # ====== for x-vector ====== #
    pp.speech.PowerSpecExtractor(power=2.0, output_name='spec'),
    pp.speech.MelsSpecExtractor(n_mels=24,
                                fmin=FeatureConfigs.fmin, fmax=FeatureConfigs.fmax,
                                input_name=('spec', 'sr'), output_name='mspec'),
    # ====== BNF ====== #
    pp.speech.MelsSpecExtractor(n_mels=FeatureConfigs.n_mels,
                                fmin=FeatureConfigs.fmin, fmax=FeatureConfigs.fmax,
                                input_name=('spec', 'sr'), output_name='mspec_bnf'),
    pp.speech.MFCCsExtractor(n_ceps=FeatureConfigs.n_ceps, remove_first_coef=False,
                             input_name='mspec', output_name='mfcc_bnf'),
    pp.base.AsType(dtype='float32', input_name='mfcc_bnf'),
    pp.speech.BNFExtractor(input_name='mfcc_bnf', output_name='bnf',
                           stack_context=10, pre_mvn=True,
                           sad_name='sad', remove_non_speech=False,
                           network=bnf_network,
                           batch_size=2048),
    # ====== MFCCs with deltas ====== #
    pp.speech.MFCCsExtractor(n_ceps=20, remove_first_coef=True,
                             input_name='mspec', output_name='mfcc'),
    pp.base.DeltaExtractor(input_name='mfcc', order=(0, 1, 2)),
    # ====== SDC ====== #
    pp.speech.MFCCsExtractor(n_ceps=7, remove_first_coef=True,
                             input_name='mspec', output_name='sdc'),
    pp.speech.RASTAfilter(rasta=True, input_name='sdc', output_name='sdc'),
    # ====== normalization ====== #
    pp.base.DeleteFeatures(input_name=('stft', 'spec',
                                       'mspec_bnf', 'mfcc_bnf',
                                       'sad_threshold')),
    pp.speech.AcousticNorm(mean_var_norm=True, windowed_mean_var_norm=True,
                           sad_name=None, ignore_sad_error=True,
                           input_name=('mspec', 'mfcc', 'sdc', 'bnf')),
    # ====== post processing ====== #
    pp.base.EqualizeShape0(input_name=('mspec', 'mfcc', 'sdc', 'bnf',
                                       'energy', 'sad')),
    pp.base.AsType(dtype='float16'),
], debug=args.debug)
# ====== enable debug mode ====== #
if args.debug:
  with np.warnings.catch_warnings():
    np.warnings.filterwarnings('ignore')
    for i, name in enumerate(all_files[:12]):
      tmp = extractors.transform(name)
      if isinstance(tmp, pp.base.ExtractorSignal):
        print(tmp)
        exit()
      else:
        V.plot_multiple_features(tmp, title=name)
    V.plot_save(os.path.join(PATH_EXP, 'feature_debug.pdf'))
    exit()
# ===========================================================================
# Processor
# ===========================================================================
with np.warnings.catch_warnings():
  np.warnings.filterwarnings('ignore')
  processor = pp.FeatureProcessor(jobs=all_files,
      path=PATH_ACOUSTIC,
      extractor=extractors,
      n_cache=0.12,
      ncpu =min(18, cpu_count() - 2) if args.ncpu <= 0 else int(args.ncpu),
      override=True,
      identifier='name',
      log_path=os.path.join(PATH_EXP, 'processor.log'),
      stop_on_failure=True # small dataset, enable stop on failure
  )
  with UnitTimer():
    processor.run()
  n_error = len(processor.error_log)
  print(processor)
# ====== copy readme and check the preprocessed dataset ====== #
if n_error == 0:
  readme_path = os.path.join(audio.path, [i for i in os.listdir(audio.path)
                                          if 'README' in i][0])
  shutil.copy(readme_path,
              os.path.join(PATH_ACOUSTIC, 'README.md'))

  ds = F.Dataset(PATH_ACOUSTIC, read_only=True)
  print(ds)
  pp.validate_features(ds,
                       path=os.path.join(PATH_EXP, 'acoustic'),
                       nb_samples=12, override=True)
else:
  print("%s errors happened during processing!" % ctext(n_error, 'red'))
