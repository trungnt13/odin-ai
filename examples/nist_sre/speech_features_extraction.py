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
from odin.utils import ctext, unique_labels, Progbar, UnitTimer, stdio
from odin import fuel as F, preprocessing as pp
from utils import (PATH_FEATURE_EXTRACTION_LOG, PATH_ACOUSTIC, PATH_ACOUSTIC_FIG,
                   FeatureConfigs)

stdio(PATH_FEATURE_EXTRACTION_LOG)
audio = F.TIDIGITS.load()
print(audio)
all_files = sorted(list(audio['indices'].keys()))
debug = False
# ===========================================================================
# Extractor
# ===========================================================================
extractors = pp.make_pipeline(steps=[
    pp.speech.AudioReader(sr=FeatureConfigs.sr, dataset=audio),
    pp.speech.PreEmphasis(coeff=0.97),
    pp.speech.Dithering(),
    # ====== STFT ====== #
    pp.speech.STFTExtractor(frame_length=FeatureConfigs.frame_length,
                            step_length=FeatureConfigs.step_length,
                            n_fft=FeatureConfigs.n_fft, window=FeatureConfigs.window),
    pp.base.RenameFeatures(input_name='stft_energy', output_name='energy'),
    pp.speech.PowerSpecExtractor(power=2.0, output_name='spec'),
    pp.base.DeleteFeatures(input_name=['stft', 'raw']),
    # ====== SAD ====== #
    pp.speech.SADextractor(nb_mixture=3, nb_train_it=25,
                           input_name='energy', output_name='sad'),
    # ====== for x-vector ====== #
    pp.speech.MelsSpecExtractor(n_mels=24,
                                fmin=FeatureConfigs.fmin,
                                fmax=FeatureConfigs.fmax,
                                top_db=80.0,
                                input_name=('spec', 'sr'), output_name='mspec24'),
    # ====== BNF ====== #
    pp.speech.MelsSpecExtractor(n_mels=FeatureConfigs.n_mels,
                                fmin=FeatureConfigs.fmin,
                                fmax=FeatureConfigs.fmax,
                                top_db=80.0,
                                input_name=('spec', 'sr'), output_name='mspec'),
    pp.speech.MFCCsExtractor(n_ceps=FeatureConfigs.n_ceps, remove_first_coef=False,
                             input_name='mspec', output_name='mfcc'),
    pp.speech.BNFExtractorCPU(input_name='mfcc', output_name='bnf',
                              stack_context=10, pre_mvn=True,
                              sad_name='sad', remove_non_speech=False,
                              network=N.models.BNF_2048_MFCC40, batch_size=128),
    # ====== SDC ====== #
    pp.speech.MFCCsExtractor(n_ceps=7, remove_first_coef=True,
                             input_name='mspec', output_name='sdc'),
    pp.speech.RASTAfilter(rasta=True, input_name='sdc', output_name='sdc'),
    # ====== normalization ====== #
    pp.speech.AcousticNorm(mean_var_norm=True, windowed_mean_var_norm=True,
                           sad_name=None, ignore_sad_error=True,
                           input_name=('spec', 'mspec', 'mspec24',
                                       'mfcc', 'sdc', 'bnf')),
    pp.base.RunningStatistics(),
    # ====== post processing ====== #
    pp.base.EqualizeShape0(input_name=('spec', 'mspec', 'mspec24',
                                       'mfcc', 'sdc', 'bnf',
                                       'energy', 'sad')),
    pp.base.AsType(dtype='float16'),
], debug=debug)
# If debug is ran, tensorflow session created,
# multi-processing will be stopped during execution of tensorflow BNF
if debug:
  for i, name in enumerate(all_files[:8]):
    tmp = extractors.transform(name)
    print('\n')
    V.plot_multiple_features(tmp, title=name)
  V.plot_save(PATH_ACOUSTIC_FIG + '.pdf')
  exit()
# ===========================================================================
# Processor
# ===========================================================================
processor = pp.FeatureProcessor(jobs=all_files, path=PATH_ACOUSTIC,
                                extractor=extractors,
                                n_cache=0.12, ncpu=None, override=True)
with UnitTimer():
  processor.run()
readme_path = os.path.join(audio.path, [i for i in os.listdir(audio.path)
                                        if 'README' in i][0])
shutil.copy(readme_path,
            os.path.join(PATH_ACOUSTIC, 'README.md'))
pp.calculate_pca(processor, override=True)
# ====== check the preprocessed dataset ====== #
ds = F.Dataset(PATH_ACOUSTIC, read_only=True)
print(ds)
pp.validate_features(ds, path=PATH_ACOUSTIC_FIG,
                     nb_samples=8, override=True)
# ====== print pipeline ====== #
print(processor)
# ====== check PCA components ====== #
for n in ds.keys():
  if '_pca' in n:
    pca = ds[n]
    if pca.components_ is None:
      print(ctext(n, 'yellow'), 'components is None !')
    elif np.any(np.isnan(pca.components_)):
      print(ctext(n, 'yellow'), 'contains NaN !')
    else:
      print(ctext(n, 'yellow'),
          ':', ' '.join(['%.2f' % i + '-' + '%.2f' % j
          for i, j in zip(pca.explained_variance_ratio_[:8],
                          pca.explained_variance_[:8])]))
