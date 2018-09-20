from __future__ import print_function, division, absolute_import

import numpy as np
import soundfile as sf

from odin import nnet as N
from odin import fuel as F, preprocessing as pp

from helpers import Config
# ===========================================================================
# Customized Extractor
# ===========================================================================
class SREAudioReader(pp.base.Extractor):
  """ SREAudioReader """

  def __init__(self):
    super(SREAudioReader, self).__init__(is_input_layer=True)

  def _transform(self, row):
    # `row`:
    #  0       1      2      3       4          5         6
    # path, channel, name, spkid, dataset, start_time, end_time
    path, channel, name, spkid, dataset, start_time, end_time = row
    if start_time == '-':
      start_time = None
    if end_time == '-':
      end_time = None
    # ====== read audio ====== #
    # for voxceleb1
    if dataset == 'voxceleb1':
      with open(path, 'rb') as f:
        y, sr = sf.read(f)
        y = pp.signal.resample(y, sr_orig=sr, sr_new=8000,
                               best_algorithm=True)
        sr = 8000
    # for sre, fisher and swb
    elif dataset[:3] == 'sre' or \
    dataset == 'swb' or \
    dataset == 'fisher':
      with open(path, 'rb') as f:
        y, sr = sf.read(f)
        y = pp.signal.resample(y, sr_orig=sr, sr_new=8000,
                               best_algorithm=True)
        if y.ndim == 2:
          y = y[:, int(channel)]
        sr = 8000
    # all other dataset: mix6, voxceleb2
    else:
      y, sr = pp.signal.anything2wav(inpath=path, outpath=None,
                                     channel=channel,
                                     dataset=dataset,
                                     start=start_time, end=end_time,
                                     sample_rate=Config.SAMPLE_RATE,
                                     return_data=True)
    # ====== error happen ignore file ====== #
    if len(y) == 0:
      return None
    # ====== remove DC offset ====== #
    y = y - np.mean(y, 0)
    duration = max(y.shape) / sr
    return {'raw': y, 'sr': sr, 'duration': duration, # in second
            'path': path, 'spkid': spkid, 'name': name,
            'dsname': dataset}

# ===========================================================================
# Extractors
# ===========================================================================
def mspec():
  extractors = pp.make_pipeline(steps=[
      SREAudioReader(),
      pp.speech.PreEmphasis(coeff=0.97, input_name='raw'),
      # ====== STFT ====== #
      pp.speech.STFTExtractor(frame_length=Config.FRAME_LENGTH,
                              step_length=Config.STEP_LENGTH,
                              n_fft=Config.NFFT, window=Config.WINDOW),
      pp.base.RenameFeatures(input_name='stft_energy', output_name='energy'),
      # ====== SAD ====== #
      pp.speech.SADextractor(nb_mixture=3, nb_train_it=25,
                             input_name='energy', output_name='sad'),
      # ====== for x-vector ====== #
      pp.speech.PowerSpecExtractor(power=2.0, input_name='stft', output_name='spec'),
      pp.speech.MelsSpecExtractor(n_mels=Config.NMELS,
                                  fmin=Config.FMIN, fmax=Config.FMAX,
                                  input_name=('spec', 'sr'), output_name='mspec'),
      pp.speech.ApplyingSAD(input_name='mspec', sad_name='sad'),
      # ====== normalization ====== #
      pp.speech.AcousticNorm(mean_var_norm=True, windowed_mean_var_norm=True,
                             win_length=301, input_name='mspec'),
      # ====== post processing ====== #
      pp.base.DeleteFeatures(input_name=['stft', 'spec', 'raw',
                                         'sad', 'energy', 'sad_threshold']),
      pp.base.AsType(dtype='float16'),
  ])
  return extractors

def bnf():
  bnf_network = N.models.BNF_2048_MFCC40()
  recipe = pp.make_pipeline(steps=[
      SREAudioReader(),
      pp.speech.PreEmphasis(coeff=0.97),
      # ====== STFT ====== #
      pp.speech.STFTExtractor(frame_length=Config.FRAME_LENGTH,
                              step_length=Config.STEP_LENGTH,
                              n_fft=Config.NFFT,
                              window=Config.WINDOW),
      # ====== SAD ====== #
      pp.base.RenameFeatures(input_name='stft_energy', output_name='energy'),
      pp.speech.SADextractor(nb_mixture=3, smooth_window=3,
                             input_name='energy', output_name='sad'),
      # ====== BNF ====== #
      pp.speech.PowerSpecExtractor(power=2.0, output_name='spec'),
      pp.speech.MelsSpecExtractor(n_mels=Config.NCEPS,
                                  fmin=Config.FMIN, fmax=Config.FMAX,
                                  input_name='spec', output_name='mspec'),
      pp.speech.MFCCsExtractor(n_ceps=Config.NCEPS, remove_first_coef=False,
                               input_name='mspec', output_name='mfcc'),
      pp.base.AsType(dtype='float32', input_name='mfcc'),
      pp.speech.BNFExtractor(input_name='mfcc', output_name='bnf', sad_name='sad',
                             network=bnf_network,
                             remove_non_speech=True,
                             stack_context=10, pre_mvn=True,
                             batch_size=5218),
      # ====== normalization ====== #
      pp.speech.AcousticNorm(input_name=('bnf',),
                             mean_var_norm=True,
                             windowed_mean_var_norm=True,
                             win_length=301),
      # ====== cleaning ====== #
      pp.base.DeleteFeatures(input_name=('stft', 'raw', 'energy',
                                         'sad', 'sad_threshold',
                                         'spec', 'mspec', 'mfcc')),
      pp.base.AsType(dtype='float16')
  ])
  return recipe
