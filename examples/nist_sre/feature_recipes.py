from __future__ import print_function, division, absolute_import

import os
import subprocess
from io import BytesIO
from six import string_types
from collections import defaultdict

import numpy as np
import soundfile as sf

from odin import nnet as N
from odin import fuel as F, preprocessing as pp

from helpers import Config, PATH_ACOUSTIC_FEATURES, CURRENT_STATE, SystemStates
# ===========================================================================
# Customized Audio Reader Extractor
# `row`:
#  0       1      2      3       4          5         6
# path, channel, name, spkid, dataset, start_time, end_time
# ===========================================================================
class SREAudioReader(pp.base.Extractor):
  """ SREAudioReader """

  def __init__(self):
    super(SREAudioReader, self).__init__(is_input_layer=True)

  def _transform(self, row):
    if len(row) == 7:
      path, channel, name, spkid, dataset, start_time, end_time = row
    else:
      path, channel, name, spkid, dataset = row[:5]
      start_time = None
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
    elif (dataset[:3] == 'sre' or
     dataset == 'swb' or
     dataset == 'fisher'):
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
    ret = {'raw': y, 'sr': sr, 'duration': duration, # in second
           'path': path,
           'spkid': spkid,
           'name': name,
           'dsname': dataset}
    return ret

class SREAugmentor(pp.base.Extractor):
  """ SREAugmentor
  New name for each utterance is:
    [utt_name]/[noise1_name]/[noise2_name]...
  """

  def __init__(self, noise_ds):
    super(SREAugmentor, self).__init__(is_input_layer=True)
    from helpers import ALL_NOISE
    self.noise_ds = str(noise_ds)
    assert self.noise_ds in ALL_NOISE, \
    "Cannot find noise dataset with name: %s; given following option: %s" % \
    (self.noise_ds, ', '.join(list(ALL_NOISE.keys())))
    # ====== mapping noise_type -> list of row ====== #
    self.noise_type = defaultdict(list)
    for path, channel, name, ntype, duration in ALL_NOISE[self.noise_ds]:
      self.noise_type[ntype].append((path, name, duration))
    self.noise_type = {k: np.array(v)
                       for k, v in self.noise_type.items()}
    # ====== fixed RandomState ====== #
    self.rand = np.random.RandomState(seed=Config.SUPER_SEED)

  def _transform(self, row):
    # we need to know the audio duration here
    path, channel, name, spkid, dataset, start_time, end_time, duration = row
    duration = float(duration)
    # rirs: mediumroom, smallroom
    # musan: music, speech, noise
    noise_type = self.rand.choice(a=sorted((self.noise_type.keys())),
                                  size=1, replace=False)[0]
    noise_data = self.noise_type[noise_type]
    # ====== wav command ====== #
    cmd_wav = pp.signal.anything2wav(inpath=path, outpath=None,
                                     channel=channel,
                                     dataset=dataset,
                                     start=start_time, end=end_time,
                                     sample_rate=Config.SAMPLE_RATE,
                                     codec='pcm16',
                                     return_data=False)
    # ====== reverberation ====== #
    if self.noise_ds == 'rirs':
      idx = self.rand.randint(low=0, high=len(noise_data), size=1, dtype=int)[0]
      noise_path, noise_name, noise_dur = noise_data[idx]
      cmd = '%s | ' + \
      'wav-reverberate --shift-output=true --impulse-response="sox %s -r 8000 -t wav - |"  - -'
      cmd = cmd % (cmd_wav, noise_path)
      # update the name
      name += '/%s' % noise_name
    # ====== MUSAN ====== #
    elif self.noise_ds == 'musan':
      ### noise (in kaldi: noise snrs is choose from one of
      # following value 15:10:5:0)
      if noise_type == 'noise':
        # duration until start the next noise audio
        noise_interval = 1 # in second
        noise = []; noise_start = []; noise_snrs = []
        curr_dur = 0; indices = np.arange(len(noise_data))
        # adding noise until the end of the utterance
        while curr_dur < duration:
          idx = self.rand.choice(indices, size=1)[0]
          noise_path, noise_name, noise_dur = noise_data[idx]
          # this does not apply anymore if we don't have enough
          # noise duration
          if len(noise) < len(noise_data) and noise_path in noise:
            continue
          noise.append(noise_path)
          noise_start.append(curr_dur)
          noise_snrs.append(self.rand.randint(low=0, high=15, size=1, dtype=int)[0])
          curr_dur += float(noise_dur) + noise_interval
          # update the name
          name += '/%s' % noise_name
        # start creating command, careful all space must be there
        cmd = '%s | wav-reverberate --shift-output=true --additive-signals=\'' % cmd_wav
        for path in noise:
          cmd += 'sox -t wav %s -r 8k -t wav - |,' % path
        cmd = cmd[:-1]
        cmd += '\' --start-times=\'%s\' ' % ','.join(['%f' % i for i in noise_start])
        cmd += '--snrs=\'%s\' - -' % ','.join(['%d' % i for i in noise_snrs])
      ### music in background
      elif noise_type == 'music':
        idx = self.rand.randint(low=0, high=len(noise_data), size=1, dtype=int)[0]
        noise_path, noise_name, noise_dur = noise_data[idx]
        snrs = self.rand.randint(low=5, high=15, size=1, dtype=int)[0]
        cmd = '%s | ' + \
        'wav-reverberate --shift-output=true --additive-signals=' + \
        '\'wav-reverberate --duration=%f "sox -t wav %s -r 8k -t wav - |" - |\' ' + \
        '--start-times=\'0\' --snrs=\'%d\' - -'
        cmd = cmd % (cmd_wav, duration, noise_path, snrs)
        # update the name
        name += '/%s' % noise_name
      ### combined multiple speech from different speakers
      #(in kaldi: snrs is one of 20:17:15:13)
      elif noise_type == 'speech':
        n_speaker = self.rand.randint(3, 7, size=1, dtype=int)[0]
        idx = self.rand.choice(a=np.arange(len(noise_data)),
                               size=n_speaker, replace=False)
        speech = noise_data[idx]
        snrs = self.rand.randint(low=13, high=20, size=n_speaker, dtype=int)
        # start creating command, careful all space must be there
        cmd = '%s | wav-reverberate --shift-output=true --additive-signals=\'' % cmd_wav
        for spk_path, spk_name, spk_dur in speech:
          cmd += 'wav-reverberate --duration=%f "sox -t wav %s -r 8k -t wav - |" - |,' % \
          (duration, spk_path)
          # update the name
          name += '/%s' % spk_name
        cmd = cmd[:-1] # remove the `,` in the end
        cmd += '\' --start-times=\'%s\' ' % ','.join(['0'] * n_speaker)
        cmd += '--snrs=\'%s\' - -' % ','.join(['%d' % i for i in snrs])
      ### Error
      else:
        raise RuntimeError("Unknown MUSAN noise type: %s" % noise_type)
    # ====== error ====== #
    else:
      raise RuntimeError("No support noise dataset with name: %s" % self.noise_ds)
    # ====== get the data ====== #
    try:
      with subprocess.Popen(args=cmd, shell=True,
                           stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE) as p:
        data = BytesIO(p.stdout.read())
        y, sr = sf.read(data)
    except Exception as e:
      signal = pp.base.ExtractorSignal()
      signal.set_message(extractor=self,
                         msg=str(e),
                         last_input=row)
      signal.set_action(self.action)
      return signal
    # ====== error happen ignore file ====== #
    if len(y) == 0:
      return None
    # ====== remove DC offset ====== #
    y = y - np.mean(y, 0)
    duration = max(y.shape) / sr
    return {'raw': y, 'sr': sr, 'duration': duration, # in second
            'path': path, 'spkid': spkid, 'name': name,
            'dsname': dataset,
            'dsnoise': self.noise_ds,
            'noisetype': noise_type,
            'cmd': cmd}

class SADreader(pp.base.Extractor):
  """ This class read SAD label from acoustic dataset and
  use it for augmentation dataset
  """

  def __init__(self, ds_path):
    super(SADreader, self).__init__()
    indices = os.path.join(ds_path, 'indices_sad')
    assert os.path.isfile(indices), "Cannot find indices at path: %s" % indices
    self.indices = F.MmapDict(indices, read_only=True)

    data = os.path.join(ds_path, 'sad')
    assert os.path.isfile(data), "Cannot find SAD at path: %s" % data
    self.data = F.MmapData(data, read_only=True)

  def _transform(self, X):
    name = X['name'].split('/')[0]
    if name not in self.indices:
      return None
    start, end = self.indices[name]
    sad = self.data[start:end][:]
    return {'sad': sad}

# ===========================================================================
# Extractor
# NOTE: you must save the SAD label for augmentation data later
# ===========================================================================
def mfcc(augmentation=None):
  delete_list = ['stft', 'spec', 'raw',
                 'mfcc_energy', 'sad_threshold']
  if augmentation is not None:
    delete_list.append('sad')

  extractors = pp.make_pipeline(steps=[
      SREAugmentor(augmentation)
      if isinstance(augmentation, string_types) else
      SREAudioReader(),
      pp.speech.PreEmphasis(coeff=0.97, input_name='raw'),
      # ====== STFT ====== #
      pp.speech.STFTExtractor(frame_length=Config.FRAME_LENGTH,
                              step_length=Config.STEP_LENGTH,
                              n_fft=Config.NFFT, window=Config.WINDOW,
                              padding=False, energy=False),
      # ====== for x-vector ====== #
      pp.speech.PowerSpecExtractor(power=2.0,
                                   input_name='stft', output_name='spec'),
      pp.speech.MelsSpecExtractor(n_mels=24, fmin=20, fmax=3700,
                                  input_name=('spec', 'sr'), output_name='mspec'),
      pp.speech.MFCCsExtractor(n_ceps=24,
                               remove_first_coef=True, first_coef_energy=True,
                               input_name='mspec', output_name='mfcc'),
      # ====== extract SAD ====== #
      pp.speech.SADthreshold(energy_threshold=0.5, energy_mean_scale=0.5,
                             frame_context=2, proportion_threshold=0.12,
                             smooth_window=5,
                             input_name='mfcc_energy', output_name='sad')
      if augmentation is None else
      SADreader(ds_path=os.path.join(PATH_ACOUSTIC_FEATURES, 'mfcc')),
      pp.speech.ApplyingSAD(input_name=('mspec', 'mfcc'), sad_name='sad',
          keep_unvoiced=False if CURRENT_STATE == SystemStates.EXTRACT_FEATURES else True),
      # ====== normalization ====== #
      pp.speech.AcousticNorm(mean_var_norm=True, windowed_mean_var_norm=True,
                             win_length=301, input_name=('mspec', 'mfcc')),
      # ====== post processing ====== #
      pp.base.DeleteFeatures(input_name=delete_list),
      pp.base.AsType(dtype='float16'),
  ])
  return extractors

def bnf(augmentation=None):
  raise NotImplementedError
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
      pp.speech.SADgmm(nb_mixture=3, smooth_window=3,
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
                                         'sad_threshold',
                                         'spec', 'mspec', 'mfcc')),
      pp.base.AsType(dtype='float16')
  ])
  return recipe
