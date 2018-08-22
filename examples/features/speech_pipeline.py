from __future__ import print_function, division, absolute_import

import numpy as np
import scipy as sp

from odin.preprocessing import signal, speech, make_pipeline, base

from matplotlib import pyplot as plt

AUDIO_PATH = '/tmp/test.wav'

# ===========================================================================
# More detail pipeline
# ===========================================================================
pp1 = make_pipeline(steps=[
    speech.AudioReader(),
    speech.STFTExtractor(frame_length=0.025, padding=False),
    speech.PowerSpecExtractor(output_name='spec', power=1.0),
    speech.PowerSpecExtractor(output_name='pspec'),
    speech.MelsSpecExtractor(n_mels=40, input_name=('pspec', 'sr')),
    speech.MFCCsExtractor(n_ceps=13),
    speech.Power2Db(input_name='pspec', output_name='db'),
    speech.PitchExtractor(frame_length=0.025, f0=True),
    speech.SADextractor(input_name='stft_energy'),
    speech.RASTAfilter(input_name='mfcc', output_name='rasta'),
    speech.ApplyingSAD(input_name='mfcc', output_name='mfcc_sad'),
    speech.AcousticNorm(input_name=('mfcc', 'mfcc_sad'),
                        output_name=('mfcc_norm', 'mfcc_sad_norm')),
    base.EqualizeShape0(input_name=None),
    base.StackFeatures(n_context=4, input_name='mfcc')
])
for name, val in sorted(pp1.transform(AUDIO_PATH).items(),
                        key=lambda x: x[0]):
  print(name,
    val.shape if hasattr(val, 'shape') else val,
    val.dtype if hasattr(val, 'dtype') else val.__class__.__name__)

print("///////////////////////////\n")
# ===========================================================================
# Fast pipeline
# ===========================================================================
pp2 = make_pipeline(steps=[
    speech.AudioReader(),
    speech.SpectraExtractor(frame_length=0.025, n_mels=40, n_ceps=13),
    speech.CQTExtractor(frame_length=0.025, n_mels=40, n_ceps=13),
    base.DeltaExtractor(input_name=('mspec', 'mfcc'),
                        output_name=('mspec_d', 'mfcc_d')),
    base.RunningStatistics(),
    base.AsType(dtype='float32'),
    base.DuplicateFeatures('spec', 'mag'),
    base.RemoveFeatures('spec')
])
for name, val in sorted(pp2.transform(AUDIO_PATH).items(),
                        key=lambda x: x[0]):
  print(name,
    val.shape if hasattr(val, 'shape') else val,
    val.dtype if hasattr(val, 'dtype') else val.__class__.__name__)

print("///////////////////////////\n")
# ===========================================================================
# OpenSMILE
# ===========================================================================
pp3 = make_pipeline(steps=[
    speech.AudioReader(),
    speech.openSMILEpitch(frame_length=0.025),
    speech.openSMILEf0(frame_length=0.025),
    speech.openSMILEloudness(frame_length=0.025),
    speech.openSMILEsad(frame_length=0.025),
])
for name, val in sorted(pp3.transform(AUDIO_PATH).items(),
                        key=lambda x: x[0]):
  print(name,
    val.shape if hasattr(val, 'shape') else val,
    val.dtype if hasattr(val, 'dtype') else val.__class__.__name__)
