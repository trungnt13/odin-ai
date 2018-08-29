from __future__ import print_function, division, absolute_import

import numpy as np
import scipy as sp

from odin.preprocessing import signal, speech, make_pipeline, base

from matplotlib import pyplot as plt

AUDIO_PATH = '/tmp/test.wav'
# ===========================================================================
# Helper
# ===========================================================================
def formatted_printer(feats):
  feats = sorted(feats.items(), key=lambda x: x[0])
  text = []
  for name, val in feats:
    text.append([
        name,
        str(val.shape if hasattr(val, 'shape') else val),
        str(val.dtype if hasattr(val, 'dtype') else val.__class__.__name__)
    ])
  max_len = [max([len(t[i]) for t in text])
             for i in range(len(text[0]))]
  fmt = '  '.join(['%-' + ('%ds' % l) for l in max_len])
  for line in text:
    print(fmt % tuple(line))
# ===========================================================================
# More detail pipeline
# ===========================================================================
pp1 = make_pipeline(steps=[
    speech.AudioReader(),
    speech.STFTExtractor(frame_length=0.025, padding=False),
    # spectra analysis
    speech.PowerSpecExtractor(output_name='spec', power=1.0),
    speech.PowerSpecExtractor(output_name='pspec', power=2.0),
    speech.Power2Db(input_name='pspec', output_name='db'),
    # Cepstra analysis
    speech.MelsSpecExtractor(n_mels=40, input_name=('pspec', 'sr')),
    speech.MFCCsExtractor(n_ceps=13, input_name='mspec'),
    # others
    speech.PitchExtractor(frame_length=0.025, f0=True),
    speech.SADextractor(input_name='stft_energy'),
    speech.RASTAfilter(input_name='mfcc', output_name='rasta'),
    base.EqualizeShape0(input_name=None),
    speech.AcousticNorm(input_name=('mfcc', 'mspec', 'spec'),
                        output_name=('mfcc_norm', 'mspec_norm', 'spec_norm')),
    speech.ApplyingSAD(input_name='mfcc', output_name='mfcc_sad'),
    base.StackFeatures(n_context=4, input_name='mfcc')
])
formatted_printer(feats=pp1.transform(AUDIO_PATH))
print("///////////////////////////")
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
    base.AsType(dtype='float16'),
    base.DuplicateFeatures('spec', 'mag'),
    base.DeleteFeatures('spec')
])
formatted_printer(feats=pp2.transform(AUDIO_PATH))
print("///////////////////////////")
# ===========================================================================
# OpenSMILE
# ===========================================================================
pp3 = make_pipeline(steps=[
    speech.AudioReader(),
    speech.Dithering(output_name='dither'),
    speech.PreEmphasis(coeff=0.97, output_name='preemphasis'),
    speech.openSMILEpitch(frame_length=0.025),
    speech.openSMILEf0(frame_length=0.025),
    speech.openSMILEloudness(frame_length=0.025),
    speech.openSMILEsad(frame_length=0.025),
])
formatted_printer(feats=pp3.transform(AUDIO_PATH))
