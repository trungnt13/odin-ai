from odin import nnet as N
from odin import preprocessing as pp
from utils import WAV_FILES

def feat_sad(debugging):
  bnf_network = N.models.BNF_2048_MFCC40()
  recipe = pp.make_pipeline(steps=[
      pp.speech.AudioReader(sr=16000, sr_new=8000,
                            best_resample=True, remove_dc=True),
      pp.speech.PreEmphasis(coeff=0.97),
      pp.base.Converter(converter=WAV_FILES,
                        input_name='path', output_name='name'),
      # ====== STFT ====== #
      pp.speech.STFTExtractor(frame_length=0.025, step_length=0.010,
                              window='hamm', n_fft=512),
      # ====== SAD ====== #
      pp.base.RenameFeatures(input_name='stft_energy', output_name='energy'),
      pp.speech.SADextractor(nb_mixture=3, smooth_window=3,
                             input_name='energy', output_name='sad'),
      # ====== spectrogram ====== #
      pp.speech.PowerSpecExtractor(power=2.0, output_name='spec'),
      pp.speech.MelsSpecExtractor(n_mels=24, fmin=100, fmax=4000,
                                  input_name='spec', output_name='mspec'),
      pp.speech.MFCCsExtractor(n_ceps=20, remove_first_coef=True,
                               input_name='mspec', output_name='mfcc'),
      pp.base.DeltaExtractor(input_name='mfcc', order=(0, 1, 2)),
      pp.speech.ApplyingSAD(input_name=('mspec', 'mfcc')),
      # ====== BNF ====== #
      pp.speech.MelsSpecExtractor(n_mels=40, fmin=100, fmax=4000,
                                  input_name='spec', output_name='mspec_bnf'),
      pp.speech.MFCCsExtractor(n_ceps=40, remove_first_coef=False,
                               input_name='mspec_bnf', output_name='mfcc_bnf'),
      pp.base.AsType(dtype='float32', input_name='mfcc_bnf'),
      pp.speech.BNFExtractor(input_name='mfcc_bnf', output_name='bnf',
                             sad_name='sad',
                             network=bnf_network,
                             remove_non_speech=True,
                             stack_context=10, pre_mvn=True,
                             batch_size=5218),
      # ====== normalization ====== #
      pp.speech.AcousticNorm(input_name=('mspec', 'bnf', 'mfcc'),
                             mean_var_norm=True,
                             windowed_mean_var_norm=True,
                             win_length=301),
      # ====== cleaning ====== #
      pp.base.DeleteFeatures(input_name=('stft', 'raw', 'energy',
                                         'sad', 'sad_threshold',
                                         'spec', 'mspec_bnf', 'mfcc_bnf')),
      pp.base.AsType(dtype='float16')
  ], debug=debugging)
  return recipe

def feat_all(debugging):
  bnf_network = N.models.BNF_2048_MFCC40()
  recipe = pp.make_pipeline(steps=[
      pp.speech.AudioReader(sr=16000, sr_new=8000,
                            best_resample=True, remove_dc=True),
      pp.speech.PreEmphasis(coeff=0.97),
      pp.base.Converter(converter=WAV_FILES,
                        input_name='path', output_name='name'),
      # ====== STFT ====== #
      pp.speech.STFTExtractor(frame_length=0.025, step_length=0.010,
                              window='hamm', n_fft=512),
      # ====== SAD ====== #
      pp.base.RenameFeatures(input_name='stft_energy', output_name='energy'),
      pp.speech.SADextractor(nb_mixture=3, smooth_window=3,
                             input_name='energy', output_name='sad'),
      # ====== spectrogram ====== #
      pp.speech.PowerSpecExtractor(power=2.0, output_name='spec'),
      pp.speech.MelsSpecExtractor(n_mels=24, fmin=100, fmax=4000,
                                  input_name='spec', output_name='mspec'),
      pp.speech.MFCCsExtractor(n_ceps=20, remove_first_coef=True,
                               input_name='mspec', output_name='mfcc'),
      pp.base.DeltaExtractor(input_name='mfcc', order=(0, 1, 2)),
      # ====== BNF ====== #
      pp.speech.MelsSpecExtractor(n_mels=40, fmin=100, fmax=4000,
                                  input_name='spec', output_name='mspec_bnf'),
      pp.speech.MFCCsExtractor(n_ceps=40, remove_first_coef=False,
                               input_name='mspec_bnf', output_name='mfcc_bnf'),
      pp.base.AsType(dtype='float32', input_name='mfcc_bnf'),
      pp.speech.BNFExtractor(input_name='mfcc_bnf', output_name='bnf',
                             sad_name='sad',
                             network=bnf_network,
                             remove_non_speech=False,
                             stack_context=10, pre_mvn=True,
                             batch_size=5218),
      # ====== normalization ====== #
      pp.speech.AcousticNorm(input_name=('mspec', 'bnf', 'mfcc'),
                             mean_var_norm=True,
                             windowed_mean_var_norm=True,
                             win_length=301),
      # ====== cleaning ====== #
      pp.base.DeleteFeatures(input_name=('stft', 'raw', 'energy',
                                         'sad', 'sad_threshold',
                                         'spec', 'mspec_bnf', 'mfcc_bnf')),
      pp.base.AsType(dtype='float16')
  ], debug=debugging)
  return recipe
