from __future__ import print_function, division, absolute_import
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import os
os.environ['ODIN'] = 'float32,gpu,seed=5218'

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import init_ops

from odin.utils import ctext, Progbar, mpi, cache_disk
from odin import fuel as F, nnet as N, backend as K
from odin import preprocessing as pp
from odin.visual import plot_multiple_features, plot_save, plot_spectrogram
from odin.config import get_random_state
from odin.preprocessing.signal import segment_axis, one_hot
# ===========================================================================
# Configuration
# ===========================================================================
SR = 8000
NFFT = 512
NMELS = 40
NCEPS = 40
FRAME_LENGTH = 0.025
STEP_LENGTH = 0.01
FMIN = 100
FMAX = 4000
WINDOW = 'hamm'
rand = get_random_state()
# ====== for training ====== #
FEATURE = 'mspec'
PAD_MODE = 'pre'
LEARNING_RATE = 0.01
# ====== helpers ====== #
file2name = lambda f: os.path.basename(f).replace('.wav', '')
# ===========================================================================
# Load the wav
# ===========================================================================
files, meta = F.FSDD.load()
all_speakers = [i[0] for i in meta[1:]]
all_numbers = sorted(set([int(os.path.basename(f)[0]) for f in files]))
nb_classes = len(all_numbers)
print("#Files:", ctext(len(files), 'cyan'))
print("#Speakers:", ctext(all_speakers, 'cyan'))
print("Labels:", ctext(all_numbers, 'cyan'))
# ====== split train, valid, test by speakers ====== #
train_spk = rand.choice(a=all_speakers, size=2, replace=False)
test_spk = [i for i in all_speakers if i not in train_spk]
print("Train speakers:", ctext(train_spk, 'cyan'))
print("Test speakers:", ctext(test_spk, 'cyan'))
# ====== get all utterances from given speakers ====== #
train_utt = [file2name(f)
             for f in files
             if any(s in file2name(f) for s in train_spk)]
valid_utt = rand.choice(a=train_utt, size=int(0.2 * len(train_utt)),
                        replace=False)
train_utt = [i for i in train_utt if i not in valid_utt]

test_utt = [file2name(f)
            for f in files
            if any(s in file2name(f) for s in test_spk)]
print('#TrainUtt:', ctext(len(train_utt), 'cyan'))
print('#ValidUtt:', ctext(len(valid_utt), 'cyan'))
print('#TestUtt:', ctext(len(test_utt), 'cyan'))
# ===========================================================================
# Speech processing
# ===========================================================================
@cache_disk
def extract_acoustic_features(SR=SR, NFFT=NFFT, NMELS=NMELS,
                              NCEPS=NCEPS, FRAME_LENGTH=FRAME_LENGTH,
                              STEP_LENGTH=STEP_LENGTH,
                              FMIN=FMIN, FMAX=FMAX, WINDOW=WINDOW):
  pipeline = pp.make_pipeline(steps=[
      pp.speech.AudioReader(sr=SR, sr_new=None, remove_dc_n_dither=False,
                            preemphasis=None),
      pp.base.NameConverter(converter=file2name, input_name='path'),
      pp.speech.STFTExtractor(frame_length=FRAME_LENGTH, step_length=STEP_LENGTH,
                              window='hamm', nfft=NFFT, energy=True),
      pp.speech.SADextractor(nb_mixture=3, smooth_window=3),
      # ====== spectrogram ====== #
      pp.speech.PowerSpecExtractor(power=2.0),
      pp.speech.MelsSpecExtractor(nmels=NMELS, fmin=FMIN, fmax=FMAX, top_db=80.0),
      pp.speech.MFCCsExtractor(nceps=NCEPS, output_name='mfcc',
                               remove_first_coef=False),
      # ====== SDC features ====== #
      pp.speech.MFCCsExtractor(nceps=7, output_name='sdc',
                               remove_first_coef=True),
      pp.speech.RASTAfilter(rasta=True, sdc=1,
                            input_name='sdc', output_name='sdc'),
      # ====== post processing ====== #
      pp.speech.AcousticNorm(mean_var_norm=True, windowed_mean_var_norm=True,
                             win_length=301,
                             feat_name=('spec', 'mspec', 'mfcc', 'sdc')),
      pp.base.RemoveFeatures(feat_name=('stft')),
  ], debug=False)
  features = {}
  prog = Progbar(target=len(files), print_report=True, print_summary=True,
                 name="Extracting acoustic features")
  for X in mpi.MPI(jobs=files, func=pipeline.transform,
                   ncpu=4, batch=1):
    prog['name'] = X['name']
    prog.add(1)
    features[X['name']] = X
  return features
# ====== find the longest utterances ====== #
features = extract_acoustic_features()
LONGEST_UTT = max(len(i['energy'])
                  for i in features.values())
print("Longest utterance:", ctext(LONGEST_UTT, 'cyan'))
# ===========================================================================
# Train test spliting the dataset
# ===========================================================================
@cache_disk
def generate_data(flist, LONGEST_UTT=LONGEST_UTT, PAD_MODE=PAD_MODE):
  flist = np.array(flist)
  np.random.shuffle(flist)
  X = []
  y = []
  for f in flist:
    feat = segment_axis(features[f][FEATURE],
                        frame_length=LONGEST_UTT, step_length=1,
                        end='pad', pad_value=0, pad_mode=PAD_MODE)
    label = int(f.split('_')[0])
    X.append(feat)
    y.append(label)
  X = np.concatenate(X, axis=0)
  y = np.array(y)
  indices = np.random.permutation(X.shape[0])
  return X[indices], one_hot(y[indices], nb_classes=len(all_numbers))

X_train, y_train = generate_data(train_utt)
X_valid, y_valid = generate_data(valid_utt)
X_test, y_test = generate_data(test_utt)
input_ndim = X_train.shape[-1]
print('train:', X_train.shape, y_train.shape)
print('valid:', X_valid.shape, y_valid.shape)
print('test:', X_test.shape, y_test.shape)
# ===========================================================================
# Create the network
# ===========================================================================
# (num_samples, num_timestep, num_features)
INPUT_SHAPE = (None, X_train.shape[1], X_train.shape[2])
X = K.placeholder(shape=INPUT_SHAPE, name='X')
y = K.placeholder(shape=(None, len(all_numbers)), name='y')

# ====== all functions ====== #
f_encoder = N.Sequence(ops=[
    N.Dimshuffle(pattern=(0, 1, 2, 'x')),
    N.Conv(num_filters=32, filter_size=(7, 9), strides=1,
           pad='same', activation=K.linear),
    N.BatchNorm(activation=K.relu),

    N.Conv(num_filters=64, filter_size=(5, 7), strides=1,
           pad='same', activation=K.linear),
    N.BatchNorm(activation=K.relu),

    N.Flatten(outdim=3),
], debug=True, name="Encoder")

f_attn = N.Sequence(ops=[
    N.Dimshuffle(pattern=(0, 2, 1)),
    N.Dense(num_units=X_train.shape[1], activation=tf.nn.softmax),
    N.Reduce(fn=tf.reduce_mean, axis=1, keep_dims=True),
    N.Dimshuffle(pattern=(0, 2, 1))
], debug=True, name="Attention")

f_decoder = N.Sequence(ops=[
    N.CudnnRNN(num_units=64, rnn_mode='gru', num_layers=1,
               is_bidirectional=True),

    N.Flatten(outdim=2),
    N.Dense(num_units=1024, activation=K.linear),
    N.BatchNorm(axes=0, activation=K.relu),

    N.Dense(num_units=nb_classes, activation=K.linear),
], debug=True, name="Decoder")
# ====== attention ====== #
Z = f_encoder(X)
A = f_attn(Z)
y_pred_logits_train = f_decoder(Z * A, training=True)
y_pred_logits_pred = f_decoder(Z * A, training=False)
# ====== prediction ====== #
y_pred_probas = f_decoder(y_pred_logits_pred)
y_pred = tf.argmax(y_pred_probas, axis=-1)
# ===========================================================================
# Create objectives
# ===========================================================================
#loss
loss = tf.losses.softmax_cross_entropy(y, y_pred_logits_train)
acc = K.metrics.categorical_accuracy(y, y_pred_probas)
cm = K.metrics.confusion_matrix(y, y_pred_probas, labels=all_numbers)
train_op = K.optimizers.Adam(lr=LEARNING_RATE).minimize(loss)
K.initialize_all_variables()
# ====== create function ====== #
f_train = K.function(inputs=[X, y], outputs=[loss, acc, cm],
                     updates=train_op, training=True)
f_score = K.function(inputs=[X, y], outputs=[loss, acc, cm],
                     training=False)
f_pred = K.function(inputs=X, outputs=y_pred_probas, training=False)
# ===========================================================================
# Create trainer
# ===========================================================================
