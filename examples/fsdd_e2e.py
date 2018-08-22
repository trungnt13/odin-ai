from __future__ import print_function, division, absolute_import
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import os
os.environ['ODIN'] = 'float32,gpu,seed=5218'

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import init_ops

from odin.utils import ctext, Progbar, mpi, cache_disk, get_modelpath
from odin import fuel as F, nnet as N, backend as K, training as T, visual as V
from odin import preprocessing as pp
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
PAD_MODE = 'post'
LEARNING_RATE = 0.01
BATCH_SIZE = 32
NB_EPOCH = 12
NB_VISUAL_SAMPLES = 2
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
train_utt = [i for i in train_utt
             if i not in valid_utt]
test_utt = [file2name(f) for f in files
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
      pp.base.Converter(converter=file2name, input_name='path'),
      pp.speech.STFTExtractor(frame_length=FRAME_LENGTH, step_length=STEP_LENGTH,
                              window='hamm', n_fft=NFFT, energy=True),
      pp.speech.SADextractor(nb_mixture=3, smooth_window=3),
      # ====== spectrogram ====== #
      pp.speech.PowerSpecExtractor(power=2.0),
      pp.speech.MelsSpecExtractor(n_mels=NMELS, fmin=FMIN, fmax=FMAX, top_db=80.0),
      pp.speech.MFCCsExtractor(n_ceps=NCEPS, output_name='mfcc',
                               remove_first_coef=False),
      # ====== SDC features ====== #
      pp.speech.MFCCsExtractor(n_ceps=7, output_name='sdc',
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
  mask = []
  for f in flist:
    feat = features[f][FEATURE]
    ids = np.zeros(shape=(LONGEST_UTT,), dtype='float32')
    if PAD_MODE == 'pre':
      ids[-feat.shape[0]:] = 1
    else:
      ids[:feat.shape[0]] = 1
    assert ids.sum() == feat.shape[0]
    feat = segment_axis(feat,
                        frame_length=LONGEST_UTT, step_length=1,
                        end='pad', pad_value=0, pad_mode=PAD_MODE)
    label = int(f.split('_')[0])
    X.append(feat)
    y.append(label)
    mask.append(ids)
  X = np.concatenate(X, axis=0)
  y = np.array(y)
  mask = np.array(mask)
  indices = np.random.permutation(X.shape[0])
  return (X[indices],
          one_hot(y[indices], nb_classes=len(all_numbers)),
          mask[indices])
# ====== Generate data ====== #
X_train, y_train, mask_train = generate_data(train_utt)
X_valid, y_valid, mask_valid = generate_data(valid_utt)
X_test, y_test, mask_test = generate_data(test_utt)
input_ndim = X_train.shape[-1]
print('train:', X_train.shape, y_train.shape, mask_train.shape)
print('valid:', X_valid.shape, y_valid.shape, mask_valid.shape)
print('test:', X_test.shape, y_test.shape, mask_test.shape)
# ====== select some sample for visualization ====== #
idx = np.random.choice(a=range(X_train.shape[0]),
                       size=NB_VISUAL_SAMPLES,
                       replace=False)
X_train_visual = X_train[idx]
y_train_visual = y_train[idx]
mask_train_visual = mask_train[idx]

idx = np.random.choice(a=range(X_test.shape[0]),
                       size=NB_VISUAL_SAMPLES,
                       replace=False)
X_test_visual = X_test[idx]
y_test_visual = y_test[idx]
mask_test_visual = mask_test[idx]
# ===========================================================================
# Create the network
# ===========================================================================
# (num_samples, num_timestep, num_features)
INPUT_SHAPE = (None, X_train.shape[1], X_train.shape[2])
X = K.placeholder(shape=INPUT_SHAPE, name='X')
mask = K.placeholder(shape=(None,) + mask_train.shape[1:], name='mask')
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
# ====== different attention approach ====== #
f_attn = N.Sequence(ops=[
    N.Dimshuffle(pattern=(0, 2, 1)),
    N.Dense(num_units=X_train.shape[1], activation=tf.nn.softmax),
    N.Reduce(fn=tf.reduce_max, axis=1, keepdims=True),
    N.Dimshuffle(pattern=(0, 2, 1))
], debug=True, name="Attention")

# f_attn = N.Sequence(ops=[
#     N.Dense(num_units=1, activation=tf.nn.relu),
#     N.Flatten(outdim=2),
#     N.Activate(fn=tf.nn.softmax),
#     N.Dimshuffle(pattern=(0, 1, 'x'))
# ], debug=True, name="Attention")
# ====== the decoder (discriminator) ====== #
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
# Z = Z * tf.expand_dims(mask, axis=-1)
A = f_attn(Z)
Z_ = Z * A * tf.expand_dims(mask, axis=-1)
y_pred_logits_train = f_decoder(Z_, training=True)
y_pred_logits_pred = f_decoder(Z_, training=False)
# ====== prediction ====== #
y_pred_probas = tf.nn.softmax(y_pred_logits_pred)
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
f_train = K.function(inputs=[X, y, mask], outputs=[loss, acc, cm],
                     updates=train_op, training=True)
f_score = K.function(inputs=[X, y, mask], outputs=[loss, acc, cm],
                     training=False)
f_z = K.function(inputs=[X, mask], outputs=Z, training=False)
f_pred = K.function(inputs=[X, mask], outputs=[y_pred_probas, A],
                    training=False)
# ===========================================================================
# Training the network
# ===========================================================================
trainer = T.MainLoop(batch_size=BATCH_SIZE, seed=5218, shuffle_level=2,
                     allow_rollback=True, verbose=3)
trainer.set_checkpoint(get_modelpath(name='fsdd_ai', override=True),
                       obj=[f_encoder, f_attn, f_decoder])
trainer.set_callbacks([
    T.NaNDetector(),
    T.EarlyStopGeneralizationLoss('valid', loss, threshold=5, patience=3)
])
trainer.set_train_task(func=f_train,
                       data=(X_train, y_train, mask_train),
                       epoch=NB_EPOCH,
                       name='train')
trainer.set_valid_task(func=f_score,
                    data=(X_valid, y_valid, mask_valid),
                    freq=T.Timer(percentage=0.8),
                    name='valid')
trainer.set_eval_task(func=f_score,
                   data=(X_test, y_test, mask_test),
                   name='eval')
trainer.run()
# ===========================================================================
# Visualization
# ===========================================================================
def fast_plot(x, y, m, train_set):
  y = np.argmax(y)
  y_, a = f_pred(x[None, :, :], m[None, :])
  y_ = np.argmax(y_)
  a = a.ravel()
  assert a.shape[0] == x.shape[0]
  V.plot_multiple_features(
      features={'spec': x,
                'sad': m,
                'attention': a},
      title='[%s] %d and %d' % ('Train' if train_set else 'Test', y, y_))
for x, y, m in zip(X_train_visual, y_train_visual, mask_train_visual):
  fast_plot(x, y, m, train_set=True)
for x, y, m in zip(X_test_visual, y_test_visual, mask_test_visual):
  fast_plot(x, y, m, train_set=False)
V.plot_save('/tmp/tmp.pdf')
