from __future__ import print_function, division, absolute_import
import matplotlib
matplotlib.use('Agg')

import os
os.environ['ODIN'] = 'gpu,float32,seed=12082518'
from collections import OrderedDict

import numpy as np
import tensorflow as tf

from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, accuracy_score

from odin import backend as K, nnet as N, fuel as F
from odin.stats import train_valid_test_split, freqcount
from odin import training
from odin import preprocessing as pp
from odin.ml import evaluate, fast_tsne
from odin.visual import (print_dist, print_confusion, print_hist,
                         plot_scatter, plot_figure, plot_spectrogram, plot_save,
                         plot_confusion_matrix,
                         generate_random_colors, generate_random_marker)
from odin.utils import (get_logpath, get_modelpath, get_datasetpath, get_figpath,
                        Progbar, unique_labels, chain,
                        as_tuple_of_shape, stdio, ctext, ArgController)
# ===========================================================================
# Const
# ===========================================================================
FEAT = ['mspec', 'sad']
MODEL_PATH = get_modelpath(name='DIGITS', override=True)
LOG_PATH = get_logpath(name='digits.log', override=True)
FIG_PATH = get_figpath(name='DIGITS', override=True)
stdio(LOG_PATH)

DEBUG = False
# ====== trainign ====== #
BATCH_SIZE = 32
NB_EPOCH = 20
NB_SAMPLES = 8
VALID_PERCENTAGE = 0.4
# ===========================================================================
# Load dataset
# ===========================================================================
path = get_datasetpath(name='TIDIGITS_feats', override=False)
assert os.path.isdir(path), \
    "Cannot find preprocessed feature at: %s, try to run 'odin/examples/features.py'" % path
ds = F.Dataset(path, read_only=True)
assert all(f in ds for f in FEAT), "Cannot find features with name: %s" % FEAT
# ====== get all the indices of single digit ====== #
indices = [(name, (s, e))
           for name, (s, e) in list(ds['indices'].items())
           if len(name.split('_')[-1]) == 1]
K.get_rng().shuffle(indices)
print("Found %s utterances of single digit" % ctext(len(indices), 'cyan'))
# ===========================================================================
# Load and visual the dataset
# ===========================================================================
train = []
test = []
max_length = 0
min_length = np.inf
for name, (start, end) in indices:
  assert end - start > 0
  if name.split('_')[0] == 'train':
    train.append((name, (start, end)))
  else:
    test.append((name, (start, end)))
  max_length = max(end - start, max_length)
  min_length = min(end - start, min_length)
print(ctext("#Train:", 'yellow'), len(train), train[:2])
print(ctext("#Test:", 'yellow'), len(test), test[:2])
print("Min Length:", ctext(min_length, 'cyan'))
print("Max Length:", ctext(max_length, 'cyan'))
# ====== gender and single digit distribution ====== #
gender_digit = lambda x: x[0].split('_')[1] + '-' + x[0].split('_')[-1]
print(print_dist(d=freqcount(train, key=gender_digit),
                 show_number=True,
                 title="Training distribution"))
print(print_dist(d=freqcount(test, key=gender_digit),
                 show_number=True,
                 title="Testing distribution"))
# ====== digits ====== #
f_digits, digits = unique_labels(
    [i[0] for i in train + test],
    key_func=lambda x: x.split('_')[-1], return_labels=True)
print(ctext("All digits:", 'yellow'), ctext(digits, 'cyan'))
# ====== genders ====== #
f_genders, genders = unique_labels(
    [i[0] for i in train + test],
    key_func=lambda x: x.split('_')[1], return_labels=True)
print(ctext("All genders:", 'yellow'), ctext(genders, 'cyan'))
# ====== marker and color for visualization ====== #
digit_colors = generate_random_colors(n=len(digits))
digit_color_map = {i: j for i, j in zip(digits, digit_colors)}

gender_markers = generate_random_marker(n=len(genders))
gender_marker_map = {i: j for i, j in zip(genders, gender_markers)}

legends = OrderedDict()
for g, m in zip(genders, gender_markers):
  for d, c in zip(digits, digit_colors):
    legends[(c, m)] = str(g) + '-' + str(d)
# ===========================================================================
# SPlit dataset
# ===========================================================================
split_spkID = lambda x: x[0].split('_')[4]
split_dialID_spkID = lambda x: x[0].split('_')[3] + x[0].split('_')[4]
split_genID_spkID = lambda x: x[0].split('_')[1] + x[0].split('_')[4]
split_genID = lambda x: x[0].split('_')[1]
split_ageID = lambda x: x[0].split('_')[2]
# stratified sampling for each digit, splited based on speaker ID
train, valid = train_valid_test_split(x=train, train=0.6, inc_test=False,
    idfunc=split_spkID,
    seed=K.get_rng().randint(0, 10e8))
# make sure both train and valid set have all the numbers
assert set(i[0].split('_')[-1] for i in train) == set(digits)
assert set(i[0].split('_')[-1] for i in valid) == set(digits)
# ====== report ====== #
report_info = lambda idx, flist: sorted(list(set(i[0].split('_')[idx] for i in flist)))
print(ctext("#File train:", 'yellow'), len(train), train[:2])
print(' * Genders:', ctext(report_info(1, train), 'cyan'))
print(' * Age:', ctext(report_info(2, train), 'cyan'))
print(' * Dialects:', ctext(report_info(3, train), 'cyan'))
print(' * Speakers:', ctext(report_info(4, train), 'cyan'))
print(ctext("#File valid:", 'yellow'), len(valid), valid[:2])
print(' * Genders:', ctext(report_info(1, valid), 'cyan'))
print(' * Age:', ctext(report_info(2, valid), 'cyan'))
print(' * Dialects:', ctext(report_info(3, valid), 'cyan'))
print(' * Speakers:', ctext(report_info(4, valid), 'cyan'))
print(ctext("#File test:", 'yellow'), len(test), test[:2])
# ====== create recipe ====== #
recipes = [
    F.recipes.Slice(slices=slice(40), axis=-1, data_idx=0),
    F.recipes.Sequencing(frame_length=max_length, step_length=1,
                         end='pad', pad_mode='post', pad_value=0,
                         data_idx=None),
    F.recipes.Name2Label(converter_func=f_digits),
    F.recipes.LabelOneHot(nb_classes=len(digits), data_idx=-1),
]
data = [ds[f] for f in FEAT]
train = F.Feeder(F.IndexedData(data=data, indices=train),
                 dtype='float32', ncpu=6,
                 buffer_size=len(digits),
                 batch_mode='batch')
valid = F.Feeder(F.IndexedData(data=data, indices=valid),
                 dtype='float32', ncpu=2,
                 buffer_size=len(digits),
                 batch_mode='batch')
test = F.Feeder(F.IndexedData(data=data, indices=test),
                dtype='float32', ncpu=1,
                buffer_size=1,
                batch_mode='file')
train.set_recipes(recipes)
valid.set_recipes(recipes)
test.set_recipes(recipes)
# ===========================================================================
# Create model
# ===========================================================================
inputs = [K.placeholder(shape=(None,) + shape[1:], dtype='float32', name='input%d' % i)
          for i, shape in enumerate(train.shape)]
print("Inputs:", ctext(inputs, 'cyan'))
# ====== create the network ====== #
f_encoder = N.Sequence([
    N.Dimshuffle(pattern=(0, 1, 2, 'x')),
    N.Conv(num_filters=32, filter_size=(7, 7), b_init=None, activation=K.linear),
    N.BatchNorm(),
    N.Pool(pool_size=(3, 2), strides=2),
], debug=True, name='Encoder')
f_latent = N.Sequence([
    N.Flatten(outdim=3),
    N.CudnnRNN(num_units=128, num_layers=1, is_bidirectional=False,
               rnn_mode='lstm'),
], debug=True, name='Latent')
f_decoder = N.Sequence([
    N.Flatten(outdim=2),
    N.Dense(num_units=1024, b_init=None, activation=K.linear),
    N.BatchNorm(axes=0, activation=K.relu)
], debug=True, name='Decoder')
f_output = N.Sequence([
    N.Dense(len(digits), activation=K.linear)
], debug=True, name='Output')
# ====== applying ====== #
E = f_encoder(inputs[0])

Z_train = f_latent(E, training=True)
Z_infer = f_latent(E, training=False)

D_train = f_decoder(Z_train)
D_infer = f_decoder(Z_infer)

y_logit = f_output(D_train)
y_prob = tf.nn.softmax(f_output(D_infer))
# ====== create loss ====== #
y = inputs[-1]
ce = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=y_logit)
acc = K.metrics.categorical_accuracy(y_true=y, y_pred=y_prob)
cm = K.metrics.confusion_matrix(y_pred=y_prob, y_true=y, labels=len(digits))
# ====== params and optimizing ====== #
updates = K.optimizers.RMSProp(lr=0.0001).minimize(
    loss=ce, roles=[K.role.Weight, K.role.Bias])
K.initialize_all_variables()
# ====== Functions ====== #
print('Building training functions ...')
f_train = K.function(inputs=inputs, outputs=[ce, acc, cm],
                     updates=updates, training=True)
print('Building testing functions ...')
f_test = K.function(inputs=inputs, outputs=[ce, acc, cm],
                    training=False)
print('Building predicting functions ...')
f_pred = K.function(inputs=inputs, outputs=y_prob, training=False)
print("Building other functions ...")
f_e = K.function(inputs=inputs, outputs=E, training=False)
f_z = K.function(inputs=inputs, outputs=Z_infer, training=False)
f_d = K.function(inputs=inputs, outputs=D_infer, training=False)
# ===========================================================================
# Training
# ===========================================================================
print('Start training ...')
task = training.MainLoop(batch_size=BATCH_SIZE,
                         seed=120825,
                         shuffle_level=2,
                         allow_rollback=True,
                         labels=digits)
task.set_checkpoint(MODEL_PATH, [f_encoder, f_decoder])
task.set_callbacks([
    training.NaNDetector(),
    training.EarlyStopGeneralizationLoss('valid', ce,
                                         threshold=5, patience=5)
])
task.set_train_task(f_train, train, epoch=25, name='train')
task.set_valid_task(f_test, valid,
                    freq=training.Timer(percentage=0.8),
                    name='valid')
task.run()
# ===========================================================================
# Latent space
# ===========================================================================
def evaluate_latent(fn, feeder, title):
  y_true = []
  Z = []
  for outputs in Progbar(feeder.set_batch(batch_mode='file'),
                         name=title,
                         print_report=True,
                         print_summary=False,
                         count_func=lambda x: x[-1].shape[0]):
    name = str(outputs[0])
    idx = int(outputs[1])
    data = outputs[2:]
    assert idx == 0
    y_true.append(name)
    Z.append(fn(*data))
  Z = np.concatenate(Z, axis=0)
  # ====== visualize spectrogram ====== #
  if Z.ndim >= 3:
    sample = np.random.choice(range(len(Z)), size=3, replace=False)
    spec = Z[sample.astype('int32')]
    y = [y_true[int(i)] for i in sample]
    plot_figure(nrow=6, ncol=6)
    for i, (s, tit) in enumerate(zip(spec, y)):
      s = s.reshape(len(s), -1)
      plot_spectrogram(s.T, ax=(1, 3, i + 1), title=tit)
  # ====== visualize each point ====== #
  # flattent to 2D
  Z = np.reshape(Z, newshape=(len(Z), -1))
  # tsne if necessary
  if Z.shape[-1] > 3:
    Z = fast_tsne(Z, n_components=3, n_jobs=8,
                  random_state=K.get_rng().randint(0, 10e8))
  # color and marker
  Z_color = [digit_color_map[i.split('_')[-1]] for i in y_true]
  Z_marker = [gender_marker_map[i.split('_')[1]] for i in y_true]
  plot_figure(nrow=6, ncol=20)
  for i, azim in enumerate((15, 60, 120)):
    plot_scatter(x=Z[:, 0], y=Z[:, 1], z=Z[:, 2], ax=(1, 3, i + 1),
                 size=4, color=Z_color, marker=Z_marker, azim=azim,
                 legend=legends if i == 1 else None, legend_ncol=11, fontsize=10,
                 title=title)
  plot_save(os.path.join(FIG_PATH, '%s.pdf' % title))
# ====== differnt latent space ====== #
# encoder
evaluate_latent(f_e, valid, title="valid_encoder")
evaluate_latent(f_e, test, title="test_encoder")
# RNN latent
evaluate_latent(f_z, valid, title="valid_latent")
evaluate_latent(f_z, test, title="test_latent")
# Discriminator
evaluate_latent(f_d, valid, title="valid_decoder")
evaluate_latent(f_d, test, title="test_decoder")
# ===========================================================================
# Prediction
# ===========================================================================
def evaluate_feeder(feeder, title):
  y_true_digit = []
  y_true_gender = []
  y_pred = []
  for outputs in Progbar(feeder.set_batch(batch_mode='file'),
                         name=title,
                         print_report=True,
                         print_summary=False,
                         count_func=lambda x: x[-1].shape[0]):
    name = str(outputs[0])
    idx = int(outputs[1])
    data = outputs[2:]
    assert idx == 0
    y_true_digit.append(f_digits(name))
    y_true_gender.append(f_genders(name))
    y_pred.append(f_pred(*data))
  # ====== post processing ====== #
  y_true_digit = np.array(y_true_digit, dtype='int32')
  y_true_gender = np.array(y_true_gender, dtype='int32')
  y_pred_proba = np.concatenate(y_pred, axis=0)
  y_pred_all = np.argmax(y_pred_proba, axis=-1).astype('int32')
  # ====== plotting for each gender ====== #
  plot_figure(nrow=6, ncol=25)
  for gen in range(len(genders)):
    y_true, y_pred = [], []
    for i, g in enumerate(y_true_gender):
      if g == gen:
        y_true.append(y_true_digit[i])
        y_pred.append(y_pred_all[i])
    if len(y_true) == 0:
      continue
    cm = confusion_matrix(y_true, y_pred, labels=range(len(digits)))
    plot_confusion_matrix(cm, labels=digits, fontsize=8,
                          ax=(1, 4, gen + 1),
                          title='[%s]%s' % (genders[gen], title))
  plot_save(os.path.join(FIG_PATH, '%s.pdf' % title))
evaluate_feeder(valid, title="valid")
evaluate_feeder(test, title="test")
# ===========================================================================
# print some log
# ===========================================================================
print("Log path:", ctext(LOG_PATH, 'cyan'))
print("Fig path:", ctext(FIG_PATH, 'cyan'))
