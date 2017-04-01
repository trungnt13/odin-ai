# ===========================================================================
# Using TIDIGITS dataset to predict gender (Boy, Girl, Woman, Man)
#
# ===========================================================================
from __future__ import print_function, division, absolute_import

import os
os.environ['ODIN'] = 'gpu,float32,tensorflow'

import numpy as np

from odin import backend as K, nnet as N, fuel as F
from odin.stats import train_valid_test_split
from odin import training

# ===========================================================================
# Load data
# Saved WAV file:
#     /audio/train_g_08_17_as_a_4291815
#     -------------------------------------
#     train material, child "g"irl, age is "08", dialect group is "17",
#     speaker code "as", "a" is first production,
#     digit sequence "4-2-9-1-8-1-5".
# ===========================================================================
PATH = '/home/trung/data/tidigits'
SEED = 12082518
ds = F.Dataset(PATH, read_only=True)
print(ds)
gender = list(set([i.split('_')[1] for i in ds['indices'].keys()]))
digits = [i.split('_')[-1] for i in ds['indices'].keys()]

# ===========================================================================
# SPlit dataset
# ===========================================================================
np.random.seed(SEED)
train = [(name, start, end) for name, (start, end) in ds['indices']
         if 'train_' in name]
# split by speaker ID
train, valid = train_valid_test_split(train, train=0.8,
    idfunc=lambda x: x[0].split('_')[4], inc_test=False, seed=12082518)
test = [(name, start, end) for name, (start, end) in ds['indices']
        if 'test_' in name]
print("#File train:", len(train))
print("#File valid:", len(valid))
print("#File test:", len(test))

recipes = [
    F.recipes.Name2Trans(
        converter_func=lambda f: gender.index(f.split('_')[1])),
    F.recipes.LabelOneHot(nb_classes=len(gender)),
    F.recipes.Sequencing(frame_length=256, hop_length=128,
        end='pad', endmode='post', endvalue=0)
]

feeder_train = F.Feeder(ds['mspec'], indices=train, ncpu=1)
feeder_valid = F.Feeder(ds['mspec'], indices=valid, ncpu=1)
feeder_test = F.Feeder(ds['mspec'], indices=test, ncpu=1)

feeder_train.set_recipes(recipes + [F.recipes.CreateBatch()])
feeder_valid.set_recipes(recipes + [F.recipes.CreateBatch()])
feeder_test.set_recipes(recipes + [F.recipes.CreateBatch()])

# ===========================================================================
# Create model
# ===========================================================================
X = K.placeholder(shape=(None,) + feeder_train.shape[1:], name='X')
y = K.placeholder(shape=(None, len(gender)), name='y')

f = N.Sequence([
    N.Dimshuffle(pattern=(0, 1, 2, 'x')),
    N.Conv(num_filters=32, filter_size=3, strides=1, b_init=None, pad='valid'),
    N.BatchNorm(activation=K.relu),
    N.Pool(pool_size=2, mode='avg'),

    N.Conv(num_filters=64, filter_size=3, strides=1, b_init=None, pad='valid'),
    N.BatchNorm(activation=K.relu),
    N.Pool(pool_size=2, mode='avg'),

    N.Flatten(outdim=3),
    N.Dense(num_units=512, b_init=None),
    N.BatchNorm(axes=(0, 1)),
    N.AutoRNN(num_units=128, rnn_mode='gru', num_layers=2,
              input_mode='linear', direction_mode='unidirectional'),

    N.Flatten(outdim=2),
    N.Dense(num_units=len(gender), activation=K.softmax)
], debug=True)

K.set_training(True); y_train = f(X)
K.set_training(False); y_score = f(X)
f_pred = K.function(inputs=X, outputs=y_score)


param = [p for p in f.parameters]
opt = K.optimizers.RMSProp(lr=0.00001)

# ===========================================================================
# Training
# ===========================================================================
train, hist = training.standard_trainer(
    train_data=feeder_train, valid_data=feeder_valid,
    X=X, y_train=y_train, y_score=y_score, y_target=y,
    cost_train=K.categorical_crossentropy,
    cost_score=[K.categorical_crossentropy, K.categorical_accuracy],
    parameters=param,
    batch_size=64, valid_freq=0.6,
    optimizer=opt, stop_callback=opt.get_lr_callback(),
    confusion_matrix=gender)
train.run()
