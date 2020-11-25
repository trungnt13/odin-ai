# ===========================================================================
# NORMALIZED 0-1:
#  - FLOAT16:
#    * test-ce:[ 1.62965763]
#    * test-acc:[ 0.41739649]
#  - FLOAT32:
#    * test-ce:[ 1.64211428]
#    * test-acc:[ 0.41013137]
# Gaussian NORMALIZED:
#  - FLOAT16:
#    * test-ce:[ 1.43444645]
#    * test-acc:[ 0.49900478]
#  - FLOAT32:
#    * test-ce:[ 1.42292392]
#    * test-acc:[ 0.49472532]
# => Gaussian normalized is better, and float16 is no different from float32
# ===========================================================================
from __future__ import print_function, division, absolute_import

import numpy as np

import os
os.environ['ODIN'] = 'float32,gpu,theano,seed=12,cnmem=0.4'

from odin import backend as K
from odin import nnet as N
from odin import fuel, training
from odin.utils import get_modelpath, ArgController, stdio, get_logpath
from six.moves import cPickle

stdio(get_logpath('tmp.log'))

# ===========================================================================
# Load data
# ===========================================================================
ds = fuel.load_cifar10()
print(ds)

X_train = K.placeholder(shape=(None,) + ds['X_train'].shape[1:], name='X_train')
X_score = K.placeholder(shape=(None,) + ds['X_train'].shape[1:], name='X_score')
y = K.placeholder(shape=(None,), name='y', dtype='int32')

# ===========================================================================
# Build network
# ===========================================================================
ops = N.Sequence([
    N.Flatten(outdim=2),
    N.Dense(512, activation=K.relu),
    N.Dense(256, activation=K.relu),
    N.Dense(10, activation=K.softmax)
])
ops = cPickle.loads(cPickle.dumps(ops)) # test if the ops is pickle-able

y_pred_train = ops(X_train)
y_pred_score = ops(X_score)
cost_train = K.mean(K.categorical_crossentropy(y_pred_train, y))
cost_test_1 = K.mean(K.categorical_crossentropy(y_pred_score, y))
cost_test_2 = K.mean(K.categorical_accuracy(y_pred_score, y))
cost_test_3 = K.confusion_matrix(y_pred_score, y, labels=range(10))

parameters = ops.parameters
optimizer = K.optimizers.RMSProp(lr= 0.0001, clipnorm=100.)
updates = optimizer(cost_train, parameters)
print('Building training functions ...')
f_train = K.function([X_train, y], [cost_train, optimizer.norm],
                     updates=updates)
print('Building testing functions ...')
f_test = K.function([X_score, y], [cost_test_1, cost_test_2, cost_test_3])

# ====== normalize 0-1 ====== #
if False:
    print('Normalized data in range [0-1]')
    X_train = ds['X_train'][:]
    X_train = (X_train - np.min(X_train, 0)) / (np.max(X_train) - np.min(X_train))
    X_test = ds['X_test'][:]
    X_test = (X_test - np.min(X_test, 0)) / (np.max(X_test) - np.min(X_test))
# ====== Gaussian normalize ====== #
else:
    print('Gaussian normalized the data')
    X_train = ds['X_train'][:]
    X_train = (X_train - np.mean(X_train, 0)) / (np.std(X_train))
    X_test = ds['X_test'][:]
    X_test = (X_test - np.mean(X_test, 0)) / (np.std(X_test))

datatype = 'float16'
X_train = X_train.astype(datatype)
X_test = X_test.astype(datatype)
print('Data type:', X_train.dtype, X_test.dtype)
# ===========================================================================
# Build trainer
# ===========================================================================
print('Start training ...')
task = training.MainLoop(batch_size=32, seed=12, shuffle_level=2)
task.set_checkpoint(get_modelpath(name='mnist.ai', override=True), ops)
task.set_task(f_train, (X_train, ds['y_train']), epoch=3, name='train')
task.set_subtask(f_test, (X_test, ds['y_test']), freq=0.6, name='valid')
task.set_subtask(f_test, (X_test, ds['y_test']), when=-1, name='test')
task.set_callback([
    training.ProgressMonitor(name='train', format='Loss:{:.4f}, Norm:{:.4f}'),
    training.ProgressMonitor(name='valid', format='Results: {:.4f}-{:.4f}',
                             tracking={2: lambda x: sum(x)}),
    training.ProgressMonitor(name='test', format='Results: {:.4f}-{:.4f}'),
    training.History(),
    training.EarlyStopGeneralizationLoss('valid', threshold=5, patience=3),
    training.NaNDetector(('train', 'valid'), patience=3, rollback=True)
])
task.run()

# ====== plot the training process ====== #
task['History'].print_info()
task['History'].print_batch('train')
task['History'].print_batch('valid')
task['History'].print_epoch('test')
