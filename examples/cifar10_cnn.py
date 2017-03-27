from __future__ import print_function, division, absolute_import

import os
os.environ['ODIN'] = 'float32,gpu,tensorflow,seed=12082518'

import numpy as np

from odin import fuel as F, nnet as N, backend as K, training, utils

# ===========================================================================
# Load dataset
# ===========================================================================
ds = F.load_cifar10()
print(ds)
X_learn = ds['X_train'][:].astype('float32') / 255.
y_learn = ds['y_train']
X_test = ds['X_test'][:].astype('float32') / 255.
y_test = ds['y_test']

# ===========================================================================
# Create network
# ===========================================================================
X = K.placeholder(shape=(None,) + X_learn.shape[1:], name='X')
y_true = K.placeholder(shape=(None,), name='y_true', dtype='int32')

f = N.Sequence([
    N.Dimshuffle(pattern=(0, 2, 3, 1)),
    N.Conv(32, (3, 3), pad='same', stride=(1, 1), activation=K.relu),
    N.Conv(32, (3, 3), pad='same', stride=(1, 1), activation=K.relu),
    N.Pool(pool_size=(2, 2), strides=None, mode='max'),
    N.Dropout(level=0.25),

    N.Conv(64, (3, 3), pad='same', stride=(1, 1), activation=K.relu),
    N.Conv(64, (3, 3), pad='same', stride=(1, 1), activation=K.relu),
    N.Pool(pool_size=(2, 2), strides=None, mode='max'),
    N.Dropout(level=0.25),

    N.Flatten(outdim=2),
    N.Dense(512, activation=K.relu),
    N.Dropout(level=0.5),
    N.Dense(10, activation=K.softmax)
], debug=True)
K.set_training(True); y_train = f(X)
K.set_training(False); y_pred = f(X)

cost_train = K.mean(K.categorical_crossentropy(y_train, y_true))
cost_pred = K.mean(K.categorical_accuracy(y_pred, y_true))
cost_eval = K.mean(K.categorical_crossentropy(y_pred, y_true))
parameters = f.parameters
print('Parameters:', [p.name for p in parameters])

optz = K.optimizers.RMSProp()
updates = optz.get_updates(cost_train, parameters)

print("Build training function ...")
f_train = K.function([X, y_true], cost_train, updates=updates)
print("Build scoring function ...")
f_score = K.function([X, y_true], [cost_pred, cost_eval])

# ===========================================================================
# Create trainer
# ===========================================================================
print("Create trainer ...")
trainer = training.MainLoop(batch_size=32, seed=12082518,
                            shuffle_level=2)
trainer.set_save(utils.get_modelpath('cifar10.ai', override=True), f)
trainer.set_task(f_train, [X_learn, y_learn], epoch=25, p=1, name='Train')
trainer.set_subtask(f_score, [X_test, y_test], freq=1, name='Valid')
trainer.set_callback([
    training.ProgressMonitor(name='Train', format='Results: {:.4f}'),
    training.ProgressMonitor(name='Valid', format='Results: {:.4f},{:.4f}'),
    # early stop based on crossentropy on test (not a right procedure,
    # but only for testing)
    training.EarlyStopGeneralizationLoss(name='Valid', threshold=5, patience=3,
            get_value=lambda x: np.mean([entropy for acc, entropy in x])),
    training.History()
])
trainer.run()

# ===========================================================================
# Evaluation and visualization
# ===========================================================================
# trainer['History'].print_epoch('Train')
# trainer['History'].print_epoch('Valid')
