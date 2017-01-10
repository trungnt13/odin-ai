from __future__ import print_function, division, absolute_import

from odin.utils import ArgController, get_all_files, pad_sequences, get_modelpath, one_hot

args = ArgController(
).add('-dev', 'device: gpu or cpu', 'gpu'
).add('-bk', 'backend: tensorflow or theano', 'tensorflow'
).add('-nclass', 'number of classes in newsgroup dataset will be used', 20
).add('-lr', 'learning rate', 0.0001
).add('-epoch', 'number of epoch', 3
).add('--rebuild', 'rebuild the tokenizer or not', False
).parse()

import os
os.environ['ODIN'] = 'float32,%s,%s' % (args['dev'], args['bk'])
import cPickle
from itertools import chain

import numpy as np

from odin import backend as K, nnet as N, fuel as F
from odin.basic import has_roles, EMBEDDING
from odin.preprocessing.text import (Tokenizer, language, POSfilter, TYPEfilter)
from odin import training

# ===========================================================================
# Const
# ===========================================================================
embedding_dims = 100
MAX_SEQ_LEN = 1000
MAX_NB_WORDS = 20000
OVERRIDE = args['rebuild']
MODEL_PATH = get_modelpath(name='word_embedding_test.ai',
                           override=True)
# ===========================================================================
# Load embedding
# ===========================================================================
embedding = F.load_glove(ndim=embedding_dims)
newsgroup = F.load_20newsgroup()
labels = newsgroup.keys()
nb_labels = len(labels)
for i, j in newsgroup.iteritems():
    print(i, ':', len(j))

tokenizer_path = get_modelpath('tokenizer', override=False)
if OVERRIDE and os.path.exists(tokenizer_path):
    os.remove(tokenizer_path)
if os.path.exists(tokenizer_path):
    print("Load saved tokenizer ...")
    tk = cPickle.load(open(tokenizer_path, 'r'))
else:
    print('\nRebuild tokenizer ...')
    tk = Tokenizer(nb_words=MAX_NB_WORDS, char_level=False,
                   stopwords=False, lemmatization=True, lower=True,
                   filters=[TYPEfilter(is_alpha=True, is_digit=True),
                            POSfilter(NOUN=True, PROPN=True,
                                      ADJ=True, VERB=True, ADV=True)],
                   nb_threads=None, batch_size=1024 * 3,
                   engine='odin'
    )
    tk.fit(chain(*newsgroup.values()), vocabulary=embedding)
    cPickle.dump(tk, open(tokenizer_path, 'w'), protocol=cPickle.HIGHEST_PROTOCOL)
print('Top popular words:', tk.top_k(n=12))
print('Length of dictionary:', len(tk.dictionary), tk.nb_words)

# ===========================================================================
# Build dataset
# ===========================================================================
X = tk.transform(chain(*newsgroup.values()), maxlen=MAX_SEQ_LEN, end_document=None,
                 token_not_found='raise')
y = []
for i in newsgroup.keys():
    y += [labels.index(i)] * len(newsgroup[i])
y = one_hot(np.array(y, dtype='int32'), n_classes=nb_labels)

n = X.shape[0]
idx = np.random.permutation(n)
X = X[idx]
y = y[idx]

X_train = X[:int(0.6 * n)]
y_train = y[:int(0.6 * n)]

X_valid = X[int(0.6 * n):int(0.8 * n)]
y_valid = y[int(0.6 * n):int(0.8 * n)]

X_test = X[int(0.8 * n):]
y_test = y[int(0.8 * n):]

print('X:', X.shape, 'y:', y.shape)
print('X_train:', X_train.shape, 'y_train:', y_train.shape)
print('X_valid:', X_valid.shape, 'y_valid:', y_valid.shape)
print('X_test:', X_test.shape, 'y_test:', y_test.shape)

# ===========================================================================
# Building model
# ===========================================================================
X_in = K.placeholder(shape=(None, MAX_SEQ_LEN), dtype='int32', name='X_in')
y_in = K.placeholder(shape=(None, nb_labels), dtype='float32', name='y_in')

f = N.Sequence([
    N.Embedding(tk.nb_words, embedding_dims,
                W_init=tk.embed(embedding)),
    N.Dimshuffle(pattern=(0, 1, 'x', 2)),

    N.Conv(num_filters=128, filter_size=(5, 1), strides=1, pad='valid',
           activation=K.relu),
    N.Pool(pool_size=(5, 1), pad='valid', mode='max'),

    N.Conv(num_filters=128, filter_size=(5, 1), strides=1, pad='valid',
           activation=K.relu),
    N.Pool(pool_size=(5, 1), pad='valid', mode='max'),

    N.Conv(num_filters=128, filter_size=(5, 1), strides=1, pad='valid',
           activation=K.relu),
    N.Pool(pool_size=(35, 1), pad='valid', mode='max'),

    N.Flatten(outdim=2),
    N.Dense(num_units=128, activation=K.relu),
    N.Dense(num_units=nb_labels, activation=K.softmax)
], debug=True)

y_out = f(X_in)
params = [p for p in f.parameters if not has_roles(p, EMBEDDING)]
print('Parameters:', [p.name for p in params])

cost_train = K.mean(K.categorical_crossentropy(y_out, y_in))
cost_score = K.mean(K.categorical_accuracy(y_out, y_in))

optimizer = K.optimizers.RMSProp(lr=args['lr'])
updates = optimizer.get_updates(cost_train, params)

print('Build training function ...')
f_train = K.function([X_in, y_in], cost_train, updates)
print('Build scoring function ...')
f_score = K.function([X_in, y_in], cost_score)

# ===========================================================================
# Create trainer
# ===========================================================================
print('Start training ...')
task = training.MainLoop(batch_size=128, seed=1208, shuffle_level=2)
task.set_save(MODEL_PATH, f)
task.set_task(f_train, (X_train, y_train), epoch=args['epoch'], name='train')
task.set_subtask(f_score, (X_valid, y_valid), freq=0.6, name='valid')
task.set_subtask(f_score, (X_test, y_test), when=-1, name='test')
task.set_callback([
    training.ProgressMonitor(name='train', format='Results: {:.4f}'),
    training.ProgressMonitor(name='valid', format='Results: {:.4f}'),
    training.ProgressMonitor(name='test', format='Results: {:.4f}'),
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
print('Benchmark TRAIN-batch:', task['History'].benchmark('train', 'batch_end').mean)
print('Benchmark TRAIN-epoch:', task['History'].benchmark('train', 'epoch_end').mean)
print('Benchmark PRED-batch:', task['History'].benchmark('valid', 'batch_end').mean)
print('Benchmark PRED-epoch:', task['History'].benchmark('valid', 'epoch_end').mean)
