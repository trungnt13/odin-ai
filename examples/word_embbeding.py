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
from six.moves import cPickle
from itertools import chain

import numpy as np

from odin import backend as K, nnet as N, fuel as F
from odin.preprocessing.text import (Tokenizer, language, POSfilter,
                                     TYPEfilter, CasePreprocessor,
                                     TransPreprocessor)
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
labels = []
texts = []
for i, j in newsgroup.items():
    labels += [i] * len(j)
    texts += j
    print(i, ':', len(j))
labels_set = list(set(labels))
nb_labels = len(labels_set)

tokenizer_path = get_modelpath('tokenizer', override=False)
if OVERRIDE and os.path.exists(tokenizer_path):
    os.remove(tokenizer_path)
if os.path.exists(tokenizer_path):
    print("Load saved tokenizer ...")
    tk = cPickle.load(open(tokenizer_path, 'r'))
else:
    print('\nRebuild tokenizer ...')
    tk = Tokenizer(nb_words=MAX_NB_WORDS, char_level=False,
                   stopwords=True, lemmatization=True,
                   preprocessors=[TransPreprocessor(),
                                  CasePreprocessor(lower=True, keep_name=False, split=' ')],
                   nb_threads=None, batch_size=1024 * 3,
                   order='word', engine='odin'
    )
    tk.fit(texts, vocabulary=None)
    cPickle.dump(tk, open(tokenizer_path, 'w'), protocol=cPickle.HIGHEST_PROTOCOL)
print('========== Summary ==========')
for i, j in tk.summary.items():
    print(i, ':', j)

# ===========================================================================
# Build dataset
# ===========================================================================
X = tk.transform(texts, mode='seq', maxlen=MAX_SEQ_LEN,
                 end_document=None, token_not_found='ignore')

y = [labels_set.index(i) for i in labels]
y = one_hot(np.array(y, dtype='int32'), nb_classes=nb_labels)

n = X.shape[0]
np.random.seed(1234)
idx = np.random.permutation(n)
X = X[idx]
y = y[idx]

X_train = X[:int(0.8 * n)]
y_train = y[:int(0.8 * n)]
X_valid = X[int(0.8 * n):]
y_valid = y[int(0.8 * n):]

print('X:', X.shape, 'y:', y.shape)
print('X_train:', X_train.shape, 'y_train:', y_train.shape)
print('X_valid:', X_valid.shape, 'y_valid:', y_valid.shape)

E = tk.embed(embedding)
# these numbers must be the same for all time
print('Tokenizer:', np.sum(E), np.sum(X_train), np.sum(y_train),
      np.sum(X_valid), np.sum(y_valid))
# ===========================================================================
# Building model
# ===========================================================================
X = K.placeholder(shape=(None, MAX_SEQ_LEN), dtype='int32', name='X')
y = K.placeholder(shape=(None, nb_labels), dtype='float32', name='y')

f = N.Sequence([
    N.Embedding(tk.nb_words, embedding_dims, W_init=E),
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

y_pred = f(X)
params = [p for p in f.parameters if not has_roles(p, EmbeddingWeight)]
print('Params:', [p.name for p in params])

cost_train = K.mean(K.categorical_crossentropy(y_pred, y))
cost_score = K.mean(K.categorical_accuracy(y_pred, y))

opt = K.optimizers.RMSProp()
updates = opt.get_updates(cost_train, params)

print('Build training function ...')
f_train = K.function([X, y], cost_train, updates)
print('Build scoring function ...')
f_score = K.function([X, y], cost_score)

trainer = training.MainLoop(batch_size=128, seed=1234, shuffle_level=2)
trainer.set_task(f_train, (X_train, y_train), epoch=args['epoch'], name='train')
trainer.set_subtask(f_score, (X_valid, y_valid), freq=1., name='valid')
trainer.set_callback([
    training.ProgressMonitor('train', format='Train:{:.4f}'),
    training.ProgressMonitor('valid', format='Test:{:.4f}'),
    training.History()
])
trainer.run()
