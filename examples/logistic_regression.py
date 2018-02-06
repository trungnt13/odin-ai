import matplotlib
matplotlib.use('Agg')

import os
os.environ['ODIN'] = 'gpu,float32'
import pickle

import numpy as np

from odin import ml
from odin import fuel as F
from odin.utils import ctext, ArgController
from odin import visual as V

from sklearn.metrics import confusion_matrix, accuracy_score
args = ArgController(
).add('--reset', "re-run the fitting of the model", False
).parse()
# ===========================================================================
# Const
# ===========================================================================
ds = F.MNIST.load()
print(ds)
nb_classes = 10
PATH = '/tmp/lore.ai'
# ===========================================================================
# Model
# ===========================================================================
if not os.path.exists(PATH) or args.reset:
  f = ml.LogisticRegression(nb_classes=nb_classes, tol=1e-4,
                            fit_intercept=True, path=PATH,
                            batch_size=256, dtype='float32')
  cross_validation = (ds['X_valid'], ds['y_valid'])
  f.fit(X=ds['X_train'], y=ds['y_train'],
        cv=cross_validation)
else:
  with open(PATH, 'rb') as f:
    f = pickle.load(f)
# ===========================================================================
# Evaluation
# ===========================================================================
f.evaluate(ds['X_test'], ds['y_test'], path='/tmp/tmp.pdf',
           title="MNIST Test Set",
           xlims=(0., 0.88), ylims=(0., 0.88))
