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
                            dtype='float32')
  cross_validation = (ds['X_valid'], ds['y_valid'])
  f.fit(X=ds['X_train'], y=ds['y_train'],
        cv=cross_validation)
else:
  with open(PATH, 'rb') as f:
    f = pickle.load(f)
# ===========================================================================
# Evaluation
# ===========================================================================
y_true = ds['y_test'][:]
y_pred = f.predict(ds['X_test'])

acc = accuracy_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)

print(ctext("======= Test set ========", 'cyan'))
print("Accuracy:", acc)
print("Confusion matrix:")
print(V.print_confusion(arr=cm))
