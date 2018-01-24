import os
os.environ['ODIN'] = 'gpu,float32'

import numpy as np
from odin import ml
from odin import fuel as F
from odin.utils import get_datasetpath, unique_labels, ctext

# ===========================================================================
# PATH
# ===========================================================================
PATH = get_datasetpath('digits', override=False)

# ===========================================================================
# Const
# ===========================================================================
ds = F.Dataset(PATH, read_only=True)
fn_label, labels = unique_labels(y=list(ds['indices'].keys()),
                                 return_labels=True,
                                 key_func=lambda x: x.split('_')[1])
X = ds['mspec']
y = []
indices = sorted(ds['indices'], key=lambda x: x[1][0])
for name, (start, end) in indices:
  y += [fn_label(name)] * (end - start)
y = np.array(y, dtype='int32')
assert len(X) == len(y)
print('#Labels:', ctext(labels, 'yellow'))
# ===========================================================================
# Model
# ===========================================================================
f = ml.LogisticRegression(nb_classes=len(labels), tol=1e-2)
f.fit(X, y)
