import glob
import io
import os
import pickle
import shutil
from collections import Counter
from functools import partial

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from scipy import sparse
from tensorflow.python import keras

from odin import visual as vs
from odin.bay.vi.autoencoder import AmortizedLDA
from odin.bay.vi.metrics import unsupervised_clustering_scores
from odin.fuel import MNIST, PBMC, Cortex, HumanEmbryos, Newsgroup5, Newsgroup20
from odin.ml import fast_lda_topics, get_topics_string
from odin.utils import ArgController

args = ArgController(
).add('-ds', 'cortex, embryo, pbmc5k, pbmc10k, news5, news20, mnist', 'cortex' \
).add('-warmup', 'warmup iteration', 30000 \
).add('-niter', 'max iteration', 35000 \
).add('-post', 'posterior distribution', "dirichlet" \
).add('-dist', 'output distribution', "categorical" \
).add('--log', 'log norm the input data', False \
).add('--override', 'override saved results', False \
).add('--lda', 'training LDA model', False \
).parse()

# python amortized_lda_test.py --log -post dirichlet -dist categorical -ds news5
# python amortized_lda_test.py --log -post dirichlet -dist negativebinomial -ds news5
# python amortized_lda_test.py --log -post dirichlet -dist zinb -ds news5
# python amortized_lda_test.py -post dirichlet -dist categorical -ds news5
# python amortized_lda_test.py -post dirichlet -dist negativebinomial -ds news5
# python amortized_lda_test.py -post dirichlet -dist zinb -ds news5
# python amortized_lda_test.py --log -post gaussian -dist categorical -ds news5
# python amortized_lda_test.py --log -post gaussian -dist negativebinomial -ds news5
# python amortized_lda_test.py --log -post gaussian -dist zinb -ds news5
# python amortized_lda_test.py -post gaussian -dist categorical -ds news5
# python amortized_lda_test.py -post gaussian -dist negativebinomial -ds news5
# python amortized_lda_test.py -post gaussian -dist zinb -ds news5

# ===========================================================================
# Config
# ===========================================================================
warmup = int(args.warmup)
max_iter = int(args.niter)
posterior = str(args.post)
distribution = str(args.dist)
valid_freq = 2000
lognorm = bool(args.log)
batch_size = 64

PATH = "/tmp/amortized_lda"
if not os.path.exists(PATH):
  os.makedirs(PATH)
if args.lda:
  LOGDIR = f"{PATH}/{args.ds}_lda"
else:
  LOGDIR = f"{PATH}/{args.ds}_{posterior}_{distribution}_{'log' if lognorm else 'raw'}_{warmup}_{max_iter}"

if args.override:
  for path in glob.glob(f"{LOGDIR}*"):
    if os.path.isfile(path):
      os.remove(path)
      print("Remove file:", path)
    elif os.path.isdir(path):
      shutil.rmtree(path)
      print("Remove dir:", path)
print("Logdir:", LOGDIR)
# ===========================================================================
# Data
# ===========================================================================
if args.ds == "cortex":
  sc = Cortex()
elif args.ds == "embryo":
  sc = HumanEmbryos()
elif args.ds == "pbmc5k":
  sc = PBMC('5k')
elif args.ds == "pbmc10k":
  sc = PBMC('10k')
elif args.ds == "news20":
  sc = Newsgroup20()
elif args.ds == "news5":
  sc = Newsgroup5()
elif args.ds == "mnist":
  raise NotImplementedError
  sc = MNIST()
else:
  raise NotImplementedError(args.ds)
shape = sc.shape
train = sc.create_dataset(batch_size=batch_size, partition='train')
valid = sc.create_dataset(batch_size=batch_size, partition='valid')
test = sc.create_dataset(batch_size=batch_size, partition='test')

# concat both valid and test to final evaluation set
X_test = []
y_test = []
for x, y in sc.create_dataset(batch_size=batch_size,
                              partition='valid',
                              inc_labels=True).concatenate(
                                  sc.create_dataset(batch_size=batch_size,
                                                    partition='test',
                                                    inc_labels=True)):
  X_test.append(x)
  y_test.append(y)
X_test = tf.concat(X_test, axis=0)
y_test = tf.concat(y_test, axis=0)

try:
  from odin.ml import fast_umap
  x_ = fast_umap(X_test.numpy())
  algo = "umap"
except:
  from odin.ml import fast_tsne
  x_ = fast_tsne(X_test.numpy())
  algo = "tsne"


# ===========================================================================
# MOdel
# ===========================================================================
def fig2image(fig: plt.Figure, dpi=180) -> tf.Tensor:
  r""" Return an image shape `[1, h, w, 4]` """
  buf = io.BytesIO()
  plt.savefig(buf, format='png', dpi=dpi)
  buf.seek(0)
  image = tf.image.decode_png(buf.getvalue(), channels=4)
  # add batch dimension
  image = tf.expand_dims(image, 0)
  return image


def callback(vae: AmortizedLDA):
  perp = vae.perplexity(test, verbose=False)
  topics = vae.get_topics_string(vocabulary=sc.vocabulary, n_topics=sc.n_labels)
  print(f"Perplexity: {perp:.2f}")
  print(topics)
  tf.summary.text("topics", topics)
  tf.summary.scalar("perplexity", perp)


### LDA with EM
if args.lda:
  total_samples = sum(x.shape[0] for x in train.repeat(1))
  if os.path.exists(LOGDIR):
    with open(LOGDIR, 'rb') as f:
      lda = pickle.load(f)
  else:
    lda = fast_lda_topics(train,
                          n_components=sc.n_labels,
                          max_iter=1000,
                          total_samples=total_samples)
    with open(LOGDIR, 'wb') as f:
      pickle.dump(lda, f)
  topics_text = get_topics_string(lda,
                                  vocabulary=sc.vocabulary,
                                  n_topics=sc.n_labels,
                                  n_words=20)
### LDA with amortized inference
else:
  layers = [keras.Input(shape=shape)]
  if lognorm:
    layers.append(keras.layers.Lambda(lambda x: tf.math.log1p(x)))
  layers += [
      keras.layers.Dense(300, activation='relu'),
      keras.layers.Dense(300, activation='relu'),
      keras.layers.Dense(300, activation='relu'),
  ]
  encoder = keras.Sequential(
      layers,
      name="Encoder",
  )
  lda = AmortizedLDA(n_words=shape[-1],
                     n_topics=sc.n_labels,
                     lda_posterior=posterior,
                     word_distribution=distribution,
                     prior_warmup=warmup,
                     encoder=encoder,
                     path=LOGDIR)
  lda.fit(train,
          valid,
          optimizer='adam',
          learning_rate=3e-4,
          valid_freq=valid_freq,
          max_iter=max_iter,
          compile_graph=True,
          callback=partial(callback, vae=lda),
          skip_fitted=True,
          logdir=LOGDIR)
  lda.save_weights()
  topics_text = lda.get_topics_string(vocabulary=sc.vocabulary,
                                      n_topics=sc.n_labels,
                                      n_words=20)

# ===========================================================================
# Visualize
# ===========================================================================
with open(f"{LOGDIR}.txt", "w") as f:
  for topic in topics_text:
    f.write(topic + '\n')

if args.lda:
  y_pred = np.argmax(lda.transform(X_test), axis=-1)
else:
  y_pred = lda.predict_topics(X_test, hard_topics=True, verbose=False)
counts = Counter(y_pred)
y_pred_labels = np.array([f"#{i}({counts[i]})" for i in y_pred])

y_true = np.argmax(y_test, axis=-1)
counts = Counter(y_true)
y_true_labels = np.array([f"{sc.labels[i]}({counts[i]})" for i in y_true])

scores_text = ", ".join([
    f"{key}:{val:.2f}" for key, val in unsupervised_clustering_scores(
        factors=y_true, predictions=y_pred).items()
])

fig = plt.figure(figsize=(14, 8))
vs.plot_scatter(x=x_,
                color=y_pred_labels,
                size=12.0,
                ax=(1, 2, 1),
                title="Topics")
vs.plot_scatter(x=x_,
                color=y_true_labels,
                size=12.0,
                ax=(1, 2, 2),
                title="Celltype")
plt.suptitle(f"[{algo}]{os.path.basename(LOGDIR)}\n{scores_text}")
plt.tight_layout(rect=[0.05, 0.0, 1.0, 0.95])
fig.savefig(f"{LOGDIR}.png", dpi=200)
print("Saved image:", f"{LOGDIR}.png")
