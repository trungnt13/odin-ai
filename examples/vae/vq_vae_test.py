from __future__ import absolute_import, division, print_function

import functools
import glob
import os
from collections import defaultdict

import numpy as np
import seaborn as sns
import tensorflow as tf
from matplotlib import pyplot as plt
from tqdm import tqdm

from odin.bay.vi.autoencoder import VQVAE, RandomVariable, VectorQuantizer
from odin.fuel import BinarizedMNIST
from odin.utils import ArgController

tf.config.experimental.set_memory_growth(
    tf.config.list_physical_devices('GPU')[0], True)
tf.debugging.set_log_device_placement(False)
tf.autograph.set_verbosity(0)
tf.random.set_seed(1)
np.random.seed(1)

args = ArgController(
).add("--override", "Override trained model", False \
).add("-niter", "Number of training iteration", 10000 \
).parse()

# ===========================================================================
# Config
# ===========================================================================
SAVE_PATH = "/tmp/vq_vae"
if not os.path.exists(SAVE_PATH):
  os.makedirs(SAVE_PATH)
MODEL_PATH = os.path.join(SAVE_PATH, "model")
if args.override:
  for f in sorted(
      glob.glob(f"{SAVE_PATH}/*.png") + glob.glob(f"{SAVE_PATH}/model*")):
    os.remove(f)
    print("Removed file:", f)

batch_size = 128
code_size = 16
n_codes = 64
commitment = 0.25
activation = "elu"
base_depth = 32
n_images = 10  # for visualization

# ===========================================================================
# Load dataset
# ===========================================================================
ds = BinarizedMNIST()
train = ds.create_dataset(batch_size=batch_size,
                          partition='train',
                          inc_labels=False)
test = ds.create_dataset(batch_size=batch_size,
                         partition='test',
                         inc_labels=False)
input_shape = tf.data.experimental.get_structure(train).shape[1:]

# ===========================================================================
# Create the layers
# ===========================================================================
conv = functools.partial(tf.keras.layers.Conv2D,
                         padding="SAME",
                         activation=activation)
deconv = functools.partial(tf.keras.layers.Conv2DTranspose,
                           padding="SAME",
                           activation=activation)
encoder = tf.keras.Sequential(
    [
        tf.keras.layers.InputLayer(input_shape=input_shape),
        conv(base_depth, 5, 1),
        conv(base_depth, 5, 2),
        conv(2 * base_depth, 5, 1),
        conv(2 * base_depth, 5, 2),
        conv(code_size, 7, padding="VALID"),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(code_size, activation=None),
        tf.keras.layers.Reshape([1, code_size])
    ],
    name="Encoder",
)
decoder = tf.keras.Sequential(
    [
        tf.keras.layers.InputLayer(input_shape=encoder.output_shape[1:]),
        tf.keras.layers.Reshape((1, 1, code_size)),
        deconv(2 * base_depth, 7, padding="VALID"),
        deconv(2 * base_depth, 5),
        deconv(2 * base_depth, 5, 2),
        deconv(base_depth, 5),
        deconv(base_depth, 5, 2),
        deconv(base_depth, 5),
        conv(input_shape[-1], 5, activation=None),
        tf.keras.layers.Flatten()
    ],
    name="Decoder",
)

# ===========================================================================
# Create the VAE
# ===========================================================================
vae = VQVAE(encoder=encoder,
            decoder=decoder,
            n_codes=n_codes,
            commitment_weight=commitment,
            decay=0.99,
            ema_perturb=1e-5,
            outputs=RandomVariable((28, 28, 1),
                                   posterior="bernoulli",
                                   projection=False,
                                   name="Image"),
            path=MODEL_PATH)
print(vae)
x = next(iter(test))[:n_images]
z = vae.sample_prior(n_images)


def show_image(img, ax):
  ax.imshow(img.numpy().squeeze(), interpolation="none", cmap="binary")
  ax.axis('off')


def callback(name=None):
  pX = vae.decode(z, training=False)
  pX_, _ = vae(x, training=False)
  x_z = pX.mean()
  x_ = pX.mean()
  step = int(vae.step)
  # show the figure
  nrow, ncol, dpi = 3, n_images, 180
  fig = plt.figure(figsize=(5 * n_images, 4 * 3), dpi=dpi)
  for i in range(n_images):
    ax1 = plt.subplot(nrow, ncol, i + 1)
    show_image(x[i], ax1)
    ax2 = plt.subplot(nrow, ncol, i + 1 + n_images)
    show_image(x_[i], ax2)
    ax3 = plt.subplot(nrow, ncol, i + 1 + n_images * 2)
    show_image(x_z[i], ax3)
    if i == 0:
      ax1.set_title("Original", fontsize=32)
      ax2.set_title("Reconstructed", fontsize=32)
      ax3.set_title("Sampled", fontsize=32)
  plt.suptitle(f"Step: {int(step)}", fontsize=48)
  plt.tight_layout(rect=[0.0, 0.02, 1.0, 0.98])
  path = os.path.join(SAVE_PATH,
                      f'step{step}.png' if name is None else f"{name}.png")
  fig.savefig(path, dpi=dpi)
  plt.close(fig)
  vae.plot_learning_curves(os.path.join(SAVE_PATH, "learning_curves.png"),
                           summary_steps=[100, 10])
  print("Saved figure to path:", path)


vae.fit(train,
        valid=test,
        valid_freq=500,
        max_iter=int(args.niter),
        optimizer='adam',
        learning_rate=0.001,
        callback=callback,
        compile_graph=True,
        checkpoint=MODEL_PATH,
        skip_fitted=True)
######## final evaluation
callback(name="final")
print("Counts:", [int(i) for i in vae.ema_counts.numpy()])
scores = defaultdict(list)
for x in tqdm(test.repeat(1)):
  pxz, qzx = vae(x, training=False)
  llk, kl = vae.elbo(x, pxz, qzx, return_components=True)
  for k, v in list(llk.items()) + list(kl.items()):
    scores[k].append(v)
for name, values in scores.items():
  values = tf.reduce_mean(tf.concat(values, axis=0))
  print(f"{name}: {values:.4f}")
