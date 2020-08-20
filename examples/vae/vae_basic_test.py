from __future__ import absolute_import, division, print_function

import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp

from odin.bay.vi import RandomVariable, VariationalAutoencoder
from odin.utils import ArgController

tfk = tf.keras
tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions
tf.random.set_seed(1)
tf.config.experimental.set_memory_growth(
    tf.config.list_physical_devices('GPU')[0], True)
tf.debugging.set_log_device_placement(False)
tf.autograph.set_verbosity(0)
np.random.seed(1)

args = ArgController(\
  ).add("-epochs", "Number of training epochs", 200\
  ).parse()

# ===========================================================================
# configs
# ===========================================================================
learning_rate = 1e-3
epochs = int(args.epochs)
SAVE_PATH = "/tmp/vae_basic"
if os.path.exists(SAVE_PATH):
  shutil.rmtree(SAVE_PATH)
os.makedirs(SAVE_PATH)

# ===========================================================================
# load data
# ===========================================================================
datasets, datasets_info = tfds.load(name='mnist',
                                    with_info=True,
                                    as_supervised=False)


def _preprocess(sample):
  image = tf.cast(sample['image'], tf.float32) / 255.  # Scale to unit interval.
  image = image < tf.random.uniform(tf.shape(image))  # Randomly binarize.
  image = tf.cast(image, tf.float32)
  return image, image


train_dataset = (datasets['train'].map(_preprocess).batch(256).prefetch(
    tf.data.experimental.AUTOTUNE).shuffle(int(10e3)))
eval_dataset = (datasets['test'].map(_preprocess).batch(256).prefetch(
    tf.data.experimental.AUTOTUNE))

input_shape = datasets_info.features['image'].shape
encoded_size = 16
base_depth = 32

prior = tfd.Independent(tfd.Normal(loc=tf.zeros(encoded_size), scale=1),
                        reinterpreted_batch_ndims=1)


def create_encoder():
  return [
      tfkl.InputLayer(input_shape=input_shape),
      tfkl.Lambda(lambda x: tf.cast(x, tf.float32) - 0.5),
      tfkl.Conv2D(base_depth,
                  5,
                  strides=1,
                  padding='same',
                  activation=tf.nn.leaky_relu),
      tfkl.Conv2D(base_depth,
                  5,
                  strides=2,
                  padding='same',
                  activation=tf.nn.leaky_relu),
      tfkl.Conv2D(2 * base_depth,
                  5,
                  strides=1,
                  padding='same',
                  activation=tf.nn.leaky_relu),
      tfkl.Conv2D(2 * base_depth,
                  5,
                  strides=2,
                  padding='same',
                  activation=tf.nn.leaky_relu),
      tfkl.Conv2D(4 * encoded_size,
                  7,
                  strides=1,
                  padding='valid',
                  activation=tf.nn.leaky_relu),
      tfkl.Flatten(),
      tfkl.Dense(tfpl.MultivariateNormalTriL.params_size(encoded_size),
                 activation=None),
      tfpl.MultivariateNormalTriL(
          encoded_size,
          activity_regularizer=tfpl.KLDivergenceRegularizer(prior)),
  ]


def create_decoder():
  return [
      tfkl.InputLayer(input_shape=[encoded_size]),
      tfkl.Reshape([1, 1, encoded_size]),
      tfkl.Conv2DTranspose(2 * base_depth,
                           7,
                           strides=1,
                           padding='valid',
                           activation=tf.nn.leaky_relu),
      tfkl.Conv2DTranspose(2 * base_depth,
                           5,
                           strides=1,
                           padding='same',
                           activation=tf.nn.leaky_relu),
      tfkl.Conv2DTranspose(2 * base_depth,
                           5,
                           strides=2,
                           padding='same',
                           activation=tf.nn.leaky_relu),
      tfkl.Conv2DTranspose(base_depth,
                           5,
                           strides=1,
                           padding='same',
                           activation=tf.nn.leaky_relu),
      tfkl.Conv2DTranspose(base_depth,
                           5,
                           strides=2,
                           padding='same',
                           activation=tf.nn.leaky_relu),
      tfkl.Conv2DTranspose(base_depth,
                           5,
                           strides=1,
                           padding='same',
                           activation=tf.nn.leaky_relu),
      tfkl.Conv2D(filters=1,
                  kernel_size=5,
                  strides=1,
                  padding='same',
                  activation=None),
      tfkl.Flatten(),
      tfpl.IndependentBernoulli(input_shape, tfd.Bernoulli.logits),
  ]


# ===========================================================================
# Tensorflow model
# ===========================================================================
encoder = tfk.Sequential(create_encoder(), name="Encoder")
decoder = tfk.Sequential(create_decoder(), name="Decoder")
vae_tfp = tfk.Model(inputs=encoder.inputs, outputs=decoder(encoder.outputs[0]))
negloglik = lambda x, rv_x: -rv_x.log_prob(x)
vae_tfp.compile(optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
                loss=negloglik)
_ = vae_tfp.fit(train_dataset, epochs=epochs, validation_data=eval_dataset)
history = vae_tfp.history.history
# learning curves
fig = plt.figure(figsize=(6, 4), dpi=200)
plt.plot(history['loss'], linestyle='-', label="Training Loss")
plt.plot(history['val_loss'], linestyle='--', label="Validation Loss")
plt.legend()
plt.title("VAE TFP")
plt.grid(True)
plt.ylabel("Loss value")
plt.xlabel("Epochs")
fig.savefig(os.path.join(SAVE_PATH, "tfp.png"), dpi=200)
plt.close(fig)

# ===========================================================================
# Odin model
# ===========================================================================
vae_odin = VariationalAutoencoder(
    encoder=tfk.Sequential(create_encoder()[:-2], name="Encoder"),
    decoder=tfk.Sequential(create_decoder()[:-1], name="Decoder"),
    latents=RandomVariable(event_shape=(encoded_size,),
                           posterior='gaussiandiag',
                           projection=True,
                           prior=prior,
                           name="Latent"),
    outputs=RandomVariable(event_shape=input_shape,
                           posterior="bernoulli",
                           projection=False,
                           name="Image"),
)
vae_odin.fit(train_dataset,
             valid=eval_dataset,
             valid_freq=235,
             optimizer='adam',
             learning_rate=learning_rate,
             epochs=epochs,
             max_iter=-1)
vae_odin.plot_learning_curves(os.path.join(SAVE_PATH, "odin.png"),
                              summary_steps=[100, 10],
                              dpi=200,
                              title="VAE_ODIN")
# ===========================================================================
# evaluation
# ===========================================================================
# calculate the log-likelihood on test set
llk_tfp = []
llk_odin = []
for x, x in eval_dataset.repeat(1):
  llk_tfp.append(vae_tfp(x, training=False).log_prob(x))
  llk_odin.append(vae_odin(x, training=False)[0].log_prob(x))
llk_tfp = tf.reduce_mean(tf.concat(llk_tfp, axis=0))
llk_odin = tf.reduce_mean(tf.concat(llk_odin, axis=0))

n_images = 10
# We'll just examine ten random digits.
x = next(iter(eval_dataset))[0][:n_images]
xhat_tfp = vae_tfp(x, training=False)
xhat_odin, _ = vae_odin(x, training=False)
# Now, let's generate ten never-before-seen digits.
z = prior.sample(n_images)
xtilde_tfp = decoder(z, training=False)
xtilde_odin = vae_odin.decode(z, training=False)

# storing the images
images = [
    (x, "original"),
    #
    (xhat_tfp.sample(), "tfp_xsample"),
    (xhat_tfp.mode(), "tfp_xmode"),
    (xhat_tfp.mean(), "tfp_xmean"),
    (xhat_odin.sample(), "odin_xsample"),
    (xhat_odin.mode(), "odin_xmode"),
    (xhat_odin.mean(), "odin_xmean"),
    #
    (xtilde_tfp.sample(), "tfp_zsample"),
    (xtilde_tfp.mode(), "tfp_zmode"),
    (xtilde_tfp.mean(), "tfp_zmean"),
    (xtilde_odin.sample(), "odin_zsample"),
    (xtilde_odin.mode(), "odin_zmode"),
    (xtilde_odin.mean(), "odin_zmean"),
]

# plotting the images
ncol = n_images
nrow = len(images)
fig = plt.figure(figsize=(ncol * 4, nrow * 4), dpi=200)
count = 0
for img, name in images:
  img = np.asarray(img)
  for i in range(n_images):
    count += 1
    ax = plt.subplot(nrow, ncol, count)
    ax.imshow(img[i].squeeze(), interpolation='none', cmap='gray')
    ax.axis('off')
    if i == 0:
      ax.set_title(name, fontsize=32)
plt.tight_layout(rect=[0.0, 0.02, 1.0, 0.98])
plt.suptitle(f"Test-LLK tfp:{llk_tfp:.2f} odin:{llk_odin:.2f}", fontsize=48)
path = os.path.join(SAVE_PATH, "vae_mnist_test.png")
print("Save figure:", path)
fig.savefig(path)
plt.close(fig=fig)
