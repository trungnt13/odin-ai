from __future__ import absolute_import, division, print_function

import os
import time

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras

from odin.training import Trainer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

tf.random.set_seed(8)
np.random.seed(8)

# ===========================================================================
# Load data
# ===========================================================================
train, valid, test = tfds.load('fashion_mnist:3.0.0',
                               split=['train[:80%]', 'train[80%:]', 'test'],
                               read_config=tfds.ReadConfig(
                                   shuffle_seed=1,
                                   shuffle_reshuffle_each_iteration=True))

input_shape = tf.data.experimental.get_structure(train)['image'].shape


def process(data):
  image = tf.cast(data['image'], tf.float32)
  label = tf.cast(data['label'], tf.float32)
  image = (image / 255. - 0.5) * 2.
  return image, label


# ===========================================================================
# Test
# ===========================================================================
network = keras.Sequential([
    keras.layers.Flatten(input_shape=input_shape),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax'),
])
opt = tf.optimizers.Adam(learning_rate=0.001,
                         beta_1=0.9,
                         beta_2=0.999,
                         epsilon=1e-07,
                         amsgrad=False)


def optimize(inputs, training):
  X, y_true = inputs
  with tf.GradientTape(watch_accessed_variables=bool(training)) as tape:
    y_pred = network(X, training=training)
    loss = tf.reduce_mean(
        tf.losses.sparse_categorical_crossentropy(y_true, y_pred))
    acc = tf.cast(y_true == tf.cast(tf.argmax(y_pred, axis=-1), tf.float32),
                  tf.float32)
    acc = tf.reduce_sum(acc) / tf.cast(tf.shape(y_true)[0], tf.float32)
    if training:
      Trainer.apply_gradients(tape, opt, loss, network)
  return loss, acc


def callback():
  signal = Trainer.early_stop(trainer.valid_loss, threshold=0.25, verbose=True)
  if signal == Trainer.SIGNAL_BEST:
    print(" - Save the best weights!")
    Trainer.save_weights(network)
  elif signal == Trainer.SIGNAL_TERMINATE:
    print(" - Restore the best weights!")
    Trainer.restore_weights(network)
  return signal


trainer = Trainer()

start_time = time.time()
trainer.fit(Trainer.prepare(train,
                            postprocess=process,
                            parallel_postprocess=False,
                            shuffle=True,
                            epochs=32),
            optimize,
            valid_ds=Trainer.prepare(valid, postprocess=process),
            valid_freq=2500,
            autograph=True,
            logging_interval=2,
            on_valid_end=callback)
print("Total:", time.time() - start_time)
