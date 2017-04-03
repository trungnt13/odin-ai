from __future__ import print_function, division, absolute_import

import matplotlib
matplotlib.use("Agg")

import os
os.environ['ODIN'] = 'float32,gpu,tensorflow'
import cPickle

import numpy as np
import tensorflow as tf

from odin.config import get_rng
from odin import nnet as N, backend as K, fuel as F, training
from odin.utils import Progbar, get_modelpath, get_logpath


# ====== load dataset ====== #
ds = F.load_mnist()
print(ds)
input_shape = (None,) + ds['X_train'].shape[1:]
print("Input shape:", input_shape)

# ====== get model ====== #
model = N.get_model_descriptor('convolutional_vae')
K.set_training(True); train_loss = model(input_shape)
cPickle.dumps(model)
exit()
K.set_training(False); y_score = model()
parameters = model.parameters
print("Parameters:", [p.name for p in parameters])

# ====== create optimizer ====== #
opt = K.optimizers.Adam(lr=0.01)

# ====== trainng ====== #
trainer, hist = training.standard_trainer(
    train_data=ds['X_train'], valid_data=None, test_data=None,
    cost_train=train_loss, cost_score=train_loss, cost_regu=0.,
    optimizer=opt, parameters=parameters,
    batch_size=128, nb_epoch=12, valid_freq=1.,
    save_path=get_modelpath(name='cnn_vae.ai', override=True), save_obj=model,
    report_path=get_logpath(name='cnn_vae.pdf', override=True),
    stop_callback=opt.get_lr_callback())
