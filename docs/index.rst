O.D.I.N' documentation is here!
=================================

O.D.I.N is a framework for building "Organized Digital Intelligent Networks", it uses Tensorflow or Theano to create and manage computational graph.

Its end-to-end design aims for a versatile input-to-output framework, that minimized the burden of repetitive work in machine learning pipeline, and allows researchers to conduct experiments in a faster and more flexible way.

Start by :doc:`installing <setup>` O.D.I.N and having a look at the :ref:`quickstart <quickstart>` further down this page.
Once you're hooked, try your hand at the :ref:`tutorials <tutorials>`.

.. warning::
   O.D.I.N is a new project which is still under development. As such, certain
   (all) parts of the framework are subject to change. The last stable (and
   thus likely an outdated) version can be found in the ``stable`` branch.

.. tip::

   That said, if you are interested in using O.D.I.N and run into any problems,
   feel free to ask your question by sending email to ``admin[at]imito[dot]ai``. Also, don't hesitate to file bug reports and feature requests by `making a GitHub issue`_.

.. _making a GitHub issue: https://github.com/imito/odin/issues/new


.. ======================== Tutorial ========================
.. _tutorials:

Tutorials
---------
.. toctree::
   :maxdepth: 1

   setup
   principle

.. ======================== Quick start ========================
.. _quickstart:

Quickstart
==========

The source code is here: `mnist.py <https://github.com/imito/odin/blob/master/examples/mnist.py>`_

O.D.I.N is straightforward, all the configuration can be controlled within the script.
The configuration is designed given a fixed set of keywords to limit human mistakes at the beginning.

.. code-block:: python

    import os
    os.environ['ODIN'] = 'float32,gpu,tensorflow,seed=12'
    from odin import backend as K
    from odin import nnet as N
    from odin import fuel, training

Loading experimental dataset with only *one* line of code:

.. code-block:: python

    ds = fuel.load_mnist()

Creating input and output variables:

.. code-block:: python

    X = K.placeholder(shape=(None,) + ds['X_train'].shape[1:], name='X')
    y = K.placeholder(shape=(None,), name='y', dtype='int32')

Creating model is intuitive, no *input shapes* are required at the beginning, everything is automatically inferred based on *input variables*.

.. code-block:: python

    ops = N.Sequence([
        N.Dimshuffle((0, 1, 2, 'x')),
        N.BatchNorm(axes='auto'),
        N.Conv(32, (3, 3), strides=(1, 1), pad='same', activation=K.relu),
        N.Pool(pool_size=(2, 2), strides=None),
        N.Conv(64, (3, 3), strides=(1, 1), pad='same', activation=K.relu),
        N.Pool(pool_size=(2, 2), strides=None),
        N.Flatten(outdim=2),
        N.Dense(256, activation=K.relu),
        N.Dense(10, activation=K.softmax)
    ], debug=True)

O.D.I.N is a functional API, all neural network operators are functions, they can be applied
on different variables and configuration to get different outputs (i.e. creating different model sharing the same set of parameters).

.. code-block:: python

    K.set_training(True); y_pred_train = ops(X)
    K.set_training(False); y_pred_score = ops(X)

O.D.I.N provides identical interface for both Theano and tensorflow, hence, the following functions are operate the same in both backends:

.. code-block:: python

    cost_train = K.mean(K.categorical_crossentropy(y_pred_train, y))
    cost_test_1 = K.mean(K.categorical_crossentropy(y_pred_score, y))
    cost_test_2 = K.mean(K.categorical_accuracy(y_pred_score, y))
    cost_test_3 = K.confusion_matrix(y_pred_score, y, labels=range(10))

We also provides a set of optimization algorithms to train your network, all the optimizers are implemented in `optimizers.py <https://github.com/imito/odin/blob/master/odin/backend/optimizers.py>`_

.. code-block:: python

    parameters = ops.parameters
    optimizer = K.optimizers.SGD(lr=0.01) # R
    updates = optimizer(cost_train, parameters)
    print('Building training functions ...')
    f_train = K.function([X, y], [cost_train, optimizer.norm],
                         updates=updates)
    print('Building testing functions ...')
    f_test = K.function([X, y], [cost_test_1, cost_test_2, cost_test_3])
    print('Building predicting functions ...')
    f_pred = K.function(X, y_pred_score)

In O.D.I.N, we implement a generic process of optimizing any network. The training script is independent from all other parts of the framework, and can be extended by inheriting :class:`Callback` in `<https://github.com/imito/odin/blob/master/odin/training/callbacks.py>`_.

.. code-block:: python

    task = training.MainLoop(batch_size=32, seed=12, shuffle_level=2)
    task.set_save(get_modelpath(name='mnist.ai', override=True), ops)
    task.set_task(f_train, (ds['X_train'], ds['y_train']), epoch=arg['epoch'], name='train')
    task.set_subtask(f_test, (ds['X_test'], ds['y_test']), freq=0.6, name='valid')
    task.set_subtask(f_test, (ds['X_test'], ds['y_test']), when=-1, name='test')
    task.set_callback([
        training.ProgressMonitor(name='train', format='Results: {:.4f}-{:.4f}'),
        training.ProgressMonitor(name='valid', format='Results: {:.4f}-{:.4f}',
                                 tracking={2: lambda x: sum(x)}),
        training.ProgressMonitor(name='test', format='Results: {:.4f}-{:.4f}'),
        training.History(),
        training.EarlyStopGeneralizationLoss('valid', threshold=5, patience=3),
        training.NaNDetector(('train', 'valid'), patience=3, rollback=True)
    ])
    task.run()

You can directly visualize the training progress in your terminal, using the *bashplotlib* API

.. code-block:: python

    # ====== plot the training process ====== #
    task['History'].print_info()
    task['History'].print_batch('train')
    task['History'].print_batch('valid')
    task['History'].print_epoch('test')

The code will print out something like this in your terminal

.. code-block:: text

             0.993|
             0.992|                  oo
             0.990|                  ooo
             0.989|             o    ooo
             0.988|          oo o    ooo
             0.986|          ooooo o ooo
             0.985|      o  oooooooo ooo
             0.983|     oooooooooooo ooo
             0.982|     oooooooooooooooo
             0.981|     oooooooooooooooo
             0.979|  o  oooooooooooooooo
             0.978|  o  oooooooooooooooo
             0.977|  o  oooooooooooooooo
             0.975| oo ooooooooooooooooo
             0.974| oo ooooooooooooooooo
             0.973| oo ooooooooooooooooo
             0.971| oo ooooooooooooooooo
             0.970| oo ooooooooooooooooo
             0.969| oo ooooooooooooooooo
             0.967| oo ooooooooooooooooo
             0.966| oooooooooooooooooooo
                   --------------------

    ------------------------------
    |          Summary           |
    ------------------------------
    |     observations: 785      |
    |    min value: 0.890625     |
    |      mean : 0.984614       |
    |       sd : 0.019188        |
    |    max value: 1.000000     |
    ------------------------------

Features
--------

Currently O.D.I.N supports and provides:

* End-to-end framework, provides a ``full-stack`` support from features preprocessing to inference.
* Fast, reliable and efficient class for handle **big** dataset, O.D.I.N can load terabytes of data at once, re-organize the features using multiple processes, and training the network on new features at the same time.
* Constructing computational and parametrized neural network operations.
* Pattern matching to select variables and bricks in large models.
* Algorithms to optimize your model.
* All the parametrized operations are pickle-able.
* Generic progress for optimization, many algorithms to prevent overfiting, detecting early failure, monitoring and analyzing values during training progress (on the training set as well as on test sets).

In the future we also hope to support:

* Multiple-GPUs training
* Distributing parametrized models among multiple GPUs

Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
