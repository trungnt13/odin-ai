Installation
============

O.D.I.N requires the installation of prerequisites_ in different ways, however, it is relaxed about the version of those.

This tutorial is provided based on the assumption that you are running an Unix system, but is otherwise very generic.

The most stable and simplest way to install all O.D.I.N' dependencies is :ref:`Anaconda <anaconda-section>`, and we also support *Python package manager* :ref:`pip <pip-section>`. We recommend using the *cutting-edge development* version of O.D.I.N, which is constantly updated at our `GitHub repository`_.

A rather straightforward way to install the GitHub repository is:

.. code-block:: bash

    $ pip install git+git://GitHub.com/imito/odin.git \
  -r https://raw.GitHubusercontent.com/imito/odin/master/requirements.txt

.. tip::

    If you don’t have administrative rights, you can install the package to your ``$HOME`` using the ``--user``. If you want to update O.D.I.N, simply repeat the first part of command with the ``--upgrade`` switch to pull the latest version from GitHub.

.. note::

   O.D.I.N is operated on 2 different backends for creating computational graph, these are: Theano_ and tensorflow_. The ``requirements.txt`` will install Theano_ by default.

.. _environment: https://GitHub.com/trungnt13/envs
.. _GitHub repository: https://GitHub.com/imito/odin

.. ======================== prerequisites ========================
.. _prerequisites:

Prerequisites
-------------
O.D.I.N' requirements include

* Theano_, as computational backend, for pretty much everything
* [or] tensorflow_, as computational backend, for pretty much everything
* numpy_, is used for tensor manipulation, linear algebra, and scientific computing.
* scipy_, as a mathematics and signal processing toolkits.
* six_, to support both Python 2 and 3 with a single codebase

We develop using the bleeding-edge version of Theano_, and latest stable version of tensorflow_, so it is important to install the right Theano and tensorflow version that support O.D.I.N following this instruction :ref:`pip <pip-section>` or :ref:`Anaconda <anaconda-section>`.

.. note::

    O.D.I.N provides identical interface to both Theano_ and tensorflow_. User can switch the backend on-the-fly with zero modification in the code.

External requirements for signal processing:

* SIDEKIT_, is an open source package for speech processing, especially for Speaker and Language recognition.
* resampy_, resampling library for signal processing
* imageio_, is a Python library that provides an easy interface to read and write a wide range of image and video data.
* PIL_, adds image processing capabilities to your Python interpreter.
* spacy_, is an industrial-strength natural language processing engine.
* matplotlib_, is a plotting library for visualization.

.. warning::

    All of these packages are **not** required for running neural network API in O.D.I.N, they are only involved in the data preprocessing pipeline. The computational backend is developed independently from data preprocessing API which makes O.D.I.N flexible but versatile.

.. _Theano: https://GitHub.com/Theano/Theano
.. _resampy: https://github.com/bmcfee/resampy
.. _tensorflow: https://GitHub.com/tensorflow/tensorflow
.. _numpy: http://www.numpy.org/
.. _scipy: https://www.scipy.org/
.. _matplotlib: http://matplotlib.org/
.. _SIDEKIT: http://www-lium.univ-lemans.fr/sidekit/
.. _imageio: http://imageio.GitHub.io
.. _PIL: http://www.pythonware.com/products/pil/
.. _spacy: https://spacy.io/
.. _six: http://pythonhosted.org/six/

.. ======================== Anaconda ========================
.. _anaconda-section:

Dependencies via Anaconda
-------------------------

`Anaconda <http://anaconda.org>`_ or `Miniconda <http://conda.pydata.org/miniconda.html>`_ can be used as package management for setting up O.D.I.N development environment. The simplest way is fetching the latest version of Miniconda from `<http://conda.pydata.org/miniconda.html>`_, the installation can be finished with one command:

.. code-block:: bash

    bash Miniconda[2]-latest-[MacOSX]-x86_64.sh

The python version (or Miniconda version) can *2* or *3*, and the OS can be *MacOSX*, *Linux*, or *Windows*.

A complete python environment for Miniconda is provided `here <https://GitHub.com/trungnt13/envs>`_. After downloading the *environment.yml* provided in folder *ai-linux* or *ai-osx*, execute following command:

.. code-block:: bash

    conda env create -f=/path/to/environment.yml

This will install all the necessary packages for you to run O.D.I.N or developing machine learning algorithm in general.
After the installation progress finished,you can activated the environment by:

.. code-block:: bash

    source activate ai

where **ai** is the name of our environment.

.. warning::

    If you want to manually install all the dependencies via *conda*, we recommend you take a look at our `channel <https://anaconda.org/trung/repo>`_, or you can simply include ``-c trung`` when running ``conda install``. The channel is up-to-date, and especially optimized for Theano developers.

.. ======================== Backend ========================
.. _pip-section:

Dependencies via pip
--------------------

O.D.I.N currently supports both Python 2.7 or 3.4. Please install Python via the package manager of your operating system if it is not included already.

Python includes ``pip`` for installing additional modules that are not shipped with your operating system, or shipped in an old version, and we will make use of it below.
We recommend installing these modules into your home directory via ``--user``, or into a `virtual environment
<http://www.dabapps.com/blog/introduction-to-pip-and-virtualenv-python/>`_ via ``virtualenv``.

O.D.I.N requires numpy of version 1.10 or above, and Theano also requires scipy 0.11 or above. In order to install a specific version of pip package:

.. code-block:: bash

    $ pip install numpy==1.11.2

To install a list of all required packages for O.D.I.N:

.. code-block:: bash

    $ pip install -r https://raw.GitHubusercontent.com/imito/odin/master/requirements.txt

Numpy/scipy rely on a BLAS library to provide fast linear algebra routines.
They will work fine without one, but a lot slower, so it is worth getting this right (but this is less important if you plan to use a GPU).

.. warning::

   Pip may try to install or update NumPy and SciPy if they are not present or outdated. However, pip's versions might not be linked to an optimized BLAS implementation. To prevent this from happening make sure you update NumPy and SciPy using your system's package manager (e.g.  ``apt-get`` or ``yum``), or make sure to have development headers for your BLAS library installed (e.g., the ``libopenblas-dev`` package on Debian/Ubuntu) while running the installation command.

   If the installation crashes with ``ImportError: No module named
   numpy.distutils.core``, install NumPy and try again again.

.. ======================== Development ========================
.. _development-install:

Development installation
------------------------

If you want to contribute to O.D.I.N, or write your own version of O.D.I.N, you can install the framework from source.
This is often referred to as *editable* or *development* mode. Firstly, you can obtain the latest source code from GitHub using:

.. code-block:: bash

  git clone https://github.com/imito/odin.git

It will be cloned to a subdirectory called ``odin``. Make sure to place it in some permanent location, as for an *editable* installation, Python will import the module directly from this directory and not copy over the files.

To install the O.D.I.N package itself, in editable mode (add ``--user`` to install it to your home directory), run:

.. code-block:: bash

    pip install --editable .

Alternatively, you can add the path to ``odin`` repository to ``$PYTHONPATH`` variable

.. code-block:: bash

    export PYTHONPATH=$PYTHONPATH:/Users/trungnt13/libs/odin


**Optional**: If you plan to contribute to O.D.I.N, you will need to fork the
O.D.I.N repository on GitHub. This will create a repository under your user
account. Update your local clone to refer to the official repository as
``upstream``, and your personal fork as ``origin``:

.. code-block:: bash

  git remote rename origin upstream
  git remote add origin https://github.com/<your-github-name>/odin.git

If you set up an `SSH key <https://help.github.com/categories/ssh/>`_, use the
SSH clone URL instead: ``git@github.com:<your-github-name>/odin.git``.

You can now use this installation to develop features and send us pull requests
on GitHub, see :doc:`principle`!

Documentation
~~~~~~~~~~~~~

If you want to build a local copy of the documentation, you need `Sphinx-doc <http://www.sphinx-doc.org/en/1.5.1/>`_ 1.4 or above, and follow the instruction at :doc:`documentation development guidelines <docs>`.

.. ======================== Development ========================
GPU support
-----------

If you are using Theano backend, the support for GPU is transparent and totally managed by O.D.I.N
Running the code using GPU requires NVIDIA GPU with CUDA support, and some additional software for
Theano to use it.

However, you need to build specific version of tensorflow that is enabled for GPU support. You can find more information at `this instruction <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/get_started/os_setup.md`_.

CUDA
~~~~

Install the latest CUDA Toolkit and possibly the corresponding driver available
from NVIDIA: https://developer.nvidia.com/cuda-downloads

Closely follow the *Getting Started Guide* linked underneath the download table
to be sure you don't mess up your system by installing conflicting drivers.

After installation, make sure ``/usr/local/cuda/bin`` is in your ``PATH``, so
``nvcc --version`` works. Also make sure ``/usr/local/cuda/lib64`` is in your
``LD_LIBRARY_PATH``, so the toolkit libraries can be found.

cuDNN
~~~~~

NVIDIA provides a library for common neural network operations that especially
speeds up Convolutional Neural Networks (CNNs). Again, it can be obtained from
NVIDIA (after registering as a developer): https://developer.nvidia.com/cudnn

.. note::

    O.D.I.N uses cuDNN' convolution kernel by default, hence, it is required if you want to use convolutional neural network. We also provide support cuDNN' *recurrent neural network* (RNN), which is significantly faster than traditional implementation of RNN.

To install cuDNN, copy the ``*.h`` files to ``/usr/local/cuda/include`` and the
``lib*`` files to ``/usr/local/cuda/lib64``.

.. warning::

    It requires a reasonably modern GPU with Compute Capability 3.0 or higher;
    see `NVIDIA's list of CUDA GPUs <https://developer.nvidia.com/cuda-gpus>`_.

To check whether it is found by Theano, run the following command:

.. code-block:: bash

  python -c "from theano.sandbox.cuda.dnn import dnn_available as d; print(d() or d.msg)"

It will print ``True`` if everything is fine, or an error message otherwise.
There are no additional steps required for Theano to make use of cuDNN.

For tensorflow, you can link cuDNN to your installation by following `this instruction <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/get_started/os_setup.md#configure-the-installation>`_.
