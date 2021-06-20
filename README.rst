.. image:: https://readthedocs.org/projects/odin/badge/
    :target: http://odin0.readthedocs.org/en/latest/

O.D.I.N
=======
Orgainzed Digital Intelligent Network (O.D.I.N)

O.D.I.N is a framework for building "Organized Digital Intelligent Networks", it uses Tensorflow to create and manage computational graph.

Its end-to-end design aims for a versatile input-to-output framework, that minimized the burden of repetitive work in machine learning pipeline, and allows researchers to conduct experiments in a faster and more flexible way.


Variational Autoencoder (VAE)
****************************

.. list-table::
   :widths: 25 80 25
   :header-rows: 1

   * - Model
     - Reference
     - Implementation
   * - Vanilla VAE
     - (Kingma et al. 2014). "Auto-Encoding Variational Bayes" [`Paper <https://arxiv.org/abs/1312.6114>`_]
     - [`Code <https://github.com/trungnt13/odin-ai/blob/5c83586999a15a02ebbcb7b5f7336f1cce245ae4/odin/bay/vi/autoencoder/variational_autoencoder.py#L132>`_][`Example <https://github.com/trungnt13/odin-ai/blob/master/examples/vae/vae_basic_test.py>`_]
   * - Beta-VAE
     - (Higgins et al. 2016). "beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework" [`Paper <https://openreview.net/forum?id=Sy2fzU9gl>`_]
     - [`Code <https://github.com/trungnt13/odin-ai/blob/5c83586999a15a02ebbcb7b5f7336f1cce245ae4/odin/bay/vi/autoencoder/beta_vae.py#L11>`_][`Example <https://github.com/trungnt13/odin-ai/blob/master/examples/vae/unsupervised_vae_test.py>`_]
   * - BetaGamma-VAE
     - Customized version of Beta-VAE, support re-weighing both reconstruction and regularization  ``\(\mathrm{ELBO}=\gamma \cdot E_q[log p(x|z)] - \beta \cdot KL(q(z|x)||p(z|x))\)``
     - [`Code <https://github.com/trungnt13/odin-ai/blob/5c83586999a15a02ebbcb7b5f7336f1cce245ae4/odin/bay/vi/autoencoder/beta_vae.py#L46>`_][`Example <https://github.com/trungnt13/odin-ai/blob/master/examples/vae/betavae_encoder_info_bound.py>`_]
