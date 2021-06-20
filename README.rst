.. image:: https://readthedocs.org/projects/odin/badge/
    :target: http://odin0.readthedocs.org/en/latest/

O.D.I.N
=======
Organized Digital Intelligent Network (O.D.I.N)

O.D.I.N is a framework for building "Organized Digital Intelligent Networks".

End-to-end design, versatile, plug-n-play, minimized repetitive work

This repo contains the most comprehensive implementation of variational autoencoder and disentangled representation benchmark.

TOC
---

1. `VAE`__
2. `Hierachical VAE`__
3. `Semi-supervised VAE`__
4. `Disentanglement Gym`__
5. `Faster Classical ML`__ (automatically select GPU implementation)

.. __: #variational-autoencoder-vae
.. __: #hierachical-vae
.. __: #semi-supervised-vae
.. __: #disentanglement-gym
.. __: #fast-api-for-classical-ml

Variational Autoencoder (VAE)
-----------------------------

.. list-table::
   :widths: 30 80 25
   :header-rows: 1

   * - Model
     - Reference/Description
     - Implementation
   * - Vanilla VAE
     - (Kingma et al. 2014). "Auto-Encoding Variational Bayes" [`Paper <https://arxiv.org/abs/1312.6114>`_]
     - [`Code <https://github.com/trungnt13/odin-ai/blob/5c83586999a15a02ebbcb7b5f7336f1cce245ae4/odin/bay/vi/autoencoder/variational_autoencoder.py#L132>`_][`Example <https://github.com/trungnt13/odin-ai/blob/master/examples/vae/vae_basic_test.py>`_]
   * - Beta-VAE
     - (Higgins et al. 2016). "beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework" [`Paper <https://openreview.net/forum?id=Sy2fzU9gl>`_]
     - [`Code <https://github.com/trungnt13/odin-ai/blob/master/odin/bay/vi/autoencoder/beta_vae.py>`_][`Example <https://github.com/trungnt13/odin-ai/blob/master/examples/vae/unsupervised_vae_test.py>`_]
   * - BetaGamma-VAE
     - Customized version of Beta-VAE, support re-weighing both reconstruction and regularization  ``\(\mathrm{ELBO}=\gamma \cdot E_q[log p(x|z)] - \beta \cdot KL(q(z|x)||p(z|x))\)``
     - [`Code <https://github.com/trungnt13/odin-ai/blob/master/odin/bay/vi/autoencoder/beta_vae.py>`_][`Example <https://github.com/trungnt13/odin-ai/blob/master/examples/vae/betavae_encoder_info_bound.py>`_]
   * - Annealing VAE
     - (Sønderby et al. 2016) "Ladder variational autoencoder"
     - [`Code <https://github.com/trungnt13/odin-ai/blob/master/odin/bay/vi/autoencoder/beta_vae.py>`_][`Example <https://github.com/trungnt13/odin-ai/blob/master/examples/vae/betavae_encoder_info_bound.py>`_]
   * - CyclicalAnnealing VAE
     - (Fu et al. 2019) "Cyclical Annealing Schedule: A Simple Approach to Mitigating KL Vanishing"
     - [`Code <https://github.com/trungnt13/odin-ai/blob/master/odin/bay/vi/autoencoder/beta_vae.py>`_][`Example <https://github.com/trungnt13/odin-ai/blob/master/examples/vae/betavae_encoder_info_bound.py>`_]
   * - BetaTC-VAE
     - (Chen et al. 2019) "Isolating Sources of Disentanglement in Variational Autoencoders" (regularize the latents' Total Correlation)
     - [`Code <https://github.com/trungnt13/odin-ai/blob/master/odin/bay/vi/autoencoder/beta_vae.py>`_][`Example <https://github.com/trungnt13/odin-ai/blob/master/examples/vae/betavae_encoder_info_bound.py>`_]
   * - Controlled Capacity Beta-VAE
     - (Burgess et al. 2018) "Understanding disentangling in beta-VAE"
     - [`Code <https://github.com/trungnt13/odin-ai/blob/master/odin/bay/vi/autoencoder/beta_vae.py>`_][`Example <https://github.com/trungnt13/odin-ai/blob/master/examples/vae/betavae_encoder_info_bound.py>`_]
   * - FactorVAE
     - (Kim et al. 2018) "Disentangling by Factorising"
     - [`Code <https://github.com/trungnt13/odin-ai/blob/master/odin/bay/vi/autoencoder/factor_vae.py>`_][`Example <https://github.com/trungnt13/odin-ai/blob/master/examples/vae/betavae_encoder_info_bound.py>`_]
   * - AuxiliaryVAE
     - (Maaløe et al. 2016) "Auxiliary Deep Generative Models"
     - [`Code <https://github.com/trungnt13/odin-ai/blob/master/odin/bay/vi/autoencoder/auxiliary_vae.py>`_][`Example <https://github.com/trungnt13/odin-ai/blob/master/examples/vae/betavae_encoder_info_bound.py>`_]
   * - HypersphericalVAE
     - (Davidson et al. 2018) "Hyperspherical Variational Auto-Encoders"
     - [`Code <https://github.com/trungnt13/odin-ai/blob/master/odin/bay/vi/autoencoder/hyperbolic_vae.py>`_][`Example <https://github.com/trungnt13/odin-ai/blob/master/examples/vae/betavae_encoder_info_bound.py>`_]
   * - PowersphericalVAE
     - (De Cao et al. 2020) "The Power Spherical distribution"
     - [`Code <https://github.com/trungnt13/odin-ai/blob/master/odin/bay/vi/autoencoder/hyperbolic_vae.py>`_][`Example <https://github.com/trungnt13/odin-ai/blob/master/examples/vae/betavae_encoder_info_bound.py>`_]
   * - DIPVAE
     - (Kumar et al. 2018) "Variational Inference of Disentangled Latent Concepts from Unlabeled Observations" (I - `only_mean=True`; II - `only_mean=False`)
     - [`Code <https://github.com/trungnt13/odin-ai/blob/master/odin/bay/vi/autoencoder/dip_vae.py>`_][`Example <https://github.com/trungnt13/odin-ai/blob/master/examples/vae/betavae_encoder_info_bound.py>`_]
   * - InfoVAE
     - (Zhao et al. 2018) "infoVAE: Balancing Learning and Inference in Variational Autoencoders"
     - [`Code <https://github.com/trungnt13/odin-ai/blob/master/odin/bay/vi/autoencoder/info_vae.py>`_][`Example <https://github.com/trungnt13/odin-ai/blob/master/examples/vae/betavae_encoder_info_bound.py>`_]
   * - MIVAE
     - (Ducau et al. 2017) "Mutual Information in Variational Autoencoders" (max Mutual Information I(X;Z))
     - [`Code <https://github.com/trungnt13/odin-ai/blob/master/odin/bay/vi/autoencoder/info_vae.py>`_][`Example <https://github.com/trungnt13/odin-ai/blob/master/examples/vae/betavae_encoder_info_bound.py>`_]
   * - irmVAE
     - (Jing et al. 2020) "Implicit Rank-Minimizing Autoencoder" (Implicit Rank Minimizer)
     - [`Code <https://github.com/trungnt13/odin-ai/blob/master/odin/bay/vi/autoencoder/irm_vae.py>`_][`Example <https://github.com/trungnt13/odin-ai/blob/master/examples/vae/betavae_encoder_info_bound.py>`_]
   * - ALDA
     - (Figurnov et al. 2018) "Implicit Reparameterization Gradients" (Amortized Latent Dirichlet Allocation - VAE with Dirichlet latents for topic modeling)
     - [`Code <https://github.com/trungnt13/odin-ai/blob/master/odin/bay/vi/autoencoder/lda_vae.py>`_][`Example <https://github.com/trungnt13/odin-ai/blob/master/examples/vae/betavae_encoder_info_bound.py>`_]
   * - TwoStageVAE
     - (Dai et al. 2019) "Diagnosing and Enhancing VAE Models"
     - [`Code <https://github.com/trungnt13/odin-ai/blob/master/odin/bay/vi/autoencoder/two_stage_vae.py>`_][`Example <https://github.com/trungnt13/odin-ai/blob/master/examples/vae/betavae_encoder_info_bound.py>`_]
   * - VampriorVAE
     - (Tomczak et al. 2018) "VAE with a VampPrior"
     - [`Code <https://github.com/trungnt13/odin-ai/blob/master/odin/bay/vi/autoencoder/vamprior.py>`_][`Example <https://github.com/trungnt13/odin-ai/blob/master/examples/vae/betavae_encoder_info_bound.py>`_]
   * - VQVAE
     - (Oord et al. 2017) "Neural Discrete Representation Learning"
     - [`Code <https://github.com/trungnt13/odin-ai/blob/master/odin/bay/vi/autoencoder/vq_vae.py>`_][`Example <https://github.com/trungnt13/odin-ai/blob/master/examples/vae/betavae_encoder_info_bound.py>`_]


Hierarchical VAE
----------------

.. list-table::
   :widths: 30 80 25
   :header-rows: 1

   * - Model
     - Reference/Description
     - Implementation
   * - LadderVAE
     - (Sønderby et al. 2016) "Ladder variational autoencoder"
     - [`Code <https://github.com/trungnt13/odin-ai/blob/master/odin/bay/vi/autoencoder/hierarchical_vae.py>`_][`Example <https://github.com/trungnt13/odin-ai/blob/master/examples/vae/vae_basic_test.py>`_]
   * - BidirectionalVAE
     - (Kingma et al. 2016) "Improved variational inference with inverse autoregressive flow" (Bidirectional inference hierarchical VAE)
     - [`Code <https://github.com/trungnt13/odin-ai/blob/master/odin/bay/vi/autoencoder/hierarchical_vae.py>`_][`Example <https://github.com/trungnt13/odin-ai/blob/master/examples/vae/vae_basic_test.py>`_]
   * - ParallelVAE
     - (Zhao et al. 2017) "Learning Hierarchical Features from Generative Models" (Multiple latents connects encoder-decoder from bottom to top in parallel)
     - [`Code <https://github.com/trungnt13/odin-ai/blob/master/odin/bay/vi/autoencoder/hierarchical_vae.py>`_][`Example <https://github.com/trungnt13/odin-ai/blob/master/examples/vae/vae_basic_test.py>`_]

Semi-supervised VAE
-------------------

.. list-table::
   :widths: 30 80 25
   :header-rows: 1

   * - Model
     - Reference/Description
     - Implementation
   * - Semi-supervised FactorVAE
     - Same as FactorVAE, but the discriminator also estimate the density of the labels and unlabeled data (like in semi-GAN)
     - [`Code <https://github.com/trungnt13/odin-ai/blob/master/odin/bay/vi/autoencoder/factor_vae.py>`_][`Example <https://github.com/trungnt13/odin-ai/blob/master/examples/vae/semafo_final.py>`_]
   * - MultiheadVAE
     - VAE has multiple decoders for different tasks
     - [`Code <https://github.com/trungnt13/odin-ai/blob/master/odin/bay/vi/autoencoder/multitask_vae.py>`_][`Example <https://github.com/trungnt13/odin-ai/blob/master/examples/vae/semafo_final.py>`_]
   * - SkiptaskVAE
     - VAE has multiple tasks directly constrain the latents
     - [`Code <https://github.com/trungnt13/odin-ai/blob/master/odin/bay/vi/autoencoder/multitask_vae.py>`_][`Example <https://github.com/trungnt13/odin-ai/blob/master/examples/vae/semafo_final.py>`_]
   * - ConditionalM2VAE
     - (Kingma et al. 2014) "Semi-supervised learning with deep generative models" [`Paper <https://arxiv.org/abs/1406.5298>`_]
     - [`Code <https://github.com/trungnt13/odin-ai/blob/master/odin/bay/vi/autoencoder/conditional_vae.py>`_][`Example <https://github.com/trungnt13/odin-ai/blob/master/examples/vae/semafo_final.py>`_]
   * - CCVAE (capture characteristic VAE)
     - (Joy et al. 2021) "Capturing label characteristics in VAEs" [`Paper <https://openreview.net/forum?id=wQRlSUZ5V7B>`_]
     - [`Code <https://github.com/trungnt13/odin-ai/blob/aea88577cbc972230e3d9dabfbe6144509364768/examples/vae/semafo_final.py#L1130>`_][`Example <https://github.com/trungnt13/odin-ai/blob/master/examples/vae/semafo_final.py>`_]
   * - SemafoVAE
     - (Trung et al. 2021) "The transitive information theory and its application to deep generative models" [`Paper <github.com/trungn13>`_]
     - [`Code <https://github.com/trungnt13/odin-ai/blob/aea88577cbc972230e3d9dabfbe6144509364768/examples/vae/semafo_final.py#L351>`_][`Example <https://github.com/trungnt13/odin-ai/blob/master/examples/vae/semafo_final.py>`_]


Disentanglement Gym
-------------------

`DisentanglementGym <https://github.com/trungnt13/odin-ai/blob/master/odin/bay/vi/disentanglement_gym.py>`_: fast API for benchmarks on popular datasets and renowned disentanglement metrics.

Dataset support: `['shapes3d', 'dsprites', 'celeba', 'fashionmnist', 'mnist', 'cifar10', 'cifar100', 'svhn', 'cortex', 'pbmc', 'halfmoons']`

Metrics support:

* Correlation: 'spearman', 'pearson', 'lasso'
* BetaVAE score
* FactorVAE score
* Mutual Information Estimated
* MIG (Mutual Information Gap)
* SAP (Separated Attribute Prediction)
* RDS (relative disentanglement strength)
* DCI (Disentanglement, Completeness, Informativeness)
* FID (Frechet Inception Distance)
* Total Correlation
* Clustering scores: Adjusted Rand Index, Adjusted Mutual Info, Normalized Mutual Info, Silhouette score.


Fast API for classical ML
-------------------------

Automatically accelerated by RAPIDS.ai (i.e. automatically select GPU implementation if available)

Dimension Reduction
~~~~~~~~~~~~~~~~~~~

* t-SNE [`Code <https://github.com/trungnt13/odin-ai/blob/master/odin/ml/fast_tsne.py>`_]
* UMAP [`Code <https://github.com/trungnt13/odin-ai/blob/master/odin/ml/fast_umap.py>`_]
* PCA, Probabilistic PCA, Supervised Probabilistic PCA, MiniBatch PCA, Randomize PCA [`Code <https://github.com/trungnt13/odin-ai/blob/master/odin/ml/decompositions.py>`_]
* Probabilistic Linear Discriminant Analysis (PLDA) [`Code <https://github.com/trungnt13/odin-ai/blob/master/odin/ml/plda.py>`_]
* iVector (GPU acclerated) [`Code <https://github.com/trungnt13/odin-ai/blob/master/odin/ml/ivector.py>`_]

GMM
~~~

* GMM classifier [`Code <https://github.com/trungnt13/odin-ai/blob/master/odin/ml/gmm_classifier.py>`_]
* Probabilistic embedding with GMM [`Code <https://github.com/trungnt13/odin-ai/blob/master/odin/ml/gmm_embedding.py>`_]
* Universal Background Model (GMM-Tmatrix) [`Code <https://github.com/trungnt13/odin-ai/blob/master/odin/ml/gmm_tmat.py>`_]

Clustering
~~~~~~~~~~

* KNN [`Code <https://github.com/trungnt13/odin-ai/blob/master/odin/ml/cluster.py>`_]
* KMeans [`Code <https://github.com/trungnt13/odin-ai/blob/master/odin/ml/cluster.py>`_]
* DBSCAN [`Code <https://github.com/trungnt13/odin-ai/blob/master/odin/ml/cluster.py>`_]
