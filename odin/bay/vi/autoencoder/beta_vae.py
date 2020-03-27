import tensorflow as tf

from odin.bay.vi.autoencoder.variational_autoencoder import \
    VariationalAutoencoder


class BetaVAE(VariationalAutoencoder):

  def __init__(self, beta=1.0, name='BetaVAE', **kwargs):
    super().__init__(name=name, **kwargs)
    self.beta = tf.convert_to_tensor(beta, dtype=self.dtype, name='beta')

  def elbo(self, X, pX_Z, qZ_X, analytic=False, reverse=True, n_mcmc=1):
    r""" Calculate the distortion (log-likelihood) and rate (KL-divergence)
    for contruction the Evident Lower Bound (ELBO).

    The final ELBO is:
      `ELBO = E_{z~q(Z|X)}[log(p(X|Z))] - KL_{x~p(X)}[q(Z|X)||p(Z)]`

    Arguments:
      analytic : bool (default: False)
        if True, use the close-form solution  for
      n_mcmc : {Tensor, Number}
        number of MCMC samples for MCMC estimation of KL-divergence
      reverse : `bool`. If `True`, calculating `KL(q||p)` which optimizes `q`
        (or p_model) by greedily filling in the highest modes of data (or, in
        other word, placing low probability to where data does not occur).
        Otherwise, `KL(p||q)` a.k.a maximum likelihood, or expectation
        propagation place high probability at anywhere data occur
        (i.e. averagely fitting the data).

    Return:
      log-likelihood : a `Tensor` of shape [sample_shape, batch_size].
        The sample shape could be ommited in case `n_mcmc=()`.
        The log-likelihood or distortion
      divergence : a `Tensor` of shape [n_mcmc, batch_size].
        The reversed KL-divergence or rate
    """
    llk, div = super().elbo(X=X,
                            pX_Z=pX_Z,
                            qZ_X=qZ_X,
                            analytic=analytic,
                            reverse=reverse,
                            n_mcmc=n_mcmc)
    div = self.beta * div
    return llk, div
