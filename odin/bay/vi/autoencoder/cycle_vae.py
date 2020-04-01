class CycleConsistentVAE():
  r"""

  Reference:
    Jha, A.H., Anand, S., Singh, M., Veeravasarapu, V.S.R., 2018.
      "Disentangling Factors of Variation with Cycle-Consistent
      Variational Auto-Encoders". arXiv:1804.10469 [cs].
    Implementation: https://github.com/ananyahjha93/cycle-consistent-vae
  """

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
