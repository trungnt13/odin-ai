from .base import evaluate
from .decompositions import *
from .clustering import *
from .gmm_ivec import GMM, Tmatrix
from .linear_model import LogisticRegression
from .scoring import (VectorNormalizer, Scorer,
                      compute_wccn, compute_class_avg, compute_within_cov)
from .plda import PLDA, GPLDA
