from .base import evaluate
from .decompositions import *
from .clustering import *
from .gmm_ivec import GMM, Tmatrix
from .linear_model import LogisticRegression
from .deep_model import *
from .scoring import (VectorNormalizer, Scorer,
                      compute_wccn, compute_class_avg, compute_within_cov)
from .plda import PLDA
from .gmm_classifier import GMMclassifier
from .fast_tsne import fast_tsne
