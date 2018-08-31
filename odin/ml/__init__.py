from .base import evaluate
from .decompositions import *
from .gmm_tmat import GMM, Tmatrix
from .ivector import Ivector
from .linear_model import LogisticRegression
from .deep_model import *
from .scoring import (VectorNormalizer, Scorer,
                      compute_wccn, compute_class_avg, compute_within_cov)
from .plda import PLDA
from .gmm_classifier import GMMclassifier
from .fast_tsne import fast_tsne
