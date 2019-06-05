import matplotlib
matplotlib.use('Agg')

from matplotlib import pyplot as plt

import os
os.environ['ODIN'] = 'cpu'

import numpy as np
from odin.utils import one_hot
from odin import backend as K, fuel as F, visual as V
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, classification_report, precision_recall_curve
np.random.seed(1234)


# ===========================================================================
# Copy implementation from SIDEKIT to validate
# our implementation of DET, EER and minDCF
# ===========================================================================
def pavx(y):
    """PAV: Pool Adjacent Violators algorithm.
    Non-paramtetric optimization subject to monotonicity.

    ghat = pav(y)
    fits a vector ghat with nondecreasing components to the
    data vector y such that sum((y - ghat).^2) is minimal.
    (Pool-adjacent-violators algorithm).

    optional outputs:
            width: width of pav bins, from left to right
                    (the number of bins is data dependent)
            height: corresponding heights of bins (in increasing order)

    Author: This code is a simplified version of the 'IsoMeans.m' code
    made available by Lutz Duembgen at:
    http://www.imsv.unibe.ch/~duembgen/software

    :param y: input value
    """
    assert y.ndim == 1, 'Argument should be a 1-D array'
    assert y.shape[0] > 0, 'Input array is empty'
    n = y.shape[0]

    index = np.zeros(n)
    length = np.zeros(n)

    # An interval of indices is represented by its left endpoint
    # ("index") and its length "length"
    ghat = np.zeros(n)

    ci = 0
    index[ci] = 0
    length[ci] = 1
    ghat[ci] = y[0]

    # ci is the number of the interval considered currently.
    # ghat(ci) is the mean of y-values within this interval.
    for j in range(1, n):
        # a new index interval, {j}, is created:
        ci += 1
        index[ci] = j
        length[ci] = 1
        ghat[ci] = y[j]
        while (ci >= 1) & (ghat[np.max(ci - 1, 0)] >= ghat[ci]):
            # pool adjacent violators:
            nw = length[ci - 1] + length[ci]
            ghat[ci - 1] = ghat[ci - 1] + (length[ci] / nw) * (ghat[ci] - ghat[ci - 1])
            length[ci - 1] = nw
            ci -= 1

    height = ghat[:ci + 1]
    width = length[:ci + 1]

    # Now define ghat for all indices:
    while n >= 0:
        for j in range(int(index[ci]), int(n)):
            ghat[j] = ghat[ci]

        n = index[ci] - 1
        ci -= 1

    return ghat, width, height


def rocch(tar_scores, nontar_scores):
    """ROCCH: ROC Convex Hull.
    Note: pmiss and pfa contain the coordinates of the vertices of the
    ROC Convex Hull.

    For a demonstration that plots ROCCH against ROC for a few cases, just
    type 'rocch' at the MATLAB command line.

    :param tar_scores: vector of target scores
    :param nontar_scores: vector of non-target scores

    :return: a tupple of two vectors: Pmiss, Pfa
    """
    Nt = tar_scores.shape[0]
    Nn = nontar_scores.shape[0]
    N = Nt + Nn
    scores = np.concatenate((tar_scores, nontar_scores))
    # Pideal is the ideal, but non-monotonic posterior
    Pideal = np.concatenate((np.ones(Nt), np.zeros(Nn)))
    #
    # It is important here that scores that are the same
    # (i.e. already in order) should NOT be swapped.rb
    perturb = np.argsort(scores, kind='mergesort')
    #
    Pideal = Pideal[perturb]
    Popt, width, foo = pavx(Pideal)
    #
    nbins = width.shape[0]
    pmiss = np.zeros(nbins + 1)
    pfa = np.zeros(nbins + 1)
    #
    # threshold leftmost: accept everything, miss nothing
    left = 0    # 0 scores to left of threshold
    fa = Nn
    miss = 0
    #
    for i in range(nbins):
        pmiss[i] = miss / Nt
        pfa[i] = fa / Nn
        left = int(left + width[i])
        miss = np.sum(Pideal[:left])
        fa = N - left - np.sum(Pideal[left:])
    #
    pmiss[nbins] = miss / Nt
    pfa[nbins] = fa / Nn
    #
    return pmiss, pfa


def sigmoid(log_odds):
    """SIGMOID: Inverse of the logit function.
    This is a one-to-one mapping from log odds to probability.
    i.e. it maps the real line to the interval (0,1).

    p = sigmoid(log_odds)

    :param log_odds: the input value

    :return: sigmoid(input)
    """
    p = 1 / (1 + np.exp(-log_odds))
    return p


def fast_minDCF(tar, non, plo, normalize=False):
    """Compute the minimum COST for given target and non-target scores
    Note that minDCF is parametrized by plo:

        minDCF(Ptar) = min_t Ptar * Pmiss(t) + (1-Ptar) * Pfa(t)

    where t is the adjustable decision threshold and:

        Ptar = sigmoid(plo) = 1./(1+exp(-plo))

    If normalize == true, then the returned value is:

        minDCF(Ptar) / min(Ptar,1-Ptar).

    Pmiss: a vector with one value for every element of plo.
    This is Pmiss(tmin), where tmin is the minimizing threshold
    for minDCF, at every value of plo. Pmiss is not altered by
    parameter 'normalize'.

    Pfa: a vector with one value for every element of plo.
    This is Pfa(tmin), where tmin is the minimizing threshold for
    minDCF, at every value of plo. Pfa is not altered by
    parameter 'normalize'.

    Note, for the un-normalized case:

        minDCF(plo) = sigmoid(plo).*Pfa(plo) + sigmoid(-plo).*Pmiss(plo)

    :param tar: vector of target scores
    :param non: vector of non-target scores
    :param plo: vector of prior-log-odds: plo = logit(Ptar) = log(Ptar) - log(1-Ptar)
    :param normalize: if true, return normalized minDCF, else un-normalized (optional, default = false)

    :return: the minDCF value
    :return: the miss probability for this point
    :return: the false-alarm probability for this point
    :return: the precision-recall break-even point: Where #FA == #miss
    :return the equal error rate
    """
    Pmiss, Pfa = rocch(tar, non)
    Nmiss = Pmiss * tar.shape[0]
    Nfa = Pfa * non.shape[0]
    prbep = compute_EER(Nmiss, Nfa)
    eer = compute_EER(Pmiss, Pfa)

    Ptar = sigmoid(plo)
    Pnon = sigmoid(-plo)
    cdet = np.dot(np.array([[Ptar, Pnon]]), np.vstack((Pmiss, Pfa)))
    ii = np.argmin(cdet, axis=1)
    minDCF = cdet[0, ii][0]

    Pmiss = Pmiss[ii]
    Pfa = Pfa[ii]

    if normalize:
        minDCF = minDCF / min([Ptar, Pnon])

    return minDCF, Pmiss[0], Pfa[0], prbep, eer


def compute_EER(pmiss, pfa):
    """Calculates the equal error rate (eer) from pmiss and pfa vectors.
    Note: pmiss and pfa contain the coordinates of the vertices of the
    ROC Convex Hull.
    Use rocch.m to convert target and non-target scores to pmiss and
    pfa values.

    :param pmiss: the vector of miss probabilities
    :param pfa: the vector of false-alarm probabilities

    :return: the equal error rate
    """
    eer = 0
    for i in range(pfa.shape[0] - 1):
        xx = pfa[i:i + 2]
        yy = pmiss[i:i + 2]
        # xx and yy should be sorted:
        assert (xx[1] <= xx[0]) & (yy[0] <= yy[1]), \
            'pmiss and pfa have to be sorted'

        XY = np.column_stack((xx, yy))
        dd = np.dot(np.array([1, -1]), XY)
        if np.min(np.abs(dd)) == 0:
            eerseg = 0
        else:
            # find line coefficients seg s.t. seg'[xx(i);yy(i)] = 1,
            # when xx(i),yy(i) is on the line.
            seg = np.linalg.solve(XY, np.array([[1], [1]]))
            # candidate for EER, eer is highest candidate
            eerseg = 1 / (np.sum(seg))

        eer = max([eer, eerseg])
    return eer

# ===========================================================================
# Testing
# ===========================================================================
n_true = 1000
n_false = 500
true = np.random.randn(n_true)
mean_False = -3
stdv_False = 1.5
false = stdv_False * np.random.randn(n_false) + mean_False

y_true = np.zeros(shape=(n_true + n_false,))
y_true[:n_true] = 1
y_score = np.concatenate((true, false))

Pfa, Pmiss = K.metrics.det_curve(y_true=y_true, y_score=y_score)

min_DCF, Pfa_opt, Pmiss_opt = K.metrics.compute_minDCF(Pfa, Pmiss)
print("MinDCF, Pmiss_opt, Pfa_opt:", min_DCF, Pmiss_opt, Pfa_opt)
print("EER1:", K.metrics.compute_EER(Pfa, Pmiss))

pmiss, pfa = rocch(tar_scores=true, nontar_scores=false)
min_DCF, Pfa_opt, Pmiss_opt = K.metrics.compute_minDCF(pfa, pmiss)
print("[Sidekit]MinDCF, Pmiss_opt, Pfa_opt:", min_DCF, Pmiss_opt, Pfa_opt)
print("[Sidekit]EER:", compute_EER(pmiss, pfa))
print("[Sidekit]MinDCF, Pmiss_opt, Pfa_opt, ..., EER:", fast_minDCF(tar=true, non=false, plo=0))

fpr, tpr, _ = K.metrics.roc_curve(y_true=y_true, y_score=y_score)
auc = K.metrics.compute_AUC(tpr, fpr)

# ====== specialized plotting ====== #
plt.figure()
V.plot_detection_curve(x=pfa, y=pmiss, curve='det')
plt.figure()
V.plot_detection_curve(x=Pfa, y=Pmiss, curve='det')
plt.figure()
V.plot_detection_curve(x=fpr, y=tpr, curve='roc')
V.plot_save('/tmp/tmp.pdf')
