from __future__ import print_function, division, absolute_import

from collections import defaultdict


def freqcount(x):
    freq = defaultdict(int)
    for i in x:
        freq[i] += 1
    return freq
