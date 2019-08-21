from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow_probability.python.layers import (MixtureLogistic,
                                                  MixtureNormal,
                                                  MixtureSameFamily)

__all__ = [
    'MixtureLogisticLayer', 'MixtureNormalLayer', 'MixtureSameFamilyLayer'
]
MixtureLogisticLayer = MixtureLogistic
MixtureNormalLayer = MixtureNormal
MixtureSameFamilyLayer = MixtureSameFamily
