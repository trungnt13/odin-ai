from tensorflow_probability.python.distributions import Distribution

from odin.bay import distributions, layers, mixed_membership
from odin.bay import stochastic_initializers as initializers
from odin.bay.distribution_alias import parse_distribution
# this is important utility
from odin.bay.distributions.utils import concat_distribution
from odin.bay.helpers import *
