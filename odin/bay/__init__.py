from tensorflow_probability.python.distributions import Distribution

from odin.bay import distributions, layers, mixed_membership
from odin.bay import stochastic_initializers as initializers
from odin.bay import vi
from odin.bay.distribution_alias import parse_distribution
from odin.bay.helpers import *
from odin.bay.random_variable import RandomVariable
from odin.bay.vi import autoencoder
