from __future__ import print_function, division, absolute_import

from odin.config import auto_config

config = auto_config()

if config['backend'] == 'theano':
    from .theano import *
elif config['backend'] == 'tensorflow':
    from .tensorflow import *
