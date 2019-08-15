__version__ = "1.0.0"

import os

# this should always be true, the gain in performance
# for preempting whole GPU memory is marginal (memory fragmentation)
# it further prevent you from running multiple experiments
# on 1 GPU, take all memory from other processes even though
# it does not use all the computational resources
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
