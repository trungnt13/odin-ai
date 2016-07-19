from __future__ import print_function, division, absolute_import

import numpy as np


def read(path):
    import imageio
    vid = imageio.get_reader(path)
    metadata = vid.get_meta_data()
    fps = metadata['fps']
    try:
        frames = []
        for i in vid:
            if i.ndim == 3: # swap channel first
                i = i.transpose(2, 0, 1)
            frames.append(i)
    except RuntimeError:
        pass
    frames = np.array(frames, dtype=frames[0].dtype)
    return frames, fps
