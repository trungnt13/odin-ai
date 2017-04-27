from .utils import *
from .data import *
from .dataset import *
from .features import *
from .feeders import *

import recipes
from zipfile import ZipFile, ZIP_DEFLATED

from odin.utils import get_file


def load_swb1_aligment(nb_senones=2304):
    support_nb_senones = (2304,)
    if nb_senones not in support_nb_senones:
        raise ValueError('We only support following number of senones: %s'
                        % support_nb_senones)
    fname = "swb1_%d" % nb_senones
    url = "https://s3.amazonaws.com/ai-datasets/" + fname
    alignment = get_file(fname, url)
    return MmapDict(alignment, read_only=True)


def load_glove(ndim=100):
    """ Automaticall load a MmapDict which contains the mapping
        (word -> [vector])
    where vector is the embedding vector with given `ndim`.
    """
    ndim = int(ndim)
    if ndim not in (50, 100, 200, 300):
        raise ValueError('Only support 50, 100, 200, 300 dimensions.')
    fname = 'glove.6B.%dd' % ndim
    link = 'https://s3.amazonaws.com/ai-datasets/%s' % fname
    embedding = get_file(fname, link)
    return MmapDict(embedding, read_only=True)


def load_20newsgroup():
    link = 'https://s3.amazonaws.com/ai-datasets/news20'
    dataset = get_file('news20', link)
    return MmapDict(dataset, read_only=True)


def load_word2vec():
    """ Loading google pretrained word2vec from:
    https://code.google.com/archive/p/word2vec/
    """
    pass


def load_digit_wav():
    path = 'https://s3.amazonaws.com/ai-datasets/digit_wav.zip'
    datapath = get_file('digit_wav.zip', path)
    try:
        outpath = datapath.replace('.zip', '')
        if os.path.exists(outpath):
            shutil.rmtree(outpath)
        zf = ZipFile(datapath, mode='r', compression=ZIP_DEFLATED)
        zf.extractall(path=outpath + '/../'); zf.close()
    except:
        # remove downloaded zip files
        os.remove(datapath)
        import traceback; traceback.print_exc()
    return Dataset(outpath, read_only=True)


def load_commands_wav():
    path = 'https://s3.amazonaws.com/ai-datasets/commands_wav.zip'
    datapath = get_file('commands_wav.zip', path)
    try:
        outpath = datapath.replace('.zip', '')
        if os.path.exists(outpath):
            shutil.rmtree(outpath)
        zf = ZipFile(datapath, mode='r', compression=ZIP_DEFLATED)
        zf.extractall(path=outpath + '/../'); zf.close()
    except:
        # remove downloaded zip files
        os.remove(datapath)
        import traceback; traceback.print_exc()
    return outpath
