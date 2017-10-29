from __future__ import print_function
import base64
import shutil
from abc import ABCMeta, abstractproperty
from six import add_metaclass

from .utils import *
from .data import *
from .dataset import *
from .feeder import *

import recipes

from zipfile import ZipFile, ZIP_DEFLATED

from odin.utils import get_file, get_script_path, ctext, get_datasetpath


# ===========================================================================
# Helper
# ===========================================================================
def unzip_folder(zip_path, out_path, remove_zip=True):
    if '.zip' not in zip_path:
        raise ValueError(".zip extension must be in the zip_path.")
    if not os.path.exists(zip_path):
        raise ValueError("Cannot find zip file at path: %s" % zip_path)
    try:
        zf = ZipFile(zip_path, mode='r', compression=ZIP_DEFLATED)
        zf.extractall(path=out_path)
        zf.close()
    except Exception:
        shutil.rmtree(out_path)
        import traceback; traceback.print_exc()
    finally:
        if remove_zip:
            os.remove(zip_path)


@add_metaclass(ABCMeta)
class DataLoader(object):
    PATH = 'aHR0cHM6Ly9zMy5hbWF6b25hd3MuY29tL2FpLWRhdGFzZXRzLw==\n'
    BASE_DIR = get_datasetpath(root='~')

    def __init__(self):
        super(DataLoader, self).__init__()

    @classmethod
    def get_name(clazz, ext=''):
        name = clazz.__name__
        name = name if ext is None or len(ext) == 0 \
            else '_'.join([name, ext])
        return name

    @classmethod
    def get_zip_path(clazz, ext=''):
        return os.path.join(DataLoader.BASE_DIR,
                            clazz.get_name(ext) + '.zip')

    @classmethod
    def get_ds_path(clazz, ext=''):
        return os.path.join(DataLoader.BASE_DIR, clazz.get_name(ext))

    @classmethod
    def get_dataset(clazz, ext='', override=False):
        # ====== all path ====== #
        name = clazz.get_name(ext) + '.zip'
        path = base64.decodestring(DataLoader.PATH) + name
        zip_path = clazz.get_zip_path(ext)
        out_path = clazz.get_ds_path(ext)
        # ====== check out_path ====== #
        if os.path.isfile(out_path):
            raise RuntimeError("Found a file at path: %s, we need a folder "
                               "to unzip downloaded files." % out_path)
        elif os.path.isdir(out_path):
            if override or len(os.listdir(out_path)) == 0:
                shutil.rmtree(out_path)
            else:
                return Dataset(out_path, read_only=True)
        # ====== download the file ====== #
        if os.path.exists(zip_path) and override:
            os.remove(zip_path)
        if not os.path.exists(zip_path):
            get_file(name, path, DataLoader.BASE_DIR)
        # ====== upzip dataset ====== #
        unzip_folder(zip_path, out_path, remove_zip=True)
        return Dataset(out_path, read_only=True)


# ===========================================================================
# Images dataset
# ===========================================================================
class MNIST(DataLoader):
    pass


class CIFAR10(DataLoader):
    pass


class CIFAR100(DataLoader):
    pass


# ===========================================================================
# AUdio dataset
# ===========================================================================
class WDIGITS(DataLoader):
    pass


class COMMANDS(DataLoader):
    pass

# ===========================================================================
# Others
# ===========================================================================


class IRIS(DataLoader):
    pass


def load_imdb(nb_words=None, maxlen=None):
    """ The preprocessed imdb dataset with following configuraiton:
     - nb_words=88587
     - length=2494
     - NO skip for any top popular word
     - Word_IDX=1 for beginning of sequences
     - Word_IDX=2 for ignored word (OOV)
     - Other word start from 3
     - padding='pre' with value=0
    """
    path = 'https://s3.amazonaws.com/ai-datasets/imdb.zip'
    datapath = get_file('imdb', path)
    ds = _load_data_from_path(datapath)
    X_train, y_train, X_test, y_test = \
        ds['X_train'], ds['y_train'], ds['X_test'], ds['y_test']
    # create new data with new configuration
    if maxlen is not None or nb_words is not None:
        nb_words = max(min(88587, nb_words), 3)
        path = ds.path + '_tmp'
        if os.path.exists(path):
            shutil.rmtree(path)
        ds = Dataset(path)
        # preprocess data
        if maxlen is not None:
            # for X_train
            _X, _y = [], []
            for i, j in zip(X_train[:], y_train[:]):
                if i[-maxlen] == 0 or i[-maxlen] == 1:
                    _X.append([k if k < nb_words else 2 for k in i[-maxlen:]])
                    _y.append(j)
            X_train = np.array(_X, dtype=X_train.dtype)
            y_train = np.array(_y, dtype=y_train.dtype)
            # for X_test
            _X, _y = [], []
            for i, j in zip(X_test[:], y_test[:]):
                if i[-maxlen] == 0 or i[-maxlen] == 1:
                    _X.append([k if k < nb_words else 2 for k in i[-maxlen:]])
                    _y.append(j)
            X_test = np.array(_X, dtype=X_test.dtype)
            y_test = np.array(_y, dtype=y_test.dtype)
        ds['X_train'] = X_train
        ds['y_train'] = y_train
        ds['X_test'] = X_test
        ds['y_test'] = y_test
        ds.flush()
    return ds


def load_digit_feat():
    path = 'https://s3.amazonaws.com/ai-datasets/digit.zip'
    name = 'digit'
    datapath = get_file(name, path)
    return _load_data_from_path(datapath)


def load_sad_model():
    """ Return path to pretrained openSMILE LSTM SAD model
    include: LSTM model, and init data """
    link = r'aHR0cHM6Ly9zMy5hbWF6b25hd3MuY29tL2FpLWRhdGFzZXRzL3ZhZF9tb2RlbC56aXA=\n'
    link = base64.decodestring(link)
    path = get_file('vad_model.zip', link)
    outpath = path.replace('.zip', '')
    if not os.path.exists(outpath):
        unzip_folder(path, outpath, override=True, remove_zip=True)
        print("Downloaded and extracted at:", ctext(outpath, 'magenta'))
    lstm_path = os.path.join(outpath, 'lstmvad_rplp18d_12.net')
    init_path = os.path.join(outpath, 'rplp18d_norm.dat')
    return lstm_path, init_path


def load_swb1_aligment(nb_senones=2304):
    support_nb_senones = (2304,)
    if nb_senones not in support_nb_senones:
        raise ValueError('We only support following number of senones: %s'
                        % support_nb_senones)
    url = 'aHR0cHM6Ly9zMy5hbWF6b25hd3MuY29tL2FpLWRhdGFzZXRzL3N3YjFfJWQ=\n'
    url = base64.decodestring(url) % nb_senones
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
    link = 'aHR0cHM6Ly9zMy5hbWF6b25hd3MuY29tL2FpLWRhdGFzZXRzL2dsb3ZlLjZCLiVkZA==\n'
    link = base64.decodestring(link) % ndim
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
    path = 'aHR0cHM6Ly9zMy5hbWF6b25hd3MuY29tL2FpLWRhdGFzZXRzL2RpZ2l0X3dhdi56aXA=\n'
    path = base64.decodestring(path)
    datapath = get_file('digit_wav.zip', path)
    try:
        outpath = datapath.replace('.zip', '')
        if os.path.exists(outpath):
            shutil.rmtree(outpath)
        zf = ZipFile(datapath, mode='r', compression=ZIP_DEFLATED)
        zf.extractall(path=outpath + '/../'); zf.close()
    except Exception:
        # remove downloaded zip files
        os.remove(datapath)
        import traceback; traceback.print_exc()
    return Dataset(outpath, read_only=True)


def load_digit_raw():
    path = 'aHR0cHM6Ly9zMy5hbWF6b25hd3MuY29tL2FpLWRhdGFzZXRzL2RpZ2l0X3Jhdy56aXA=\n'
    path = base64.decodestring(path)
    datapath = get_file('digit_raw.zip', path)
    try:
        outpath = datapath.replace('.zip', '')
        if os.path.exists(outpath):
            shutil.rmtree(outpath)
        zf = ZipFile(datapath, mode='r', compression=ZIP_DEFLATED)
        zf.extractall(path=outpath + '/../')
        zf.close()
    except Exception:
        # remove downloaded zip files
        os.remove(datapath)
        import traceback; traceback.print_exc()
    finally:
        os.remove(datapath)
    return Dataset(outpath, read_only=True)


def load_commands_wav():
    path = 'aHR0cHM6Ly9zMy5hbWF6b25hd3MuY29tL2FpLWRhdGFzZXRzL2NvbW1hbmRzX3dhdi56aXA=\n'
    path = base64.decodestring(path)
    datapath = get_file('commands_wav.zip', path)
    try:
        outpath = datapath.replace('.zip', '')
        if os.path.exists(outpath):
            shutil.rmtree(outpath)
        zf = ZipFile(datapath, mode='r', compression=ZIP_DEFLATED)
        zf.extractall(path=outpath + '/../'); zf.close()
    except Exception:
        # remove downloaded zip files
        os.remove(datapath)
        import traceback; traceback.print_exc()
    return outpath
