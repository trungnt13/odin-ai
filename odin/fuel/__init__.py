from __future__ import print_function
import base64
import shutil

from .utils import *
from .data import *
from .dataset import *
from .feeder import *

import recipes

from zipfile import ZipFile, ZIP_DEFLATED

from odin.utils import get_file, get_script_path, ctext


def unzip_folder(zip_path, out_path, override=False, remove_zip=True):
    if '.zip' not in zip_path:
        raise ValueError(".zip extension must be in the zip_path.")
    try:
        if os.path.exists(out_path):
            if not override:
                raise ValueError("Extracted already exists at: %s" % outpath)
            shutil.rmtree(out_path)
        zf = ZipFile(zip_path, mode='r', compression=ZIP_DEFLATED)
        zf.extractall(path=out_path + '/../')
        zf.close()
    except Exception:
        shutil.rmtree(out_path)
        import traceback; traceback.print_exc()
    finally:
        if remove_zip:
            os.remove(zip_path)


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
