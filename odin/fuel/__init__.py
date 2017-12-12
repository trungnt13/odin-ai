from __future__ import print_function
import base64
import shutil
from abc import ABCMeta, abstractproperty
from six import add_metaclass

from .utils import *
from .data import *
from .dataset import *
from .feeder import *

from . import recipes

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
    ORIGIN = b'aHR0cHM6Ly9zMy5hbWF6b25hd3MuY29tL2FpLWRhdGFzZXRzLw==\n'
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
    def load(clazz, ext='', override=False):
        return clazz.get_dataset(clazz=clazz, ext=ext, override=override)

    @classmethod
    def get_dataset(clazz, ext='', override=False):
        # ====== all path ====== #
        name = clazz.get_name(ext) + '.zip'
        path = base64.decodestring(DataLoader.ORIGIN).decode() + name
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
    """ 700 MB of data from multiple speaker for digit recognition
    from audio
    """
    pass


class DIGITS(DataLoader):
    """ 501 short file for single digits recognition from audio """
    pass


# ===========================================================================
# More experimental dataset
# ===========================================================================
class IRIS(DataLoader):
    pass


# ===========================================================================
# Others
# ===========================================================================
def load_glove(ndim=100):
    """ Automaticall load a MmapDict which contains the mapping
        (word -> [vector])
    where vector is the embedding vector with given `ndim`.
    """
    ndim = int(ndim)
    if ndim not in (50, 100, 200, 300):
        raise ValueError('Only support 50, 100, 200, 300 dimensions.')
    link = b'aHR0cHM6Ly9zMy5hbWF6b25hd3MuY29tL2FpLWRhdGFzZXRzL2dsb3ZlLjZCLiVkZA==\n'
    link = base64.decodebytes(link) % ndim
    fname = os.path.basename(str(link, 'utf-8'))
    embedding = get_file(fname, link, outdir=get_datasetpath(root='~'))
    return MmapDict(embedding, read_only=True)


class openSMILEsad(DataLoader):
    """ This dataset contains 2 files:
    * lstmvad_rplp18d_12.net
    * rplp18d_norm.dat
    """
    pass


def load_lre_sad():
    link = b'aHR0cHM6Ly9zMy5hbWF6b25hd3MuY29tL2FpLWRhdGFzZXRzL2xyZV9zYWQ=\n'
    link = str(base64.decodebytes(link), 'utf-8')
    path = get_file(fname=os.path.basename(link),
                    origin=link,
                    outdir=get_datasetpath(root='~'))
    return MmapDict(path=path, read_only=True)


def load_lre_list():
    """ The header include following column:
    * name: LDC2017E22/data/ara-acm/ar-20031215-034005_0-a.sph
    * lre: {'train17', 'eval15', 'train15', 'dev17', 'eval17'}
    * language: {'ara-arb', 'eng-sas', 'fre-hat', 'zho-wuu',
                 'eng-gbr', 'ara-ary', 'eng-usg', 'spa-lac',
                 'ara-apc', 'qsl-pol', 'spa-eur', 'fre-waf',
                 'zho-cdo', 'qsl-rus', 'spa-car', 'ara-arz',
                 'zho-cmn', 'por-brz', 'zho-yue', 'zho-nan',
                 'ara-acm'}
    * corpus: {'pcm', 'alaw', 'babel', 'ulaw', 'vast', 'mls14'}
    * duration: {'3', '30', '5', '15', '10', '20', '1000', '25'}

    Note
    ----
    Suggested namming scheme:
      `lre/lang/corpus/dur/base_name`
    """
    link = b'aHR0cHM6Ly9zMy5hbWF6b25hd3MuY29tL2FpLWRhdGFzZXRzL2xyZV9saXN0LnR4dA==\n'
    link = str(base64.decodebytes(link), 'utf-8')
    path = get_file(fname=os.path.basename(link),
                    origin=link,
                    outdir=get_datasetpath(root='~'))
    return np.genfromtxt(fname=path, dtype=str, delimiter=' ',
                         skip_header=1)
