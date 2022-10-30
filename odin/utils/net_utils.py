import os
import shutil
import tarfile
import zipfile

import requests
from six.moves.urllib.error import HTTPError, URLError
from six.moves.urllib.request import urlretrieve

from odin.utils.crypto import md5_checksum, md5_folder

__all__ = [
    'download_and_extract',
    'download_google_drive',
    'get_file',
]


def download_google_drive(id, destination, chunk_size=32 * 1024):
  r""" Original code for dowloading the dataset:
  https://github.com/suvojit-0x55aa/celebA-HQ-dataset-download """
  from tqdm import qdm
  URL = "https://docs.google.com/uc?export=download"
  URL = "https://drive.google.com/uc?id="
  session = requests.Session()
  response = session.get(URL, params={'id': id}, stream=True)
  token = None
  for key, value in response.cookies.items():
    if key.startswith('download_warning'):
      token = value
  if token:
    params = {'id': id, 'confirm': token}
    response = session.get(URL, params=params, stream=True)
  total_size = int(response.headers.get('content-length', 0))
  with open(destination, "wb") as f:
    for chunk in tqdm(response.iter_content(chunk_size),
                      total=total_size,
                      unit='B',
                      unit_scale=True,
                      desc=destination):
      if chunk:  # filter out keep-alive new chunks
        f.write(chunk)


def download_and_extract(path,
                         url,
                         extract=True,
                         md5_download=None,
                         md5_extract=None):
  r""" Download a file to given path then extract the file

  Arguments:
    path : a String path to a folder
    url : a String of download URL
    extract : a Boolean, if True decompress the file
  """
  from tqdm import tqdm
  path = os.path.abspath(os.path.expanduser(path))
  if not os.path.exists(path):
    os.makedirs(path)
  assert os.path.isdir(path), "path to '%s' is not a directory" % path
  ### file name
  filename = url.split('/')[-1]
  filepath = os.path.join(path, filename)
  ### download
  if os.path.exists(filepath) and md5_download is not None:
    md5 = md5_checksum(filepath)
    if md5 != md5_download:
      print("MD5 of downloaded file mismatch! downloaded:%s  provided:%s" %
            (md5, md5_download))
      os.remove(filepath)
  if not os.path.exists(filepath):
    prog = tqdm(desc="Download '%s'" % filename, total=-1, unit="MB")

    def _progress(count, block_size, total_size):
      # to MB
      total_size = total_size / 1024. / 1024.
      block_size = block_size / 1024. / 1024.
      if prog.total < 0:
        prog.total = total_size
      prog.update(block_size)

    filepath, _ = urlretrieve(url, filepath, reporthook=_progress)
  ### no extraction needed
  if not extract:
    return filepath
  ### extract
  extract_path = os.path.join(path, os.path.basename(filename).split('.')[0])
  if os.path.exists(extract_path) and md5_extract is not None:
    md5 = md5_folder(extract_path)
    if md5 != md5_extract:
      print("MD5 extracted folder mismatch! extracted:%s provided:%s" %
            (md5, md5_extract))
      shutil.rmtree(extract_path)
  if not os.path.exists(extract_path):
    # .tar.gz
    if '.tar.gz' in filepath:
      with tarfile.open(filepath, 'r:gz') as f:
        print("Extracting files ...")
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(f, path)
    # .zip
    elif '.zip' in filepath:
      # TODO
      raise NotImplementedError
    # unknown extension
    else:
      raise NotImplementedError("Cannot extract file: %s" % filepath)
  ### return
  return path, extract_path


def get_file(fname, origin, outdir, verbose=False):
  r'''
  Arguments:
    fname: output file name
    origin: url, link
    outdir: path to output dir
  '''
  fpath = os.path.join(outdir, fname)
  # ====== remove empty folder ====== #
  if os.path.exists(fpath):
    if os.path.isdir(fpath) and len(os.listdir(fpath)) == 0:
      shutil.rmtree(fpath)
  # ====== download package ====== #
  if not os.path.exists(fpath):
    if verbose:
      prog = Progbar(target=-1,
                     name="Downloading: %s" % os.path.basename(origin),
                     print_report=True,
                     print_summary=True)

    def dl_progress(count, block_size, total_size):
      if verbose:
        if prog.target < 0:
          prog.target = total_size
        else:
          prog.add(count * block_size - prog.seen_so_far)

    ###
    error_msg = 'URL fetch failure on {}: {} -- {}'
    try:
      try:
        urlretrieve(origin, fpath, dl_progress)
      except URLError as e:
        raise Exception(error_msg.format(origin, e.errno, e.reason))
      except HTTPError as e:
        raise Exception(error_msg.format(origin, e.code, e.msg))
    except (Exception, KeyboardInterrupt) as e:
      if os.path.exists(fpath):
        os.remove(fpath)
      raise
  return fpath
