from __future__ import print_function, division, absolute_import

import os
import struct
import random
import pickle
import hashlib
import zipfile
from io import BytesIO, StringIO
from six import string_types
from Crypto.Cipher import AES
import numpy as np

# ===========================================================================
# Helper
# ===========================================================================
def to_password(password, salt=None):
  if isinstance(password, string_types):
    password = password.encode('utf-8')
  password += str(salt).encode('utf-8')
  return hashlib.sha256(password).digest()

def _data_to_io(file_or_data):
  own_file = True
  if isinstance(file_or_data, string_types):
    # path to file
    if os.path.exists(file_or_data):
      filesize = os.path.getsize(file_or_data)
      infile = open(file_or_data, 'rb')
    # string object
    else:
      filesize = len(file_or_data)
      infile = StringIO(file_or_data)
  # bytes object
  elif isinstance(file_or_data, bytes):
    filesize = len(file_or_data)
    infile = BytesIO(file_or_data)
  # opened file
  elif hasattr(file_or_data, 'read'):
    filesize = os.fstat(file_or_data.fileno()).st_size
    infile = file_or_data
    own_file = False
  # error input data
  else:
    raise ValueError("No support for MD5 of input type: %s" % str(file_or_data))
  return infile, filesize, own_file

# ===========================================================================
# Hashing
# ===========================================================================
def md5_checksum(file_or_path, chunksize=512 * 1024):
  """ MD5 checksum for:
   * File object
   * File path
   * Bytes array
   * Numpy array
   * List or iterator of numpy array

  """
  hash_md5 = hashlib.md5()
  own_file = False
  # numpy array or list of numpy array
  if isinstance(file_or_path, np.ndarray) or \
  (isinstance(file_or_path, (tuple, list)) and
   all(isinstance(i, np.ndarray) for i in file_or_path)):
    if not isinstance(file_or_path, (tuple, list)):
      file_or_path = (file_or_path,)
    f = BytesIO()
    for arr in file_or_path:
      np.save(file=f, arr=arr, allow_pickle=False)
    f.seek(0)
    own_file = True
  # path to files
  elif isinstance(file_or_path, string_types):
    if os.path.exists(file_or_path):
      f = open(file_or_path, 'rb')
      own_file = True
    else: # just string or text
      hash_md5.update(file_or_path.encode('utf-8'))
      return hash_md5.hexdigest()
  # bytes object directly
  elif isinstance(file_or_path, bytes):
    hash_md5.update(file_or_path)
    return hash_md5.hexdigest()
  # file object or buffer object
  elif hasattr(file_or_path, 'read'):
    f = file_or_path
  else:
    raise ValueError("No support for value: %s" % str(file_or_path))
  # ====== iterate over file ====== #
  for chunk in iter(lambda: f.read(chunksize), b""):
    hash_md5.update(chunk)
  if own_file:
    f.close()
  return hash_md5.hexdigest()

# ===========================================================================
# Encryption
# ===========================================================================
def encrypt_aes(file_or_data, password=None, outfile=None, iv=None, salt=None,
                mode=AES.MODE_CBC, chunksize=512 * 1024):
  """ Flexible implementaiton of AES encryption

  Parameters
  ----------
  file_or_data : {BufferObject, string, bytes}
    input data will be converted to bytes sequence for encryption
  password : {str, None}
    if None, a prompt will ask for inputing password
  outfile : {None, path, file}
    if None, return raw encrypted data
  iv : initial vector
    16 bytes
  salt : {None, string, bytes}
    salt for password Hashing
  mode : Cipher.AES.MODE_*
  chunksize : int
    encryption chunk, multiple of 16.
  """
  if password is None:
    password = input("Your password: ")
  assert len(password) > 0, "Password length must be greater than 0"
  password = to_password(password, salt=salt)
  # Initialization vector
  if iv is None:
    iv = bytes(bytearray(random.randint(0, 0xFF) for i in range(16)))
  encryptor = AES.new(password, mode, IV=iv)
  # ====== check read stream ====== #
  infile, filesize, own_file = _data_to_io(file_or_data)
  # ====== check out stream ====== #
  close_file = False
  if isinstance(outfile, string_types) and os.path.exists(os.path.dirname(outfile)):
    outfile = open(str(outfile), 'wb')
    close_file = True
  elif hasattr(outfile, 'write') and hasattr(outfile, 'flush'):
    close_file = True
  else:
    outfile = BytesIO()
  # ====== some header information ====== #
  outfile.write(struct.pack('<Q', filesize))
  outfile.write(iv)
  while True:
    chunk = infile.read(chunksize)
    if len(chunk) == 0:
      break
    elif len(chunk) % 16 != 0:
      chunk += b' ' * (16 - len(chunk) % 16)
    outfile.write(encryptor.encrypt(chunk))
  # ====== clean and return ====== #
  if own_file:
    infile.close()
  outfile.flush()
  if close_file:
    outfile.close()
  else:
    outfile.seek(0)
    data = outfile.read()
    outfile.close()
    return data

def decrypt_aes(file_or_data, password=None, outfile=None, salt=None,
                mode=AES.MODE_CBC, chunksize=512 * 1024):
  """ Flexible implementaiton of AES decryption

  Parameters
  ----------
  file_or_data : {BufferObject, string, bytes}
    input data will be converted to bytes sequence for encryption
  password : {str, None}
    if None, a prompt will ask for inputing password
  outfile : {None, path, file}
    if None, return raw encrypted data
  salt : {None, string, bytes}
    salt for password Hashing
  mode : Cipher.AES.MODE_*
  chunksize : int
    encryption chunk, multiple of 16.
  """
  if password is None:
    password = input("Your password: ")
  assert len(password) > 0, "Password length must be greater than 0"
  password = to_password(password, salt)
  # ====== read header ====== #
  infile, filesize, own_file = _data_to_io(file_or_data)
  origsize = struct.unpack('<Q', infile.read(struct.calcsize('Q')))[0]
  iv = infile.read(16)
  decryptor = AES.new(password, mode=AES.MODE_CBC, IV=iv)
  # ====== outfile ====== #
  close_file = False
  if isinstance(outfile, string_types) and os.path.exists(os.path.dirname(outfile)):
    outfile = open(str(outfile), 'wb')
    close_file = True
  elif hasattr(outfile, 'write') and hasattr(outfile, 'flush'):
    close_file = True
  else:
    outfile = BytesIO()
  # ====== decryption ====== #
  while True:
    chunk = infile.read(chunksize)
    if len(chunk) == 0:
      break
    outfile.write(decryptor.decrypt(chunk))
  outfile.truncate(origsize)
  # ====== clean and return ====== #
  if own_file:
    infile.close()
  outfile.flush()
  if close_file:
    outfile.close()
  else:
    outfile.seek(0)
    data = outfile.read()
    outfile.close()
    return data

# ===========================================================================
# Zip
# ===========================================================================
def zip_aes(in_path, out_path, password=None, verbose=False):
  """
  Parameters
  ----------
  in_path : string
    path to a folder
  out_path : string
    path to output zip file
  """
  if password is None:
    password = input("Your password:")
  password = str(password)
  assert len(password) > 0, "`password`=%s length must be greater than 0" % password
  # ====== prepare input ====== #
  from odin.utils import get_all_files
  if not os.path.isdir(in_path):
    raise ValueError("`in_path` to %s is not a folder" % str(in_path))
  all_files = get_all_files(in_path)
  # ====== prepare output ====== #
  if not isinstance(out_path, string_types):
    raise ValueError("`out_path` must be string")
  f = zipfile.ZipFile(out_path, 'w', compression=zipfile.ZIP_STORED, allowZip64=True)
  # ====== compression ====== #
  md5_map = {}
  for path in all_files:
    name = os.path.basename(path)
    md5_map[name] = md5_checksum(path)
    f.writestr(name, encrypt_aes(path, password + name))
    if verbose:
      print('Compressed: "%s"' % name, "(MD5:%s)" % md5_map[name])
  f.writestr('_MD5_CHECKSUM_', pickle.dumps(md5_map))
  f.close()

def unzip_aes(in_path, out_path, password=None, verbose=False):
  """
  Parameters
  ----------
  in_path : string
    path to input zip file
  out_path : string
    path to output parent folder
  """
  if password is None:
    password = input("Your password:")
  password = str(password)
  assert len(password) > 0, "`password`=%s length must be greater than 0" % password
  # ====== check in_path ====== #
  if not os.path.isfile(in_path):
    raise ValueError("`in_path` to %s is not a file" % str(in_path))
  # ====== prepare output ====== #
  if os.path.isfile(out_path):
    raise ValueError("`out_path` must be a folder")
  elif not os.path.exists(out_path):
    os.mkdir(out_path)
  # ====== decompress ====== #
  with zipfile.ZipFile(in_path, 'r', compression=zipfile.ZIP_STORED, allowZip64=True) as fzip:
    md5_map = pickle.loads(fzip.read(name='_MD5_CHECKSUM_'))
    for name in fzip.namelist():
      if '_MD5_CHECKSUM_' == name:
        continue
      data = fzip.read(name=name)
      data = decrypt_aes(data, password=password + name)
      md5 = md5_checksum(data)
      assert md5 == md5_map[name], "MD5 mismatch for data name: '%s'" % name
      # save file to disk
      path = out_path
      for d in name.split('/')[:-1]:
        path += '/' + d
        if os.path.exists(path):
          os.mkdir(path)
      with open(os.path.join(path, name.split('/')[-1]), mode='wb') as f:
        f.write(data)
      if verbose:
        print('Decompressed: "%s"' % name, "(MD5:%s)" % md5)
