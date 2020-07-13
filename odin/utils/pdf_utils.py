import os
import re
from collections import defaultdict
from xml.etree import ElementTree

import requests
from six import string_types

_ARXIV_PDF = re.compile(r"^[0-9]{4}\.[0-9]{5}\.(pdf|PDF)")
_ARXIV = re.compile(r"^[0-9]{4}\.[0-9]{5}")


def _to_files(path):
  from odin.utils.python_utils import get_all_files
  if os.path.isdir(path):
    path = get_all_files(path, filter_func=lambda f: '.pdf' in f.lower())
  elif os.path.isfile(path):
    path = [path]
  else:
    raise ValueError(f"{path} does not exist")
  return path


def get_arxiv_titles(article_ids) -> str:
  r""" Get title from arxiv ID or list of arxiv ID (e.g. 2055.12221)
  Return None if not found, otherwise, title of the articles """
  if not isinstance(article_ids, (tuple, list)):
    article_ids = [article_ids]
  article_ids = ','.join([str(i) for i in article_ids if _ARXIV.match(str(i))])
  query = f"http://export.arxiv.org/api/query?id_list={article_ids}"
  title = []
  with requests.get(query) as res:
    res = str(res.content, 'utf-8')
    res = ElementTree.fromstring(res)
    for child in res:
      if "}entry" in child.tag:
        for e in child:
          if "}title" in e.tag:
            title.append(e.text)
  if len(title) == 0:
    return None
  return title[0] if len(title) == 1 else tuple(title)


def get_pdf_text(path: str) -> dict:
  from PyPDF2 import PdfFileReader
  from odin.utils.mpi import MPI

  def read_text(fpath):
    with open(fpath, 'rb') as f:
      f = PdfFileReader(f)
      text = []
      for i in range(f.numPages):
        page = f.getPage(i)
        text.append(page.extractText())
    return (fpath, text)

  results = dict()
  for filepath, text in MPI(jobs=_to_files(path),
                            func=read_text,
                            ncpu=4,
                            batch=1):
    results[filepath] = text
  return results


def get_pdf_titles(path: str) -> dict:
  r""" path : a path to pdf file or a directory contains pdf files """
  from PyPDF2 import PdfFileReader
  from PyPDF2.generic import TextStringObject
  from PyPDF2.pdf import ContentStream
  path2title = dict()
  for filepath in sorted(_to_files(path)):
    filename = '.'.join(os.path.basename(filepath).split('.')[:-1])
    # ARXIV article
    arxiv = _ARXIV.match(filename)
    if arxiv:
      title = get_arxiv_titles(
          filename[arxiv.pos:arxiv.endpos]).strip().replace("\n", "")
      path2title[filepath] = title
      continue
    # others
    with open(filepath, 'rb') as f:
      f = PdfFileReader(f)
      info = f.getDocumentInfo()
      try:
        # decode the title
        if info is not None:
          try:
            title = str(info['/Title'].get_original_bytes(), 'utf-8')
          except UnicodeDecodeError:
            title = str(info['/Title'])
        # check XMP meta
        if info is None or 'untitled' in title:
          meta = f.getXmpMetadata()
          if meta is not None:
            title = meta.dc_title
            if 'untitled' in title or not isinstance(title, string_types):
              title = None
      except KeyError:
        title = None
      if title is None:
        # Check if Outline content title
        # for outline in f.getOutlines():
        #   if '/Title' in outline:
        #     title = outline['/Title']
        #     break
        # might be something in the extracted text
        if title is None:
          # text = f.getPage(0).extractText()
          title = None
    # rename the file
    if title is not None:
      title = title.replace('\n', ' ').replace('/', '.').replace('\\',
                                                                 '.').strip()
      if len(title) == 0:
        title = None
    path2title[filepath] = title
  return path2title


def rename_pdf(path: str, verbose=True):
  r""" Rename all pdf file to its title

  path : a path to pdf file or a directory contains pdf files """
  stats = defaultdict(int)
  all_files = get_pdf_titles(path)
  log_text = []
  for path, title in all_files.items():
    if title is None:
      stats['ignored'] += 1
      if verbose:
        log_text.append(f"Ignore: {path}")
      continue
    dirpath = os.path.dirname(path)
    ext = path.split('.')[-1]
    outpath = os.path.join(dirpath, title + "." + ext)
    if os.path.basename(path) == os.path.basename(outpath):
      stats['matched'] += 1
      if verbose:
        log_text.append(f"Matched: {path}")
    else:
      stats['renamed'] += 1
      if verbose:
        log_text.append(f"Rename: {path} to {os.path.basename(outpath)}")
      os.rename(path, outpath)
  # show logs
  if len(log_text) > 0:
    print("\n".join(sorted(log_text)))
  # summary
  if verbose:
    print("Summary:")
    print(f" * Total  : {len(all_files)} (files)")
    print(f" * Ignored: {stats['ignored']} (files)")
    print(f" * Matched: {stats['matched']} (files)")
    print(f" * Renamed: {stats['renamed']} (files)")
