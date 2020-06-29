from __future__ import absolute_import, division, print_function

import os
import sys
from typing import List, Text
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import tensorflow as tf
from six import string_types
from tqdm import tqdm

from odin.utils.crypto import md5_checksum

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

tf.random.set_seed(1)
np.random.seed(1)

HEADER = [
    'hgnc_id', 'symbol', 'name', 'locus_group', 'locus_type', 'status',
    'location', 'location_sortable', 'alias_symbol', 'alias_name',
    'prev_symbol', 'prev_name', 'gene_family', 'gene_family_id',
    'date_approved_reserved', 'date_symbol_changed', 'date_name_changed',
    'date_modified', 'entrez_id', 'ensembl_gene_id', 'vega_id', 'ucsc_id',
    'ena', 'refseq_accession', 'ccds_id', 'uniprot_ids', 'pubmed_id', 'mgd_id',
    'rgd_id', 'lsdb', 'cosmic', 'omim_id', 'mirbase', 'homeodb', 'snornabase',
    'bioparadigms_slc', 'orphanet', 'pseudogene.org', 'horde_id', 'merops',
    'imgt', 'iuphar', 'kznf_gene_catalog', 'mamit-trnadb', 'cd', 'lncrnadb',
    'enzyme_id', 'intermediate_filament_db', 'rna_central_ids', 'lncipedia',
    'gtrnadb', 'agr'
]
FILTERED_HEADER = [
    'ensembl_gene_id', 'name', 'symbol', 'alias_symbol', 'alias_name',
    'locus_type', 'location', 'cd', 'uniprot_ids', 'enzyme_id'
]
PROTEIN_CODING_TEMPLATE = r"ftp://ftp.ebi.ac.uk/pub/databases/genenames/hgnc/tsv/locus_groups/protein-coding_gene_chr_{chro}.txt"
NON_CODING_TEMPLATE = r"ftp://ftp.ebi.ac.uk/pub/databases/genenames/hgnc/tsv/locus_groups/non-coding_RNA_chr_{chro}.txt"
CHROMOSOMES = list(range(1, 22 + 1)) + ['X', 'Y', 'mitochondria']


def _download_file(chromosome, url, path) -> pd.DataFrame:
  name = os.path.basename(url)
  filename = os.path.join(path, name)
  if not os.path.exists(filename):
    prog = tqdm(desc=f"Download {name}", unit="kB")

    def progress(blocknum, bs, size):
      if prog.total is None:
        prog.total = size // 1024
      prog.update(bs * blocknum // 1024 - prog.n)

    urlretrieve(url=url, filename=filename, reporthook=progress)
    prog.clear()
    prog.close()
  # read the tsv file
  data = []
  with open(filename, 'r') as f:
    for line in f:
      line = [i.replace('"', '') for i in line[:-1].split("\t")]
      data.append(line)
  data = np.asarray(data)
  assert data.shape[1] == 52, \
    f"Expect 52 columns, parsed data has shape:{data.shape}"
  assert np.all(data[0] == HEADER), f"Unknown header: {data[0]}"
  # convert to DataFrame
  data = pd.DataFrame(data[1:], columns=data[0])
  data = data[FILTERED_HEADER]
  # add chromosome
  data['chromosome'] = [str(chromosome).capitalize()] * data.shape[0]
  return data


class HumanGenome:
  r""" The header includes:

    - ensembl_gene_id : 'ENSG00000000003'; 'ENSG00000000005'; ...
    - name            : '1,4-alpha-glucan branching enzyme 1';
                        '1-acylglycerol-3-phosphate O-acyltransferase 1'; ...
    - symbol          : 'A1BG'; 'A1BG-AS1'; ...
    - alias_symbol    : '0808y08y'; '1-8D|DSPA2c'; ...
    - alias_name      : '(S)-2-hydroxy-acid oxidase|glycolate oxidase|long-chain
                        L-2-hydroxy acid oxidase|growth-inhibiting protein 16';
                        '(intestinal Kruppel-like factor'; ...
    - locus_type      : 'RNA, Y'; 'RNA, cluster'; 'RNA, long non-coding';
                        'RNA, micro'; 'RNA, misc'; 'RNA, ribosomal';
                        'RNA, small nuclear'; 'RNA, small nucleolar';
                        'RNA, transfer'; 'RNA, vault'; 'gene with protein product'
    - location        : '1'; '1 not on reference assembly'; ...
    - cd              : 'CD10', 'CD100', 'CD101', 'CD102', 'CD103', 'CD104',
                        'CD105', 'CD106', 'CD107a', 'CD107b', 'CD108', 'CD109',
                        'CD110', 'CD111', 'CD112', 'CD114', 'CD115', 'CD116',
                        'CD117', 'CD118', 'CD119', 'CD11a', 'CD11b', ...
    - uniprot_ids     : 'A0A024RBG1'; 'A0A075B759'
    - enzyme_id       : '1.1.-.-'; '1.1.1.-'; ...
    - chromosome      : '1', '10', '11', '12', '13', '14', '15', '16', '17',
                        '18', '19', '2', '20', '21', '22', '3', '4', '5', '6',
                        '7', '8', '9', 'Mitochondria', 'X', 'Y'


  Reference:
    HGNC: HUGO Gene Nonmenclature Committee
      https://www.genenames.org/download/statistics-and-files/
  """

  def __init__(self, path='~/human_genome'):
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.exists(path):
      os.makedirs(path)
    # mapping from chromosome to file path
    db = []
    for chro in CHROMOSOMES:
      args = dict(chro=chro)
      db.append(
          _download_file(chro, PROTEIN_CODING_TEMPLATE.format(**args), path))
      db.append(_download_file(chro, NON_CODING_TEMPLATE.format(**args), path))
    self.db: pd.DataFrame = pd.concat(db)
    # indexing unique values
    self.unique_index = {
        col: set([i for i in self.db[col].unique() if len(i) > 0
                 ]) for col in self.header
    }

  def unique(self, column_name) -> List[Text]:
    return sorted(self.unique_index[column_name])

  @property
  def header(self):
    return self.db.columns.to_numpy()

  def __getitem__(self, key) -> pd.DataFrame:
    if isinstance(key, (tuple, list, np.ndarray)):
      if not isinstance(key[0], (tuple, list, np.ndarray)):
        key = [key]
      key = dict(key)
    if isinstance(key, string_types):
      col = None
      for i, j in self.unique_index.items():
        if key in j:
          col = i
          break
      if col is None:
        raise ValueError(f"Cannot find gene with info: {key}")
      idx = self.db[col] == key
      return self.db[idx]
    elif isinstance(key, dict):
      db = self.db
      for i, j in key.items():
        db = db[db[str(i)] == str(j)]
      return db
    else:
      raise ValueError(f"key can be dict or string, given: {type(key)}")

  def _get(self, key, column):
    df = self[key]
    assert df.shape[0] == 1, f"Found multiple entries for key='{key}'"
    x = df[str(column)].to_numpy()[0]
    return str(x)

  def get_chromosome(self, key) -> Text:
    return self._get(key, 'chromosome')

  def get_locus_type(self, key) -> Text:
    return self._get(key, 'locus_type')

  def get_protein_cd(self, key) -> Text:
    return self._get(key, 'cd')

  def get_protein_id(self, key) -> Text:
    return self._get(key, 'uniprot_ids')

  def get_gene_symbol(self, key) -> Text:
    return self._get(key, 'symbol')

  def get_gene_id(self, key) -> Text:
    return self._get(key, 'ensembl_gene_id')

  def get_gene_name(self, key) -> Text:
    return self._get(key, 'name')
