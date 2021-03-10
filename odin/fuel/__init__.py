import inspect
from typing import Optional, Type, List
from typing_extensions import Literal

from odin.fuel.audio_data import *
from odin.fuel.bio_data import *
from odin.fuel.databases import *
from odin.fuel.dataset_base import *
from odin.fuel.image_data import *
from odin.fuel.nlp_data import *


def get_dataset(
    name: str,
    **dataset_kwargs,
) -> IterableDataset:
  """Return an instance of `IterableDataset`"""
  name = str(name).strip().lower()
  for key, val in globals().items():
    key = str(key).lower()
    if key == name and \
      inspect.isclass(val) and \
        issubclass(val, IterableDataset):
      return val(**dataset_kwargs)
  raise ValueError(f"Cannot find dataset with name: {name}")


def get_all_dataset(
    data_type: Literal['image', 'audio', 'text', 'gene']
) -> List[Type[IterableDataset]]:
  ds = []
  for key, val in globals().items():
    if inspect.isclass(val) and \
      issubclass(val, IterableDataset) and \
        val not in (IterableDataset, ImageDataset, NLPDataset, GeneDataset) and \
        val.data_type() == data_type:
      ds.append(val)
  return ds
