import inspect
from typing import Type

from odin.fuel.audio_data import *
from odin.fuel.bio_data import *
from odin.fuel.databases import *
from odin.fuel.dataset_base import *
from odin.fuel.image_data import *
from odin.fuel.nlp_data import *


def get_dataset(
    name: str,
    **kwargs,
) -> Union[IterableDataset, ImageDataset, GeneDataset, NLPDataset]:
  """Return an instance of `IterableDataset`"""
  name = str(name).strip().lower()
  for key, val in globals().items():
    key = str(key).lower()
    if (key == name and
        inspect.isclass(val) and
        issubclass(val, IterableDataset)):
      val: Type[IterableDataset]
      return val(**kwargs)
  raise ValueError(f"Cannot find dataset with name: {name}")


def get_all_dataset(
    data_type: Literal['image', 'audio', 'text', 'gene']
) -> List[Union[Type[IterableDataset],
                Type[ImageDataset],
                Type[GeneDataset],
                Type[NLPDataset]]]:
  ds = []
  for key, val in globals().items():
    if (inspect.isclass(val) and
        issubclass(val, IterableDataset) and
        val not in (IterableDataset, ImageDataset, NLPDataset, GeneDataset)):
      val: Type[IterableDataset]
      if val.data_type() == data_type:
        ds.append(val)
  return ds
