from typing import Type

from odin.fuel.audio_data import *
from odin.fuel.bio_data import *
from odin.fuel.databases import *
from odin.fuel.dataset import *
from odin.fuel.dataset_base import *
from odin.fuel.image_data import *
from odin.fuel.loaders import *
from odin.fuel.nlp_data import *


def get_dataset(name: str) -> Type[IterableDataset]:
  import inspect
  name = str(name).strip().lower()
  for key, val in globals().items():
    key = str(key).lower()
    if key == name and inspect.isclass(val):
      return val
  raise ValueError(f"Cannot find dataset with name: {name}")
