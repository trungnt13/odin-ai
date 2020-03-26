from odin.fuel.audio_data import *
from odin.fuel.databases import *
from odin.fuel.dataset import *
from odin.fuel.image_data import *
from odin.fuel.loaders import *

def get_dataset(name):
  import inspect
  name = str(name).strip().lower()
  for key, val in globals().items():
    key = str(key).lower()
    if key == name and inspect.isclass(val):
      return val
  raise ValueError("Cannot find dataset with name: %s" % name)
