from __future__ import absolute_import, division, print_function

import importlib
import inspect
from typing import List, Optional, Text, Union

from six import string_types
from sklearn.base import BaseEstimator, TransformerMixin

__all__ = ['Transformer']


def _get_model(model, pretrained, **kwargs):
  if "distil" in model:
    model = "Distil" + model.replace("distil", "").capitalize()
  else:
    model = model.capitalize()
  module = importlib.import_module(f"transformers.modeling_{model.lower()}")
  model = getattr(module, f"{model}Model")
  return model.from_pretrained(pretrained, **kwargs)


def _get_tokenizer(model, pretrained, **kwargs):
  pretrained = pretrained.replace("distil", "")
  model = model.replace("distil", "")
  module = importlib.import_module(f"transformers.tokenization_{model.lower()}")
  tokenizer = getattr(module, f"{model.capitalize()}Tokenizer")
  return tokenizer.from_pretrained(pretrained, **kwargs)


# ===========================================================================
# main classes
# ===========================================================================
class Transformer(BaseEstimator, TransformerMixin):
  r"""

  Arguments:
    pretrained : a String. The name of pretrained model,
      some common used models are:
        - distilbert-base-uncased, distilbert-base-cased
        - bert-base-uncased, bert-large-uncased
        - bert-base-cased, bert-large-cased
        - bert-base-multilingual-uncased, bert-base-multilingual-cased
        - bert-large-uncased-whole-word-masking, bert-large-cased-whole-word-masking
        - distilgpt2, gpt2, gpt2-medium, gpt2-large
        - xlnet-base-cased, xlnet-large-cased
        - roberta-base, roberta-large, distilroberta-base
        - ctrl
        - albert-base-v1, albert-large-v1, albert-base-v2, albert-large-v2
        - t5-small


  References:
    List of all pretrained models
      https://huggingface.co/transformers/pretrained_models.html
  """

  def __init__(self,
               pretrained: Text = "albert-base-v2",
               cache_dir=None,
               output_attentions=False,
               output_hidden_states=False):
    super().__init__()
    from transformers.file_utils import TRANSFORMERS_CACHE
    pretrained = str(pretrained).strip().lower()
    model = pretrained.split('-')[0]
    self._tokenizer = _get_tokenizer(model, pretrained)
    self._model = _get_model(model,
                             pretrained,
                             output_attentions=output_attentions,
                             output_hidden_states=output_hidden_states,
                             cache_dir=cache_dir)
    self.cache_dir = TRANSFORMERS_CACHE if cache_dir is None else cache_dir

  def tokenize(self, text: Text):
    assert isinstance(text, string_types)
    return self._tokenizer.tokenize(text)

  def encode(self, text: Union[Text, List[Text]]):
    if isinstance(text, string_types):
      return self._tokenizer.encode(text)
    return self._tokenizer.batch_encode_plus(text)['input_ids']

  def transform(self, text: Union[Text, List[Text]], max_len=None):
    r"""
    Return:
      last_hidden_state: `(batch_size, sequence_length, hidden_size)`
          Sequence of hidden-states at the output of the last layer of the model.
      hidden_states: (optional)
          Tuple of `(batch_size, sequence_length, hidden_size)`.
          (one for the output of the embeddings + one for the output of each layer)
          Hidden-states of the model at the output of each layer plus the initial
          embedding outputs.
      attentions: (optional)
          Tuple `(batch_size, num_heads, sequence_length, sequence_length)`.
          Attentions weights after the attention softmax, used to compute the
          weighted average in the self-attention heads.
    """
    import torch
    if isinstance(text, string_types):
      text = [text]
    inputs = self._tokenizer.batch_encode_plus(text,
                                               return_tensors='pt',
                                               max_len=max_len,
                                               pad_to_max_length=True,
                                               return_token_type_ids=False)
    with torch.no_grad():
      self._model.eval()
      outputs = list(self._model(**inputs))
    # we don't need pooler_output
    if "pooler_output" in inspect.getdoc(self._model.forward):
      outputs.pop(1)
    outputs = [i.numpy() for i in outputs]
    if len(outputs) == 1:
      outputs = outputs[0]
    return outputs

  def fit(self, X, y=None):
    raise NotImplementedError()


class EmojiRecognizer(BaseEstimator, TransformerMixin):
  r""" https://medium.com/huggingface/understanding-emotions-from-keras-to-pytorch-3ccb61d5a983 """
  pass
