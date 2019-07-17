#copyright 2019 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data generators for depression  data-set."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tarfile
from six.moves import range  # pylint: disable=redefined-builtin

from shutil import copyfile
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import text_problems
from tensor2tensor.data_generators import lm1b
from tensor2tensor.utils import registry

import tensorflow as tf


def _train_data_filenames(tmp_dir):
  return [
      os.path.join(tmp_dir,
                   "training.txt")
  ]

def _dev_data_filenames(tmp_dir):
  return [os.path.join(tmp_dir,
                       "testing.txt")]

def _maybe_fetch_corpus(tmp_dir):
  """Download and unpack the corpus.
  Args:
    tmp_dir: directory containing dataset.
  """
  corpus_filename = 'training.txt'
  corpus_filename2 = 'testing.txt'
  corpus_filepath = os.path.join(tmp_dir, corpus_filename)
  corpus_filepath2 = os.path.join(tmp_dir, corpus_filename2)
  if not os.path.exists(corpus_filepath):
      print('creating data set')
      copyfile('/home/sven/Desktop/Research/REU_2019/code/clpsych/data/lm-training-text/control_text_filtered2.txt', corpus_filepath)
      copyfile('/home/sven/Desktop/Research/REU_2019/code/clpsych/data/lm-testing-text/control_text_filtered2.txt', corpus_filepath2)

@registry.register_problem
class LanguagemodelDp(lm1b.LanguagemodelLm1b32k):
  """A language model on the 1B words corpus.
  Ratio of dev tokens (including eos) to dev words (including eos)
  176923 / 159658 = 1.108137; multiply log_ppl by this to compare results.
  """
  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    del data_dir
    split_files = {
        problem.DatasetSplit.TRAIN: self._train_data_filenames(tmp_dir),
        problem.DatasetSplit.EVAL: self._dev_data_filenames(tmp_dir),
    }
    _maybe_fetch_corpus(tmp_dir)
    original_vocab = lm1b._original_vocab(tmp_dir)
    files = split_files[dataset_split]
    for filepath in files:
      tf.logging.info("filepath = %s", filepath)
      for line in tf.gfile.Open(filepath):
        txt = lm1b._replace_oov(original_vocab, text_encoder.native_to_unicode(line))
        yield {"targets": txt}
