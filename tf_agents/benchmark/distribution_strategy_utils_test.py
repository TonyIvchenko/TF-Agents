# coding=utf-8
# Copyright 2020 The TF-Agents Authors.
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

# Lint as: python2, python3
"""Tests for tf_agents.benchmark.distribution_strategy_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.benchmark import distribution_strategy_utils


class DistributionStrategyUtilsTest(tf.test.TestCase):

  def test_unknown_strategy_raises_value_error(self):
    with self.assertRaisesRegex(ValueError, 'Unrecognized distribution_strategy'):
      distribution_strategy_utils.get_distribution_strategy('bogus_strategy')

  def test_negative_num_gpus_raises_value_error(self):
    with self.assertRaisesRegex(ValueError, 'num_gpus'):
      distribution_strategy_utils.get_distribution_strategy(num_gpus=-1)

  def test_off_strategy_with_single_gpu_returns_none(self):
    strategy = distribution_strategy_utils.get_distribution_strategy(
        distribution_strategy='off', num_gpus=1)
    self.assertIsNone(strategy)


if __name__ == '__main__':
  tf.test.main()
