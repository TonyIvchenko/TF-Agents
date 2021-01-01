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
"""Tests for tf_agents.benchmark.utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.benchmark import utils


class TimeHistoryTest(tf.test.TestCase):

  def testRejectsMissingBatchSize(self):
    with self.assertRaisesRegex(ValueError, 'batch_size'):
      utils.TimeHistory(batch_size=None, log_steps=10)

  def testRejectsNonPositiveBatchSize(self):
    with self.assertRaisesRegex(ValueError, 'batch_size'):
      utils.TimeHistory(batch_size=0, log_steps=10)

  def testRecordsTimestampOnFirstBatch(self):
    history = utils.TimeHistory(batch_size=32, log_steps=10)

    history.on_batch_begin()

    self.assertEqual(1, len(history.timestamp_log))
    self.assertEqual(1, history.timestamp_log[0].batch_index)


if __name__ == '__main__':
  tf.test.main()
