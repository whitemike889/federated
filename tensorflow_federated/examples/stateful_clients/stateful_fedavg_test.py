# Copyright 2020, The TensorFlow Federated Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""End-to-end example testing the FedAvg algorithm with stateful clients."""

import collections
import functools
from typing import Callable, List, OrderedDict

from absl.testing import parameterized
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

from tensorflow_federated.examples.stateful_clients import stateful_fedavg_tf
from tensorflow_federated.examples.stateful_clients import stateful_fedavg_tff


def _create_test_cnn_model():
  """A simple CNN model for test."""
  data_format = 'channels_last'
  input_shape = [28, 28, 1]

  max_pool = functools.partial(
      tf.keras.layers.MaxPooling2D,
      pool_size=(2, 2),
      padding='same',
      data_format=data_format)
  conv2d = functools.partial(
      tf.keras.layers.Conv2D,
      kernel_size=5,
      padding='same',
      data_format=data_format,
      activation=tf.nn.relu)

  model = tf.keras.models.Sequential([
      conv2d(filters=32, input_shape=input_shape),
      max_pool(),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(10),
      tf.keras.layers.Activation(tf.nn.softmax),
  ])

  return model


def _create_random_batch():
  return collections.OrderedDict(
      x=tf.random.uniform(tf.TensorShape([1, 28, 28, 1]), dtype=tf.float32),
      y=tf.constant(1, dtype=tf.int32, shape=[1]))


def _create_one_client_state():
  return stateful_fedavg_tf.ClientState(client_index=-1, iters_count=0)


def _simple_model_fn():
  keras_model = _create_test_cnn_model()
  loss = tf.keras.losses.SparseCategoricalCrossentropy()
  input_spec = collections.OrderedDict(
      x=tf.TensorSpec([None, 28, 28, 1], tf.float32),
      y=tf.TensorSpec([None], tf.int32))
  return stateful_fedavg_tf.KerasModelWrapper(
      keras_model=keras_model, input_spec=input_spec, loss=loss)


def _tff_learning_model_fn():
  keras_model = _create_test_cnn_model()
  loss = tf.keras.losses.SparseCategoricalCrossentropy()
  input_spec = collections.OrderedDict(
      x=tf.TensorSpec([None, 28, 28, 1], tf.float32),
      y=tf.TensorSpec([None], tf.int32))
  return tff.learning.from_keras_model(
      keras_model=keras_model, input_spec=input_spec, loss=loss)


MnistVariables = collections.namedtuple(
    'MnistVariables', 'weights bias num_examples loss_sum accuracy_sum')


def _create_mnist_variables():
  return MnistVariables(
      weights=tf.Variable(
          lambda: tf.zeros(dtype=tf.float32, shape=(784, 10)),
          name='weights',
          trainable=True),
      bias=tf.Variable(
          lambda: tf.zeros(dtype=tf.float32, shape=(10)),
          name='bias',
          trainable=True),
      num_examples=tf.Variable(0.0, name='num_examples', trainable=False),
      loss_sum=tf.Variable(0.0, name='loss_sum', trainable=False),
      accuracy_sum=tf.Variable(0.0, name='accuracy_sum', trainable=False))


def _mnist_inference(variables, batch):
  logits = tf.nn.softmax(
      tf.matmul(batch['x'], variables.weights) + variables.bias)
  predictions = tf.cast(tf.argmax(logits, 1), tf.int32)
  return logits, predictions


def _mnist_forward_pass(variables, batch):
  y, predictions = _mnist_inference(variables, batch)

  flat_labels = tf.reshape(batch['y'], [-1])
  loss = -tf.reduce_mean(
      tf.reduce_sum(tf.one_hot(flat_labels, 10) * tf.math.log(y), axis=[1]))
  accuracy = tf.reduce_mean(
      tf.cast(tf.equal(predictions, flat_labels), tf.float32))

  num_examples = tf.cast(tf.size(batch['y']), tf.float32)

  variables.num_examples.assign_add(num_examples)
  variables.loss_sum.assign_add(loss * num_examples)
  variables.accuracy_sum.assign_add(accuracy * num_examples)

  return tff.learning.BatchOutput(
      loss=loss, predictions=predictions, num_examples=num_examples)


def _get_local_mnist_metrics(variables):
  return collections.OrderedDict(
      num_examples=variables.num_examples,
      loss=variables.loss_sum / variables.num_examples,
      accuracy=variables.accuracy_sum / variables.num_examples)


@tff.federated_computation
def _aggregate_mnist_metrics_across_clients(metrics):
  return collections.OrderedDict(
      num_examples=tff.federated_sum(metrics.num_examples),
      loss=tff.federated_mean(metrics.loss, metrics.num_examples),
      accuracy=tff.federated_mean(metrics.accuracy, metrics.num_examples))


class MnistModel(tff.learning.Model):

  def __init__(self):
    self._variables = _create_mnist_variables()

  @property
  def trainable_variables(self):
    return [self._variables.weights, self._variables.bias]

  @property
  def non_trainable_variables(self):
    return []

  @property
  def weights(self):
    return tff.learning.ModelWeights(
        trainable=self.trainable_variables,
        non_trainable=self.non_trainable_variables)

  @property
  def local_variables(self):
    return [
        self._variables.num_examples, self._variables.loss_sum,
        self._variables.accuracy_sum
    ]

  @property
  def input_spec(self):
    return collections.OrderedDict(
        x=tf.TensorSpec([None, 784], tf.float32),
        y=tf.TensorSpec([None, 1], tf.int32))

  @tf.function
  def predict_on_batch(self, batch, training=True):
    del training  # Unused.
    return _mnist_inference(self._variables, batch)

  @tf.function
  def forward_pass(self, batch, training=True):
    del training  # Unused.
    return _mnist_forward_pass(self._variables, batch)

  @tf.function
  def report_local_outputs(self):
    return _get_local_mnist_metrics(self._variables)

  @property
  def federated_output_computation(self):
    return _aggregate_mnist_metrics_across_clients

  @tf.function
  def report_local_unfinalized_metrics(
      self) -> OrderedDict[str, List[tf.Tensor]]:
    """Creates an `OrderedDict` of metric names to unfinalized values."""
    return collections.OrderedDict(
        num_examples=[self._variables.num_examples],
        loss=[self._variables.loss_sum, self._variables.num_examples],
        accuracy=[self._variables.accuracy_sum, self._variables.num_examples])

  def metric_finalizers(
      self) -> OrderedDict[str, Callable[[List[tf.Tensor]], tf.Tensor]]:
    """Creates an `OrderedDict` of metric names to finalizers."""
    return collections.OrderedDict(
        num_examples=tf.function(func=lambda x: x[0]),
        loss=tf.function(func=lambda x: x[0] / x[1]),
        accuracy=tf.function(func=lambda x: x[0] / x[1]))


def _create_client_data():
  emnist_batch = collections.OrderedDict(
      label=[[5]], pixels=[np.random.rand(28, 28).astype(np.float32)])
  dataset = tf.data.Dataset.from_tensor_slices(emnist_batch)

  def client_data(num_batch=1):
    return tff.simulation.models.mnist.keras_dataset_from_emnist(
        dataset).repeat(2 * num_batch).batch(2)

  return client_data


class FedAvgTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('simple_model_wrapper', _simple_model_fn),
      ('tff.learning.Model_wrapper', _tff_learning_model_fn))
  def test_something(self, model_fn):
    it_process = stateful_fedavg_tff.build_federated_averaging_process(
        model_fn, _create_one_client_state)
    self.assertIsInstance(it_process, tff.templates.IterativeProcess)
    federated_data_type = it_process.next.type_signature.parameter[1]
    self.assertEqual(
        str(federated_data_type),
        '{<x=float32[?,28,28,1],y=int32[?]>*}@CLIENTS')

  @parameterized.named_parameters(
      ('simple_model_wrapper', _simple_model_fn),
      ('tff.learning.Model_wrapper', _tff_learning_model_fn))
  def test_simple_training(self, model_fn):
    it_process = stateful_fedavg_tff.build_federated_averaging_process(
        model_fn, _create_one_client_state)
    server_state = it_process.initialize()

    def deterministic_batch():
      return collections.OrderedDict(
          x=np.ones([1, 28, 28, 1], dtype=np.float32),
          y=np.ones([1], dtype=np.int32))

    batch = tff.tf_computation(deterministic_batch)()
    federated_data = [[batch]]
    client_states = [_create_one_client_state()]

    loss_list = []
    for _ in range(3):
      server_state, loss, client_states = it_process.next(
          server_state, federated_data, client_states)
      loss_list.append(loss)

    self.assertEqual(server_state.total_iters_count,
                     client_states[0].iters_count)
    self.assertLess(np.mean(loss_list[1:]), loss_list[0])

  def test_self_contained_example_custom_model(self):

    client_data = _create_client_data()
    train_data = [client_data(), client_data(2)]
    client_states = [_create_one_client_state(), _create_one_client_state()]

    trainer = stateful_fedavg_tff.build_federated_averaging_process(
        MnistModel, _create_one_client_state)
    state = trainer.initialize()
    losses = []
    for _ in range(2):
      state, loss, client_states = trainer.next(state, train_data,
                                                client_states)
      losses.append(loss)
    self.assertEqual(
        state.total_iters_count,
        client_states[0].iters_count + client_states[1].iters_count)
    self.assertLess(losses[1], losses[0])

  def test_keras_evaluate(self):
    keras_model = _create_test_cnn_model()
    sample_data = [
        collections.OrderedDict(
            x=np.ones([1, 28, 28, 1], dtype=np.float32),
            y=np.ones([1], dtype=np.int32))
    ]
    metric = tf.keras.metrics.SparseCategoricalAccuracy()
    accuracy = stateful_fedavg_tf.keras_evaluate(keras_model, sample_data,
                                                 metric)
    self.assertIsInstance(accuracy, tf.Tensor)
    self.assertBetween(accuracy, 0.0, 1.0)

  def test_tff_learning_evaluate(self):
    it_process = stateful_fedavg_tff.build_federated_averaging_process(
        _tff_learning_model_fn, _create_one_client_state)
    server_state = it_process.initialize()
    sample_data = [
        collections.OrderedDict(
            x=np.ones([1, 28, 28, 1], dtype=np.float32),
            y=np.ones([1], dtype=np.int32))
    ]
    keras_model = _create_test_cnn_model()
    server_state.model_weights.assign_weights_to(keras_model)

    sample_data = [
        collections.OrderedDict(
            x=np.ones([1, 28, 28, 1], dtype=np.float32),
            y=np.ones([1], dtype=np.int32))
    ]
    metric = tf.keras.metrics.SparseCategoricalAccuracy()
    accuracy = stateful_fedavg_tf.keras_evaluate(keras_model, sample_data,
                                                 metric)
    self.assertIsInstance(accuracy, tf.Tensor)
    self.assertBetween(accuracy, 0.0, 1.0)


def _server_init(model, optimizer):
  """Returns initial `ServerState`.

  Args:
    model: A `tff.learning.Model`.
    optimizer: A `tf.train.Optimizer`.

  Returns:
    A `ServerState` namedtuple.
  """
  stateful_fedavg_tff._initialize_optimizer_vars(model, optimizer)
  return stateful_fedavg_tf.ServerState(
      model_weights=model.weights,
      optimizer_state=optimizer.variables(),
      round_num=0,
      total_iters_count=0)


class ServerTest(tf.test.TestCase):

  def _assert_server_update_with_all_ones(self, model_fn):
    optimizer_fn = lambda: tf.keras.optimizers.SGD(learning_rate=0.1)
    model = model_fn()
    optimizer = optimizer_fn()
    state = _server_init(model, optimizer)
    weights_delta = tf.nest.map_structure(tf.ones_like,
                                          model.trainable_variables)

    for _ in range(2):
      state = stateful_fedavg_tf.server_update(
          model, optimizer, state, weights_delta, total_iters_count=0)

    model_vars = self.evaluate(state.model_weights)
    train_vars = model_vars.trainable
    self.assertLen(train_vars, 2)
    self.assertEqual(state.round_num, 2)
    # weights are initialized with all-zeros, weights_delta is all ones,
    # SGD learning rate is 0.1. Updating server for 2 steps.
    self.assertAllClose(train_vars, [np.ones_like(v) * 0.2 for v in train_vars])

  def test_self_contained_example_custom_model(self):
    self._assert_server_update_with_all_ones(MnistModel)


class ClientTest(tf.test.TestCase):

  def test_self_contained_example(self):

    client_data = _create_client_data()

    model = MnistModel()
    optimizer_fn = lambda: tf.keras.optimizers.SGD(learning_rate=0.1)
    losses = []
    for r in range(2):
      optimizer = optimizer_fn()
      stateful_fedavg_tff._initialize_optimizer_vars(model, optimizer)
      server_message = stateful_fedavg_tf.BroadcastMessage(
          model_weights=model.weights, round_num=r)
      outputs = stateful_fedavg_tf.client_update(model, client_data(),
                                                 _create_one_client_state(),
                                                 server_message, optimizer)
      losses.append(outputs.model_output.numpy())

    self.assertEqual(outputs.client_state.iters_count, 1)
    self.assertAllEqual(int(outputs.client_weight.numpy()), 2)
    self.assertLess(losses[1], losses[0])


if __name__ == '__main__':
  tf.test.main()
