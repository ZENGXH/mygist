__author__ = 'sxjscience'
import sys

import theano
import theano.tensor as TT

import sparnn
import sparnn.utils
from sparnn.utils import *

from sparnn.iterators import NumpyIterator
from sparnn.layers import InterfaceLayer
from sparnn.layers import AggregatePoolingLayer
from sparnn.layers import DropoutLayer
from sparnn.layers import ConvLSTMLayer
from sparnn.layers import ConvForwardLayer
from sparnn.layers import ElementwiseCostLayer
from sparnn.layers import EmbeddingLayer
from sparnn.layers import GenerationLayer
from sparnn.models import Model

from sparnn.optimizers import SGD
from sparnn.optimizers import RMSProp
from sparnn.optimizers import AdaDelta

from sparnn.helpers import movingmnist

import os
import random
import numpy
import logging

random.seed(1000)
numpy.random.seed(1000)

ACTIVATE_FUNC = sys.argv[1]  #  "sigmoid"
MINIBATCH_SIZE = 8
COST_FUNC = sys.argv[2] # "BinaryCrossEntropy"


save_path = "./hko-record-" + ACTIVATE_FUNC + "-" + COST_FUNC + "/"
log_path = save_path + "valid-" + ACTIVATE_FUNC + "-" + COST_FUNC + ".log"
print("save path is " + save_path)
if not os.path.exists(save_path):
    os.makedirs(save_path)

sparnn.utils.quick_logging_config(log_path)

iterator_param = {'path': 'data/hko-example/hko-train.npz',
		  'minibatch_size': MINIBATCH_SIZE,
                  'use_input_mask': False,
                  'input_data_type': 'float32',
                  'is_output_sequence': True,
                  'name': 'hko-train-iterator'}
train_iterator = NumpyIterator(iterator_param)
train_iterator.begin(do_shuffle=True)
train_iterator.print_stat()

iterator_param = {'path': 'data/hko-example/hko-valid.npz',
                  'minibatch_size': MINIBATCH_SIZE,
                  'use_input_mask': False,
                  'input_data_type': 'float32',
                  'is_output_sequence': True,
                  'name': 'hko-valid-iterator'}
valid_iterator = NumpyIterator(iterator_param)
valid_iterator.begin(do_shuffle=False)
valid_iterator.print_stat()

iterator_param = {'path': 'data/hko-example/hko-test.npz',
                  'minibatch_size': MINIBATCH_SIZE,
                  'use_input_mask': False,
                  'input_data_type': 'float32',
                  'is_output_sequence': True,
                  'name': 'hko-test-iterator'}
test_iterator = NumpyIterator(iterator_param)
test_iterator.begin(do_shuffle=False)
test_iterator.print_stat()

rng = sparnn.utils.quick_npy_rng()
theano_rng = sparnn.utils.quick_theano_rng()

input_seq_length = 5
output_seq_length = 15

param = {"id": "hko", "use_input_mask": False, "input_ndim": 5, "output_ndim": 5}
interface_layer = InterfaceLayer(param)

patch_size = 2 # originally 2
reshape_input = quick_reshape_patch(interface_layer.input, patch_size)
reshape_output = quick_reshape_patch(interface_layer.output, patch_size)
feature_num = patch_size * patch_size  # 2 * 2 = 4
row_num = int(100 / patch_size)  # 50
col_num = int(100 / patch_size)  # 50
data_dim = (feature_num, row_num, col_num)  # (4, 50, 50)


logger.info("Data Dim:" + str(data_dim))
minibatch_size = interface_layer.input.shape[1]  # 8

middle_layers = []
param = {"id": 0, "rng": rng, "theano_rng": theano_rng,
         "dim_in": data_dim,
         "dim_out": (64, row_num, col_num),
         "input_kernel": (3, 3),
         "transition_kernel": (3, 3),
         "minibatch_size": minibatch_size,
         #  "learn_padding": True,
         "input": reshape_input,
         "n_steps": input_seq_length}
middle_layers.append(ConvLSTMLayer(param))
param = {"id": 1, "rng": rng, "theano_rng": theano_rng, "dim_in": (64, row_num, col_num),
         "dim_out": (64, row_num, col_num),
         "input_kernel": (3, 3),
         "transition_kernel": (3, 3),
         "minibatch_size": minibatch_size,
         #"learn_padding": True,
         #"input_padding": middle_layers[0].hidden_padding,
         "input": middle_layers[0].output,
         "n_steps": input_seq_length}
middle_layers.append(ConvLSTMLayer(param))


param = {"id": 2, "rng": rng, "theano_rng": theano_rng, "dim_in": data_dim,
         "dim_out": (64, row_num, col_num),
         "input_kernel": (3, 3),
         "transition_kernel": (3, 3),
         "init_hidden_state": middle_layers[0].output[-1],
         "init_cell_state": middle_layers[0].cell_output[-1],
         "minibatch_size": minibatch_size,
         #"learn_padding": True,
         "input": None,
         "n_steps": output_seq_length - 1}
middle_layers.append(ConvLSTMLayer(param))
param = {"id": 3, "rng": rng, "theano_rng": theano_rng,
         "dim_in": (64, row_num, col_num),
         "dim_out": (64, row_num, col_num),
         "input_kernel": (3, 3),
         "transition_kernel": (3, 3),
         "init_hidden_state": middle_layers[1].output[-1],
         "init_cell_state": middle_layers[1].cell_output[-1],
         "minibatch_size": minibatch_size,
         #"learn_padding": True,
         #"input_padding": middle_layers[2].hidden_padding,
         "input": middle_layers[2].output,
         "n_steps": output_seq_length - 1}
middle_layers.append(ConvLSTMLayer(param))


param = {"id": 4, "rng": rng, "theano_rng": theano_rng,
         "dim_in": (2*64, row_num, col_num),
         "dim_out": data_dim,
         "input_kernel": (1, 1),
         "input_stride": (1, 1),
         "activation": ACTIVATE_FUNC,
         "minibatch_size": minibatch_size,
         "input": TT.concatenate([
             TT.concatenate([
                 middle_layers[0].output[-1:],
                 middle_layers[1].output[-1:]], axis=2),
             TT.concatenate([
                 middle_layers[2].output,
                 middle_layers[3].output], axis=2)])
         }
middle_layers.append(ConvForwardLayer(param))



param = {"id": "cost", "rng": rng, "theano_rng": theano_rng,
         "cost_func": COST_FUNC,
         "dim_in": data_dim, "dim_out": (1, 1, 1),
         "minibatch_size": minibatch_size,
         "input": middle_layers[4].output,
         "target": reshape_output}
cost_layer = ElementwiseCostLayer(param)

outputs = [{"name": "prediction", "value": middle_layers[4].output}]

# error_layers = [cost_layer]

param = {'interface_layer': interface_layer,
         'middle_layers': middle_layers,
         'cost_layer': cost_layer,
         'outputs': outputs, 'errors': None,
         'name': "HKO-Convolutional-Unconditional",
         'problem_type': "regression"}
model = Model(param)
model.print_stat()

param = {'id': '1', 'learning_rate': 1e-6, 'momentum':0.9, 'decay_rate': 0.5,
         'clip_threshold': None,
         'max_epoch': 200, 'start_epoch': 0,
         'max_epochs_no_best': 50, 'decay_step':100,
         'autosave_mode': ['interval', 'best'], 'save_path': save_path, 'save_interval': 3}
optimizer = SGD(model, train_iterator, valid_iterator, test_iterator, param)
optimizer.train()
