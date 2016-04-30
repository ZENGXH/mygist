# test_dataIterator

import numpy
import logging
import theano
import theano.tensor as TT
import theano.tensor.nnet
import random
from sparnn.utils import *

from sparnn.iterators import NumpyIterator


iterator_param = {'path': '../../SPARNN/data/2015/',
                  'minibatch_size': 3,
                  'use_input_mask': False,
                  'input_data_type': 'float32',
                  'is_output_sequence': True,
                  'name': 'hko-train-iterator',
                  'input_frame_num': 5,
                  'output_frame_num': 4,
                  'input_imageW': 100,
                  'mode': 'train'}


"""
iterator_param = {'path': '../data/hko-example/hko-valid.npz',
                  'minibatch_size': 10,
                  'use_input_mask': False,
                  'input_data_type': 'float32',
                  'is_output_sequence': True,
                  'name': 'hko-train-iterator',
                  'input_frame_num': 5,
                  'output_frame_num': 4,
                  'input_imageW': 100,
                  'mode': 'train'}

valid_iterator = NumpyIterator(iterator_param)
valid_iterator.begin(do_shuffle=False)
valid_iterator.print_stat()

valid_iterator.begin()
#print('inputBatch iterators[clips][0]', valid_iterator.data['clips'][0][1:10, :])
#print('current_input_length', valid_iterator.current_input_length)
#print('self.current_batch_indices: ', valid_iterator.current_batch_indices)

valid_iterator.next()


valid_iterator.next()
valid_iterator.next()
#print('current_input_length', valid_iterator.current_input_length)
#print('self.current_batch_indices: ', valid_iterator.current_batch_indices)
#print('inputBatch iterators[clips][0]', valid_iterator.data['clips'][0][1:10, :])


inputBatch = valid_iterator.input_batch()

print(len(inputBatch)) # list len 1
print(inputBatch[0].shape) # (5, 3, 1, 100, 100)
print(type(inputBatch[0]))
"""
print('simply test raw iterators:')
# test_rawiterator.py
import raw_iterator

dataIter = raw_iterator.RawIterator(iterator_param)
dataIter.print_stat()
dataIter.begin()

for i in range(3):
# dataIter.entries_dataIter.getBatch()generate()
   input_batch = dataIter.input_batch()
   output_batch = dataIter.output_batch()
   dataIter.next()
#for i in range():
# d = getdata.next()
# check the format of the dataIterator, 
# which should be same as the numpyIterator
assert(len(input_batch) == 1)
print(input_batch[0].shape)
print(output_batch[0].shape)
print(type(input_batch[0]))

# next try test the net forward & bp
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
import datetime
time = datetime.datetime.strftime(datetime.datetime.now(), '%m%d%H')

random.seed(1000)
numpy.random.seed(1000)

ACTIVATE_FUNC = "identity" #  sys.argv[1]  #  "sigmoid"
MINIBATCH_SIZE = 2
cost_func = "SquaredLoss" # sys.argv[2] # "BinaryCrossEntropy"

memory_dim = 1

input_seq_length = 2
output_seq_length = 2
save_path = "./hko-record-" + ACTIVATE_FUNC + "-" + cost_func + "/" + time + "train/"
log_path = save_path + "valid-" + ACTIVATE_FUNC + "-" + cost_func + ".log"

iterator_param = {'path': 'data/hko-example/hko-train.npz',
		            'minibatch_size': MINIBATCH_SIZE,
                  'use_input_mask': False,
                  'input_data_type': 'float32',
                  'is_output_sequence': True,
                  'name': 'hko-train-iterator'}

iterator_param = {'path': '../../SPARNN/data/2015/',
                  'minibatch_size': MINIBATCH_SIZE,
                  'use_input_mask': False,
                  'input_data_type': 'float32',
                  'is_output_sequence': True,
                  'name': 'hko-train-iterator',
                  'input_frame_num': input_seq_length,
                  'output_frame_num': output_seq_length,
                  'input_imageW': 100,
                  'mode': 'train'}
train_iterator = raw_iterator.RawIterator(iterator_param)
train_iterator.begin(do_shuffle=True)
train_iterator.print_stat()

iterator_param = {'path': '../../SPARNN/data/2015/',
                  'minibatch_size': MINIBATCH_SIZE,
                  'use_input_mask': False,
                  'input_data_type': 'float32',
                  'is_output_sequence': True,
                  'name': 'hko-train-iterator',
                  'input_frame_num': input_seq_length,
                  'output_frame_num': output_seq_length,
                  'input_imageW': 100,
                  'mode': 'test'}
test_iterator = raw_iterator.RawIterator(iterator_param)
test_iterator.begin(do_shuffle=True)
test_iterator.print_stat()

iterator_param = {'path': '../../SPARNN/data/2015/',
                  'minibatch_size': MINIBATCH_SIZE,
                  'use_input_mask': False,
                  'input_data_type': 'float32',
                  'is_output_sequence': True,
                  'name': 'hko-train-iterator',
                  'input_frame_num': input_seq_length,
                  'output_frame_num': output_seq_length,
                  'input_imageW': 100,
                  'mode': 'valid'}
valid_iterator = raw_iterator.RawIterator(iterator_param)
valid_iterator.begin(do_shuffle=True)
valid_iterator.print_stat()



rng = sparnn.utils.quick_npy_rng()
theano_rng = sparnn.utils.quick_theano_rng()



param = {"id": "hko", "use_input_mask": False, "input_ndim": 5, "output_ndim": 5}
interface_layer = InterfaceLayer(param)

patch_size = 1 # originally 2
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
         "dim_out": (memory_dim, row_num, col_num),
         "input_kernel": (3, 3),
         "transition_kernel": (3, 3),
         "minibatch_size": minibatch_size,
         #  "learn_padding": True,
         "input": reshape_input,
         "n_steps": input_seq_length}
middle_layers.append(ConvLSTMLayer(param))

param = {"id": "cost", "rng": rng, "theano_rng": theano_rng,
         "cost_func": cost_func,
         "dim_in": data_dim, "dim_out": (1, 1, 1),
         "minibatch_size": minibatch_size,
         "input": middle_layers[0].output,
         "target": reshape_output}
cost_layer = ElementwiseCostLayer(param)

outputs = [{"name": "prediction", "value": middle_layers[0].output}]

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

