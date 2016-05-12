



__author__ = 'sxjscience'

import sys
import theano
import theano.tensor as TT
import os
import cPickle
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
import raw_iterator

time = datetime.datetime.strftime(datetime.datetime.now(), '%d%H')

random.seed(1000)
numpy.random.seed(1000)

ACTIVATE_FUNC = "identity" #  sys.argv[1]  #  "sigmoid"
MINIBATCH_SIZE = 6
cost_func = "penSquaredLoss" # sys.argv[2] # "BinaryCrossEntropy"

memory_dim = 64
path = '../SPARNN/hko-record-identity-SquaredLoss/HKO-Convolutional-Unconditional-validation-best/model.pkl'

save_path = "./hko-record-" + ACTIVATE_FUNC + "-" + cost_func + "/" + time + "train/"
log_path = save_path + "valid-" + ACTIVATE_FUNC + "-" + cost_func + ".log"
print("save path is " + save_path)
if not os.path.exists(save_path):
    os.makedirs(save_path)

input_seq_length = 5
output_seq_length = 15

model = cPickle.load(open(path, 'rb'))  # open the model save during training


sparnn.utils.quick_logging_config(log_path)

iterator_param = {'path': 'data/',
                  'total_list_file': 'totalFileList.txt',
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

iterator_param = {'path': 'data/',
                  'total_list_file': 'totalFileList.txt',
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

iterator_param = {'path': 'data/',
                  'total_list_file': 'totalFileList.txt',
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

patch_size = 2 # originally 2
reshape_input = quick_reshape_patch(interface_layer.input, patch_size)
reshape_output = quick_reshape_patch(interface_layer.output, patch_size)
feature_num = patch_size * patch_size  # 2 * 2 = 4
row_num = int(100 / patch_size)  # 50
col_num = int(100 / patch_size)  # 50
data_dim = (feature_num, row_num, col_num)  # (4, 50, 50)


logger.info("Data Dim:" + str(data_dim))
minibatch_size = interface_layer.input.shape[1]  # 8

model.print_stat()

param = {'id': '1', 'learning_rate': 1e-6, 'momentum':0.9, 'decay_rate': 0.5,
         'clip_threshold': None,
         'max_epoch': 200, 'start_epoch': 0,
         'max_epochs_no_best': 50, 'decay_step':100,
         'autosave_mode': ['interval', 'best'], 'save_path': save_path, 'save_interval': 3}
optimizer = SGD(model, train_iterator, valid_iterator, test_iterator, param)
optimizer.train()
