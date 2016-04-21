

import numpy
import logging
import theano
import theano.tensor as TT
import theano.tensor.nnet
import random
import math
from random import shuffle

from sparnn.utils import *
from sparnn.iterators import DataIterator

logger = logging.getLogger(__name__)
"""
class RawIterator(DataIterator):
	def __init__(self, iterator_param):
        super(RawIterator, self).__init__(iterator_param)
"""
class RawIterator():
	def __init__(self):
			self.total_image = 87452
			self.total_list_file = "2015imgList.txt"
			self.slot_size = 640 # train : test : valid = 6:1:1, ie 480 : 80 : 80
			self.select_step = 2
			self.input_frame_num = 5
			self.output_frame_num = 15
			self.entries_generate()
        	# self.load(self.path)


	def entries_generate(self):
		""" 
			based on 
			self.total_list_file + "_train.txt" 
			self.total_list_file + "_test.txt" 
			self.total_list_file + "_valid.txt" 
			file, for each minibatch, read from the entries of the minibatch data
			stride of data = self.select_step * (self.input_frame_num + self.output_frame_num)
			e.g. in train_txt:[1,2,3,4,....,480, 640, 641, ...]
			entries with stride 60: [1,2,3,...., 420, 640, 0641, ...]
		"""
		mode = 'train'
		self.txt_entries = []
		self.stride = self.select_step * (self.input_frame_num + self.output_frame_num)
		
		for numOfSlot in range(int(math.floor(self.total_image / self.slot_size))): # number of slot: 135
			subslot_size = self.slot_size / 8
			print(numOfSlot	)
			if(mode == 'train'):
				subslot_size = subslot_size * 6 # 480
				for lineNumber in range(numOfSlot* subslot_size, numOfSlot* subslot_size + subslot_size - self.stride):
					self.txt_entries.append(lineNumber)
				shuffle(self.txt_entries)
			else:
				subslot_size = subslot_size # 80
				for lineNumber in range(numOfSlot* subslot_size, numOfSlot* subslot_size + subslot_size - self.stride):
					self.txt_entries.append(lineNumber)
		print("entries_generate done")

"""
    def slot_devide()
    	of = open(self.total_list_file, 'r')
    	lines = of.readlines()
    	of.close()
    	assert len(lines) == self.total_image
    	train_txt = open(self.total_list_file.split(".txt")[0] + "_train.txt", 'w')
    	valid_txt = open(self.total_list_file.split(".txt")[0] + "_valid.txt", 'w')
    	test_txt = open(self.total_list_file.split(".txt")[0] + "_test.txt", 'w')

    	for i in range(int(math.floor(self.total_image / self.slot_size))):
    		print(i)
    		start = self.slot_size * i
    		subslot_size = self.slot_size/8
    		train_txt.write(line for line in lines[start : start + 6 * subslot_size])
     		test_txt.write(line for line in lines[start + 6 * subslot_size : start + 7 * subslot_size])
    		train_txt.write(line for line in lines[start + 7 * subslot_size : start + 8 * subslot_size])
   		
   		train_txt.close()
   		valid_txt.close()
   		test_txt.close()

    def txtGenerator(input_frame_num, output_frame_num, select_step):

    def read(directory_path, 
    		total = 87452
    		input_frame_num=10,
	        output_frame_num=20,
	        channel=1,
	        height=330,
	        width=330,
	        time_length_per_file=240):
	    # channel, height, width
	    dims = numpy.asarray([channel, height, width], dtype=numpy.int32)
        test_instance_count = 0
	    validation_instance_count = 0
	    train_instance_count = 0
	    trainData = numpy.zeros

	def trainDataRead():
		of = open('imgList_train.txt', 'r')
			
		imageList = of.readlines()
		of.close()

		datapath = './small2015/'
		total = 87452
		i = 0
		for ind, line in enumerate(imageList):
			if i == 0:
				print('reset')
				result = np.zeros((batchSize, sa, sa))
			if i < batchSize:
				name = datapath + line.split('\n')[0]
				print(name)
				# img = mpimg.imread(name)
				# img = rgb2gray(img)
				imagePIL = Image.open(name)
				imagePIL.thumbnail(size)
				# imagePIL.show()
				img = np.asarray(imagePIL)
				print(img.shape)
				result[i] = img
				i += 1
			if i == batchSize:
				i = 0
				print('yield result')
				yield result

	def validDataRead():
		of = open('imgList_valid.txt', 'r')

"""




