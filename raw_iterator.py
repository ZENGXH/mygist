
"""
	overwrite:
	begin()
	next()
	
	input_batch(self)
	output_batch(self)
	print_stat(self)
"""

import numpy
import logging
import theano
import theano.tensor as TT
import theano.tensor.nnet
import random
from PIL import Image
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
class RawIterator(object):
	def __init__(self, iterator_param):
		
		# self.name = ''
		self.datapath = '../../mygist/small2015/'
		self.mode = iterator_param.get('mode', 'train')
		self.name = iterator_param.get('name', 'iterators')

		self.total_list_file = iterator_param.get('total_list_file', "../../mygist/2015imgList.txt")
		of = open(self.total_list_file, 'r')
		lines = of.readlines()
		of.close()
		self.total_image = len(lines)

		self.slot_size = iterator_param.get('slot_size', 640) # train : test : valid = 6:1:1, ie 480 : 80 : 80
		self.select_step = iterator_param.get('select_step', 2)
		self.input_frame_num = iterator_param.get('input_frame_num', 5)
		self.output_frame_num = iterator_param.get('output_frame_num', 15)

		self.imageW = iterator_param.get('input_imageW', 330)	
		self.imageH = self.imageW
		self.minibatch_size = iterator_param.get('minibatch_size', 1)
		self.depth = 1
		print(self.minibatch_size)

		# self.inputSize = iterator_param.get('inputSize', self.imageW)

		self.data_txt = self.total_list_file.split(".txt")[0] + "_"+ self.mode +".txt"
		# self.slot_devide(self.total_list_file, self.total_image, self.slot_size):
		self.txt_entries = []
		
		self.numOfSlot = int(math.floor(self.total_image / self.slot_size))
		self.stride = self.select_step * (self.input_frame_num + self.output_frame_num)

		self.dataSize = 0 # init
		self.entriesCal()
		self.input_nSeq = 5
		#super(RawIterator, self).__init__(iterator_param)
		# self.dataRead()
		# self._dataRead()

    	# self.data_generator = 
    	# self.load(self.path)
    	# self.dataBatch = self.dataRead()
        	

	def entriesCal(self):
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

		self.txt_entries = []
		for numOfSlot in range(self.numOfSlot): # number of slot: 135
			subslot_size = self.slot_size / 8
			# print(numOfSlot)
			if(self.mode == 'train'):
				print('shuffle data', self.mode)
				subslot_size = subslot_size * 6 # 480
				for lineNumber in range(numOfSlot* subslot_size, numOfSlot * subslot_size + subslot_size - self.stride):
					self.txt_entries.append(lineNumber)
				shuffle(self.txt_entries)
			else:
				subslot_size = subslot_size # 80
				for lineNumber in range(numOfSlot* subslot_size, numOfSlot * subslot_size + subslot_size - self.stride):
					self.txt_entries.append(lineNumber)
		print("entries_generate done: ")
		self.dataSize = len(self.txt_entries)
		print(self.txt_entries[1:10])

	def slotDevide(self):
		import math
		total_list_file = self.total_list_file
		of = open(self.total_list_file, 'r')
		lines = of.readlines()
		of.close()
		# print(type(lines))
		assert len(lines) == self.total_image
		train_txt = open(self.total_list_file.split(".txt")[0] + "_train.txt", 'w')
		valid_txt = open(self.total_list_file.split(".txt")[0] + "_valid.txt", 'w')
		test_txt = open(self.total_list_file.split(".txt")[0] + "_test.txt", 'w')

		for i in range(int(math.floor(self.total_image / self.slot_size))):
			print(i)
			start = self.slot_size * i
			subslot_size = self.slot_size / 8
			for line in lines[start : start + 6 * subslot_size]:
				train_txt.write(line)
	 		
			for line in lines[start + 6 * subslot_size : start + 7 * subslot_size]:
				test_txt.write(line) 
			
			for line in lines[start + 7 * subslot_size : start + 8 * subslot_size]:
				valid_txt.write(line)
   		train_txt.close()
   		valid_txt.close()
   		test_txt.close()


	def dataRead(self):
		print('loading data_txt')
		of = open(self.data_txt, 'r')
			
		imageList = of.readlines()
		assert(not len(imageList) == 0)
		
		of.close()

		
		
		# minibatch_size = self.minibatch_size
		# i = 0

		# for ind, line in enumerate(imageList):
		print('reset')
		result = numpy.zeros((self.input_frame_num, self.minibatch_size, self.depth, self.imageH, self.imageW))
		
		slotID = 0
		print(self.txt_entries[1:10])
		while True:
			# load batchSize
			for i in range(self.minibatch_size):
				slotID = slotID % len(self.txt_entries)
				# start point:
				lineNumber = self.txt_entries[slotID] 
				slotID += 1
				print('slotID: ' + str(slotID), len(self.txt_entries))
				print('--- for batch #' + str(i))
				for len_input in range(self.input_frame_num):
					print('lineNumber: ' + str(lineNumber) + ' ' + imageList[lineNumber])
					
					name = self.datapath + imageList[lineNumber].split('\n')[0]
					# print(name)
					# result.append(name)
					# img = mpimg.imread(name)
					# img = rgb2gray(img)
					
					imagePIL = Image.open(name)
					imagePIL = imagePIL.thumbnail((self.imageH, self.imageW))
					# print(imagePIL.type)
					numpy.asarray(imagePIL, dtype=numpy.float32)
					 
					# imagePIL.show()
					# print(img.shape)
					# result[i] = img

					result[len_input, i, 0, :, :] = imagePIL
					lineNumber += 1

				"""
				imagePIL = Image.open(name)
				imagePIL.thumbnail(size)
				
				# imagePIL.show()
				img = numpy.asarray(imagePIL)
				print(img.shape)
				result[i] = img
				"""
			print('----- \n')
			returnList = [result]

			# print('yield result')
			yield returnList

	def getNextBatch(self):
		return # next(self.data_generator)

	def print_stat(self):
		p = self.datapath
		print("   Path: " + str(p))
		print("   Minibatch Size: " + str(self.minibatch_size))
		print("   length of inpur Sequence: " + str(self.input_frame_num))
		print("   data txt file: " + self.data_txt)
		print("   input data size" + str(self.imageW))
		print("   mode: " + self.mode)

		print("    data preparation: ")
		print("      size of one slot = " + str(self.slot_size))
		print("      select_step = " + str(self.select_step))
		print("      input sequence length = " + str(self.input_frame_num))
		print("      output sequence length = " + str(self.output_frame_num))
		print("      total = " + str(self.stride))
		print("      -----")
		print("      number of slot: " + str(self.numOfSlot))
		print("      number of " + self.mode + " dataset entries: " + str(self.dataSize))

		# logger.info("Iterator Name: " + self.name)
        # logger.info("   Path: " + str(p))
        # logger.info("   Minibatch Size: " + str(m))
        # logger.info("   size of dataset " +str(self.total_image))
        # logger.info("   Input Data Type: " + str(self.input_data_type) + " Use Input Mask: " + str(self.use_input_mask))
        # logger.info(
        #   "   Output Data Type: " + str(self.output_data_type) + " Use Output Mask: " + str(self.use_output_mask))
        #logger.info("   Is Output Sequence: " + str(self.is_output_sequence))
        #logger.info("   length of inpur Sequence: ", + str(self.input_nSeq))






