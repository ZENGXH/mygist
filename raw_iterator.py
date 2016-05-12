
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
		# print(iterator_param)
		self.path = iterator_param.get('path', '../../SPARNN/data/') ##'../../SPARNN/data/2015/'
		self.mode = iterator_param.get('mode', 'train')
		self.name = iterator_param.get('name', 'iterators')

		self.total_list_file = iterator_param.get('total_list_file', "totalFileList.txt")
		of = open(self.total_list_file, 'r')
		lines = of.readlines()
		of.close()
		self.total_image = len(lines)

		self.slot_size = iterator_param.get('slot_size', 640) # train : test : valid = 6:1:1, ie slotsize*6/8 : slotsize/8 : slotsize/8
		self.select_step = iterator_param.get('select_step', 1)
		self.input_frame_num = iterator_param.get('input_frame_num', 5)
		self.output_frame_num = iterator_param.get('output_frame_num', 15)

		self.imageW = iterator_param.get('input_imageW', 100)	
		self.imageH = self.imageW
		self.minibatch_size = iterator_param.get('minibatch_size', 1)
		self.depth = iterator_param.get('depth', 1)

		self.data_txt = self.total_list_file.split(".txt")[0] + "_"+ self.mode +".txt"

		# self.slot_devide(self.total_list_file, self.total_image, self.slot_size):
		self.txt_entries = []
		
		self.numOfSlot = int(math.floor(self.total_image / self.slot_size))
		self.stride = self.select_step * (self.input_frame_num + self.output_frame_num)

		self.dataSize = 0 # init
		# init slodIDs
		generateNewText = iterator_param.get('generateNewText', False)
		if generateNewText:
			self.slotDevide()
			#self.entriesTextGenerator()
		else:
			self.calculateSlotSizeFromTxt()
			
		self.slotIDs = range(self.minibatch_size)
		#if not os.path.isfile(self.data_txt):
		#self.slotDevide()

	def begin(self, do_shuffle=True):
		print('begin')
		

		print('loading data_txt')
		of = open(self.data_txt, 'r')
		self.imageList = of.readlines()
		assert(not len(self.imageList) == 0)
		of.close()
		self.entriesCal(do_shuffle)

		return

	def total(self):
		return self.total_image


	def entriesCal(self, do_shuffle=False):
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
		totalImage = 0
		countImage = 0
		self.txt_entries_file = open(self.total_list_file.split(".txt")[0] + "_train_entry.txt", 'w')
		for numOfSlot in range(self.numOfSlot): # number of slot: 135
			subslot_size = self.slot_size / 8
			# print(numOfSlot)
			if(self.mode == 'train'):
				subslot_size = subslot_size * 6 # 480
				for lineNumber in range(numOfSlot* subslot_size, numOfSlot * subslot_size + subslot_size - self.stride):
					self.txt_entries.append(lineNumber)
					
					## -----------
					## self.lineNumber = self.txt_entries[self.slotID] 
					'''
					for length in range(self.input_frame_num):
						totalImage += 1
						# print('lineNumber: ' + str(self.lineNumber) + ' ' + self.imageList[self.lineNumber])
						imgname = self.imageList[lineNumber]
						# year = "20" + self.imageList[lineNumber].split('img')[1][0] + self.imageList[self.lineNumber].split('img')[1][1] + 'new/'
						name = self.path + self.imageList[lineNumber].split('\n')[0]
						# print(name)
						imagePIL = Image.open(name)
						numpy.asarray(imagePIL, dtype=numpy.float32)
						if(numpy.mean(imagePIL) > 2):
							#print(name)
							self.txt_entries_file.write(name)
							self.txt_entries_file.write('\n')
							countImage += 1
							self.txt_entries.append(lineNumber)
							#print('.')
						#else:
							#print('x')
							
						##result[length, i, 0, :, :] = imagePIL
						##self.lineNumber += self.select_step
					## -----------
					'''

			else:
				subslot_size = subslot_size # 80
				for lineNumber in range(numOfSlot* subslot_size, numOfSlot * subslot_size + subslot_size - self.stride):
					self.txt_entries.append(lineNumber)
			if(do_shuffle):
				# print('shuffle data')
				shuffle(self.txt_entries)
		self.txt_entries_file.close()
		print("entries_generate done: ")
		self.dataSize = len(self.txt_entries)
		print('total image: ' + str(totalImage*self.input_frame_num) + ' and count image: ' + str(countImage))
		
	def calculateSlotSizeFromTxt(self):
		print()
		valid_txt = open(self.total_list_file.split(".txt")[0] + "_valid.txt", 'r')
		valid_total = valid_txt.readlines()
		valid_txt.close()
		valid_total = len(valid_total)
		self.numOfSlot = valid_total/self.slot_size * 8
		print('update slotSize to be '+ str(self.numOfSlot))

	def slotDevide(self):
		"""
			only called when the train, valid, test file do not exist
		"""
		print("producing " + self.total_list_file.split(".txt")[0] + "_train.txt" + ' and valid and test')
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
		c = 0
		
		mean_txt = open(self.total_list_file.split(".txt")[0] + "_slot640_mean.txt", 'w')
		
		mean_list = []
		
		for i in range(self.numOfSlot):
			#print(i)
			totalMean = 0
			start = self.slot_size * i
			subslot_size = self.slot_size / 8
			for line in lines[start: start + 8*subslot_size]:
				name = self.path + line.split('\n')[0]
				imagePIL = Image.open(name)
				numpy.asarray(imagePIL, dtype=numpy.float32)
				totalMean += numpy.mean(imagePIL)
			print('slot ' + str(i) + ' mean: ' + str(totalMean))
			mean_txt.write(lines[start].split('\n')[0] + '-mean-' + str(totalMean) + '\n')
			mean_list.append(totalMean)
		# ------
   		list.sort(mean_list)
   		print('mean_list num100: ' + str(mean_list[250]))
   		threMean = mean_list[self.numOfSlot - 100]
   		mean_txt.close()
   		
   		#threMean = 12496.7436
   		print('# 100' + str(threMean))
   		print('# ')
   		# -------- get threhold
   		
   		mean_txt = open(self.total_list_file.split(".txt")[0] + "_slot640_mean.txt", 'r')
   		all_means = mean_txt.readlines()
   		mean_txt.close()
 		for i in range(self.numOfSlot):
			#print(i)
			
			#print(all_means)
			curmean = float(all_means[i].split('-')[2].split('\n')[0])
			name = all_means[i].split('-')[0]
			if curmean > threMean:
				#print('curmean ' + str(curmean) + ' ' + name)
				#continue
			#else:
				print('get ' + str(curmean))
			start = self.slot_size * i
			subslot_size = self.slot_size / 8
			# calculate total mean again
			
				  		
			if(curmean > threMean):
				c += 1
				print(name)

				for line in lines[start : start + 6 * subslot_size]:
					train_txt.write(line)
		 		
				for line in lines[start + 6 * subslot_size : start + 7 * subslot_size]:
					test_txt.write(line) 
				
				for line in lines[start + 7 * subslot_size : start + 8 * subslot_size]:
					valid_txt.write(line)

		print('self.numOfSlot: ' + str(self.numOfSlot) + ' to valid count ' + str(c))
		self.numOfSlot = c
   		train_txt.close()
   		valid_txt.close()
   		test_txt.close()




	def dataReadNextInput(self):
		# print('in')
		result = numpy.zeros((self.input_frame_num, self.minibatch_size, self.depth, self.imageH, self.imageW), dtype=numpy.float32)
		while True:
			# load batchSize
			for i in range(self.minibatch_size):
				self.slotID = self.slotIDs[i]
				# start point:
				self.lineNumber = self.txt_entries[self.slotID] 
				for length in range(self.input_frame_num):
					# print('lineNumber: ' + str(self.lineNumber) + ' ' + self.imageList[self.lineNumber])
					imgname = self.imageList[self.lineNumber]
					# year = "20" + self.imageList[self.lineNumber].split('img')[1][0] + self.imageList[self.lineNumber].split('img')[1][1] + 'new/'
					name = self.path + self.imageList[self.lineNumber].split('\n')[0]
					# print(name)
					imagePIL = Image.open(name)
					numpy.asarray(imagePIL, dtype=numpy.float32)
					result[length, i, 0, :, :] = imagePIL
					self.lineNumber += self.select_step

				"""
				imagePIL = Image.open(name)
				imagePIL.thumbnail(size)
				
				# imagePIL.show()
				img = numpy.asarray(imagePIL)
				print(img.shape)
				result[i] = img
				"""
			yield result

	def dataReadNextOutput(self):
		#print('out')
		result = numpy.zeros((self.output_frame_num, self.minibatch_size, self.depth, self.imageH, self.imageW), dtype=numpy.float32)
		#print(self.txt_entries[1:10])
		while True:
			# load batchSize
			for i in range(self.minibatch_size):
				self.slotID = self.slotIDs[i]
				# start point:
				self.lineNumber = self.txt_entries[self.slotID] + self.select_step * self.input_frame_num
				# print('self.slotID: ' + str(self.slotID), len(self.txt_entries))
				for length in range(self.output_frame_num):
					# print('lineNumber: ' + str(self.lineNumber) + ' ' + self.imageList[self.lineNumber])
					imgname = self.imageList[self.lineNumber]
					# year = "20" + self.imageList[self.lineNumber].split('img')[1][0] + self.imageList[self.lineNumber].split('img')[1][1] + 'new/'
					name = self.path + self.imageList[self.lineNumber].split('\n')[0]
					# print(name)
					imagePIL = Image.open(name)
					numpy.asarray(imagePIL, dtype=numpy.float32)
					# imagePIL.show()
					# print(img.shape)
					# result[i] = img
					result[length, i, 0, :, :] = imagePIL
					self.lineNumber += self.select_step
			yield result
	def next(self):
		"""
			generate next slot ids for next inputBatch
		"""
		for i in range(self.minibatch_size):
			self.slotIDs[i] = (self.slotIDs[i] + 1)% len(self.txt_entries)
		return 

	def no_batch_left(self):
			return False

	def input_batch(self):
		return [next(self.dataReadNextInput())]

	def output_batch(self):
		return [next(self.dataReadNextOutput())]

	def check_data(self):
		return True

	def print_stat(self):
		
		print("   DataPath: " + self.path)
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






