# dataUtils.py

def png2list():
	import glob
	fileName = "totalFileList.txt"
	f = open(fileName, 'w')
	# for year in ["2009", "2010", "2011" ,"2012", "2013", "2014", "2015"]:
	for year in ["2009", "2010", "2011" ,"2012", "2013","2014", "2015"]:
		path = "../SPARNN/data/" + year + "new/"
		print(path)
		fileList = glob.glob(path + "*png")
		for file in fileList:
			print(file)
			e = file.split('/')[3] + '/' + file.split('/')[4]
			#print(e)
			f.write(e)
			f.write('\n')
		print "num of file is " + str(len(fileList)) + " in " + path
	print "png2list done"
	print "save as " + fileName

def test():
	
	import numpy
	import raw_iterator
	input_seq_length = 3
	output_seq_length = 5
	iterator_param = {'path': '../SPARNN/data/',
	                  'total_list_file': 'totalFileList.txt',
	                  'minibatch_size': 4,
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
	train_iterator.input_batch()
	train_iterator.input_batch()
	# dataIter = raw_iterator.RawIterator(iterator_param)

	# for i in range(300):
	# dataIter.entries_dataIter.getBatch()generate()
	# dataIter.getNextBatch()

# png2list()
test()