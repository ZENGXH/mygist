"""
  devide all image list to slots each length `slot_size`
  for each slot, the first 6/8 of slot is the training set, 
  the 1/8 and 1/8 are test and valid set
  all the image name are store in the txt file
"""
import math



def slot_devide(total_list_file, total_image, slot_size):
	of = open(total_list_file, 'r')
	lines = of.readlines()
	of.close()
	print(type(lines))
	assert len(lines) == total_image
	train_txt = open(total_list_file.split(".txt")[0] + "_train.txt", 'w')
	valid_txt = open(total_list_file.split(".txt")[0] + "_valid.txt", 'w')
	test_txt = open(total_list_file.split(".txt")[0] + "_test.txt", 'w')

	for i in range(int(math.floor(total_image / slot_size))):
		print(i)
		start = slot_size * i
		subslot_size = slot_size / 8
		for line in lines[start : start + 6 * subslot_size]:
			train_txt.write(line)
 		
		for line in lines[start + 6 * subslot_size : start + 7 * subslot_size]:
			test_txt.write(line) 
		
		for line in lines[start + 7 * subslot_size : start + 8 * subslot_size]:
			train_txt.write(line)


total_image = 87452
total_list_file = "2015imgList.txt"
slot_size = 640
slot_devide(total_list_file, total_image, slot_size)