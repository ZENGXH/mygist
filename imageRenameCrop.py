## rename image file
import glob
from PIL import Image
""" 
##  glob all the name of the file in the folder

    fileList = glob.glob(path + nameOfFile)
    for target_file in fileList:
        loss_file = open(target_file, 'r')
        lines = loss_file.readlines()
        loss_file.close()

        record = 0
        for ind, line in enumerate(lines):
"""

listFile = "../SPARNN/totalFileList.txt"

of = open(listFile, 'r')
imageList = of.readlines()
of.close()
datapath = "../SPARNN/data/2015/"
num = 1
for ind, line in enumerate(imageList):

	oldname = datapath + line.split('\n')[0]
	newname = datapath + 'img15' + str(num).zfill(6) + '.png'
	img = Image.open(oldname)
	img = img.crop((75,75,405,405)) ## im.crop((left, top, right, bottom))
	img.save(newname, 'png')
	print(oldname)
	print(newname)
	num += 1
	## newname = datapath + 
	## os.rename(datapath + line,  )
