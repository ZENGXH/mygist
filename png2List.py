import glob
path = "./data/2015/"
fileName = "totalFileList.txt"
f = open(fileName, 'w')
fileList = glob.glob(path + "RAD*")
print "num of file is " + str(len(fileList)) + " in " + path

for file in fileList:
  f.write(file.split('/')[3]+"-linear.png")
  f.write('\n')
print "done"
print "save as " + fileName
