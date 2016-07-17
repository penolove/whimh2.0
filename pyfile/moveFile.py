import glob
import os

import sys

dirPath="/home/stream/whimh2/"
# Get user supplied values
#imagePath = sys.argv[1]
print "for files in temp/ ,which dir do you want move to ?: "
print "[mh_yes ,mh_not ,human_not]"
imagePath=raw_input()
fileInD=glob.glob(dirPath+imagePath+"/*.jpg")
count=len(fileInD)
file2move=glob.glob(dirPath+"temp/*.jpg")


for (from_F) in file2move:
	print dirPath+imagePath+"/"+str(count)+".jpg"
	os.rename(from_F,dirPath+imagePath+"/m"+str(count)+".jpg")
	count=count+1
