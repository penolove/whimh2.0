import glob
import os

import sys

dirPath="/home/stream/whimh2/"
# Get user supplied values
imagePath = sys.argv[1]


fileInD=glob.glob(dirPath+imagePath+"/*.jpg")
count=len(fileInD)
file2move=glob.glob(dirPath+"temp/*.jpg")


for (from_F) in file2move:
	os.rename(from_F,dirPath+imagePath+"/"+str(count)+".jpg")
	count=count+1
