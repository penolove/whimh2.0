import glob
import os
import sys
dirPath="/home/stream/whimh2/"
imagePath = sys.argv[1]

fileInD=glob.glob(dirPath+imagePath+"/*.jpg")
renames2=[dirPath+imagePath+"/"+str(i)+'.jpg' for i in range(len(fileInD))]
not_names=list(set(renames2) - set(fileInD))
more_names=list(set(fileInD) - set(renames2))
if(len(not_names)==len(more_names)):
	for (from_F,to_re) in zip(more_names,not_names):
		os.rename(from_F, to_re)
