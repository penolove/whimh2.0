import glob
import os
import sys



fileInD=glob.glob("/home/stream/whimh2/unclass/*.JPG")
for (from_F) in fileInD:
	to_re=from_F.replace("JPG","jpg")
	os.rename(from_F, to_re)