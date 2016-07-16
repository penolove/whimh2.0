import cv2
import sys
import glob
import os
import numpy as np
mh_yes_list=glob.glob("/home/stream/whimh2/mh_yes/*.jpg")
mh_not_list=glob.glob("/home/stream/whimh2/mh_not/*.jpg")
#mh_not_list=mh_not_list[:100]
human_not_list=glob.glob("/home/stream/whimh2/human_not/*.jpg")
#human_not_list=human_not_list[:100]
datarows=len(mh_yes_list)+len(mh_not_list)+len(human_not_list)
frames = np.empty((datarows, 32,32, 3))
frames2 = np.empty((datarows, 32,32, 3))

y_train=np.empty(datarows)
count=0
for k in xrange(len(mh_yes_list)):
    frames[k,:,:,:] = cv2.resize(cv2.imread(mh_yes_list[k]),(32,32))
y_train[count:count+len(mh_yes_list)]=0
count=count+len(mh_yes_list)

for k in xrange(len(mh_not_list)):
    frames[count+k,:,:,:] = cv2.resize(cv2.imread(mh_not_list[k]),(32,32))
y_train[count:count+len(mh_not_list)]=1
count=count+len(mh_not_list)

for k in xrange(len(human_not_list)):
    frames[count+k,:,:,:] = cv2.resize(cv2.imread(human_not_list[k]),(32,32))
y_train[count:count+len(human_not_list)]=2
count=count+len(human_not_list)

#R
frames2[:,:,:,0]=frames[:,:,:,2]
#G
frames2[:,:,:,1]=frames[:,:,:,1]
#B
frames2[:,:,:,2]=frames[:,:,:,0]

np.save("/home/stream/whimh2/outfile_x",frames2)
np.save("/home/stream/whimh2/outfile_y",y_train)
