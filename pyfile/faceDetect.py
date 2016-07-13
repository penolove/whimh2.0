import cv2
import sys
import glob
import os

#dir_path
dirPath = "/home/stream/whimh2/"
# Get user supplied values
cascPath = dirPath+"haarcascade_frontalface_default.xml"

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

def gogo(imagePath):
    # Read the image
    image = cv2.imread(imagePath)
    image2 = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    print "Found {0} faces!".format(len(faces))

    count=0
    font = cv2.FONT_HERSHEY_SIMPLEX
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0),3)
        cv2.putText(image,str(count),(x+w/2,y-5), font, 5,(200,75,70),5)
        #cv2.imshow("qq",image[y:y+h,x:x+w])
        count=count+1

    cv2.imshow("Faces found", cv2.resize(image,(500,500)))
    cv2.waitKey(0)

    print "which is minhan?"
    min_han=raw_input()
    print "which are not human?"
    not_human=raw_input()

    min_han=map(lambda x : int(x),filter(lambda y: y!='',min_han.split(' ')))
    not_human=map(lambda x : int(x),filter(lambda y: y!='',not_human.split(' ')))
    mh_not=filter(lambda x: (x not in min_han)and(x not in not_human),range(len(faces)))
    count=len(glob.glob(dirPath+"mh_yes/*.jpg"))+1
    if (len(faces)!=0):
        for (x,y,w,h) in faces[min_han]:
           cv2.imwrite(dirPath+"mh_yes/"+str(count) + ".jpg", image2[y:y+h,x:x+w])
           count=count+1

        count=len(glob.glob(dirPath+"human_not/*.jpg"))+1
        for (x,y,w,h) in faces[not_human]:
           cv2.imwrite(dirPath+"human_not/"+str(count) + ".jpg", image2[y:y+h,x:x+w])
           count=count+1

        count=len(glob.glob(dirPath+"mh_not/*.jpg"))+1
        for (x,y,w,h) in faces[mh_not]:
           cv2.imwrite(dirPath+"mh_not/"+str(count) + ".jpg", image2[y:y+h,x:x+w])
           count=count+1


detImg=glob.glob(dirPath+"unclass/*.jpg")

for i in detImg:
    gogo(i)
    os.remove(i)
