# whimh2.0

##Two stupid python functions:
###moveFile
###fillFiles
###JPG2jpg
###faceDetect
###whihmPre
=========
put file of specific class(mh_not/mh_yes/human_not) files in temp/ 
and excute the moveFile.py , select data_dir to put files into.
```
pytohn moveFile.py
```
=========
reorders the files in specific dirs:
i.e. xxx.jpg yyy.jpg,... => 1.jpg ,2.jpg, ....
```
pytohn fillFiles.py
```
=========
replace JPG to jpg in unclass dirs
i.e. xxx.jpg yyy.jpg,... =>  xxx.JPG yyy.JPG...
```
pytohn JPG2jpg.py
```
=========
extract faces of file in unclass, you need to provide
which is min-han ,which are not human face
using comma to split numbers
e.g
which is not human?
3,5,7
```
pytohn faceDetect.py
```
=========
to load files in three dirs to the format that using to classifiy
it will be saved in outfile_x.npy ,outfile_y.npy
```
pytohn whihmPre.py
```
