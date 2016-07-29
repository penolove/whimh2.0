# whimh2.0
need:
anaconda 2.7
opencv
tensorflow

we have consturcted a model to classify min-han/ not min-han /not human  
using CNN implemented in cs231n model  
layers: hidden_dims=[64,128,256],hidden_dim=[1024,500]   
get about ~90% accuracy  
the detail is in the whimh2_training.ipynb  
## you should install python 2.7 using anaconda
##Five stupid python functions:
####moveFile
put file of specific class(mh_not/mh_yes/human_not) files in temp/ 
and excute the moveFile.py , select data_dir to put files into.
```
pytohn moveFile.py
```
####fillFiles
=========
reorders the files in specific dirs:
i.e. xxx.jpg yyy.jpg,... => 1.jpg ,2.jpg, ....
```
pytohn fillFiles.py
```
####JPG2jpg
replace JPG to jpg in unclass dirs
i.e. xxx.jpg yyy.jpg,... =>  xxx.JPG yyy.JPG...
```
pytohn JPG2jpg.py
```

###faceDetect

####you need opencv on python
https://omoshetech.com/how-to-install-anaconda-and-opencv/
####and gtk2-devel
sudo yum groupinstall "Development Tools"  
sudo yum install pkgconfig  
sudo yum install cmake  
sudo yum install gtk2-devel  


extract faces of file in unclass, you need to provide
which is min-han ,which are not human face
using comma to split numbers
e.g
which is not human?
3,5,7
```
pytohn faceDetect.py
```

###whihmPre
to load files in three dirs to the format that using to classifiy
it will be saved in outfile_x.npy ,outfile_y.npy
```
pytohn whihmPre.py
```

you need to download these three file to whimh2/ :

1. vggface model [link](https://drive.google.com/open?id=0B0GW3-McRihWM0VsWkh1Q3JYTE0)
2. pretensorWhimh [link](https://drive.google.com/open?id=0B0GW3-McRihWRllSLXN5UUZ1ek0)
3. finetune.npy [link](https://drive.google.com/open?id=0B0GW3-McRihWSVhHZkJRcG5TU1U)

jupyter-notebook


