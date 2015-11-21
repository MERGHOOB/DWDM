# DWDM

mk_p2v.lua:
implements skip gram model of paragraph-vector
all function realted to training 

mk_main.lua:
it is a driver which drives mk_p2v functionaliy and creates vectors of train and test tweets vectors and push them
to train_vector.txt and test_vector.txt files..

mk_svm.py:
it is file which uses paragraph vectors in svm to predict test vector reading from train_vector.txt and this svm is trained using train_vector.txt .

PRJECTREPORT IS PRESENT AT GOOGLE DRIVE: 
https://drive.google.com/file/d/0B3aalu_16PGoMnlWMS1PSlhmaDg/view?usp=sharing
