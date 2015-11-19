import numpy as np 
from sklearn import svm
from sklearn.metrics import f1_score

#create feature set
f = open("train_vectors.txt")
lines = f.readlines()
feat = []
for line in lines:
		line = line.strip()
		line = line.split()
		ft = line[1:]
		# print ft
		# break
		for i in range(len(ft)):
			ft[i] = float(ft[i])
		feat.append(ft)
f.close()
# print feat

#create label list
f = open("train.tsv")
lines = f.readlines()
labels = []
for line in lines:
		line = line.strip()
		line = line.split()
		labels.append(int(line[-1]))

f.close()

#training svm
X = np.array(feat)
y = labels
clf = svm.SVC(kernel ='linear', C=1000000)
clf.fit(X,y)

#creating feature for test file
f = open("test_vectors.txt")
feat = []
lines = f.readlines()
for line in lines:
	line = line.strip();line = line.split()
	ft=line[1:]
	for i in range(len(ft)):
		ft[i] = float(ft[i])
	feat.append(ft)

#reading labels form test.tsv
f = open("test.tsv")
true_labels = []
lines = f.readlines()
for line in lines:
	line = line.strip();line = line.split()
	true_labels.append(int(line[-1]))


f.close()

#predicting labels
pred_labels=[]
for i in feat:
	pred_labels.append(clf.predict(i))

count = 0
for  i in range(len(true_labels)):
	if true_labels[i] == pred_labels[i]:
		count = count + 1

print count,len(true_labels)
print f1_score(true_labels,pred_labels,average='micro')
