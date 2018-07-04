import numpy as np 
from sklearn.datasets import fetch_mldata
import os
import cv2
#from FisherLDA import mainFisherLDAtest
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn import naive_bayes
from sklearn import neighbors

mnist =[]
NOT=60000
DFILE="traindata.csv"
TFILE="testdata.csv"
L=[]
Y=[]
T_L=[]
T_Y=[]

def croping(img):
    l,w=np.shape(img)
    minx=-1
    maxx=-1
    miny=-1
    maxy=-1
    for i in range(l):
        for j in range(w):
            if img[i][j]>=0.2:
	        if minx==-1:
		    minx=j
		    maxx=j
		else:
		    if minx>j:
		        minx=j
		    if maxx<j:
		        maxx=j
		if miny==-1:
		    miny=i
		    maxy=i
		else:
		    if miny>i:
			miny=i
		    if maxy<i:
			maxy=i
    #print (minx,miny,maxx,maxy)
    img=img[minx:minx+maxx,miny:miny+maxy]
    img=cv2.resize(img,(28,28))
    return img
    
def analyze(img):
    pca=PCA(n_components=28)
    #FisherLDA external code
    
    len,wid=np.shape(img)
   
    pca.fit(img)
    #flda=mainFisherLDAtest([pca.singular_values_],0.5)
    #print("flda:  ",flda)
    #print("pca: ",pca.singular_values_)
    return pca.singular_values_
def save2file():
    global mnist
    global L
    global Y
    global T_L
    global T_Y
    global DFILE
    global TFILE
    with  open (DFILE, "w") as file:
        for entry in L:
            t=0
            for v in entry:
                if t!=0:
                    file.write(',')
                file.write('%s' % v)
                t=1
            file.write('\n')   
        file.write("Y"+'\n')
        t=0
        for p in Y:
            if t!=0:
                file.write(',') 
            file.write('%s' % p)
            t=1
    with  open (TFILE, "w") as file:
        for entry in T_L:
            t=0
            for v in entry:
                if t!=0:
                    file.write(',')
                file.write('%s' % v)
                t=1
            file.write('\n')   
        file.write("Y"+'\n')
        t=0
        for p in T_Y:
            if t!=0:
                file.write(',') 
            file.write('%s' % p)
            t=1
    return
def readfile():
    global L
    global Y
    global DFILE
    global TFILE
    with open (DFILE,"r") as file:
        temp=file.readline()
        flag=False
        while temp:
            if temp == "Y\n":
                flag=True
            else:
                tm=[]
                tempsplits=temp.split(',')
                for gh in tempsplits:
                    tm.append(float(gh))
                if flag==False:
                    L.append(tm)
                else:
                    for rt in tm:
                        Y.append(int(rt))
            temp=file.readline()
    with open (TFILE,"r") as file:
        temp=file.readline()
        flag=False
        while temp:
            if temp == "Y\n":
                flag=True
            else:
                tm=[]
                tempsplits=temp.split(',')
                for gh in tempsplits:
                    tm.append(float(gh))
                if flag==False:
                    T_L.append(tm)
                else:
                    for rt in tm:
                        T_Y.append(int(rt))
            temp=file.readline()

def prepTdata():
    global Y
    global L
    global T_L
    global T_Y
    global mnist
    factor=0.2
    tot,dim=np.shape(mnist.data)
    for i in range(tot):
        mnist.data=np.asarray(mnist.data,dtype=float)
        for j in range(dim):
            mnist.data[i,j]=float(float(mnist.data[i,j])/float(255))
        img=np.reshape(mnist.data[i,:], (28,28))
        #print("old: ",img)
	img=croping(img)
	mnist.data[i,:]=img.flatten()
	if i<NOT:     
            L.append(analyze(img))
	    Y.append(mnist.target[i])
	else:
	    T_L.append(analyze(img))
	    T_Y.append(mnist.target[i])        

if os.path.isfile(DFILE) ==True:
    print("Pre-training data exists...")
    try:
        ans=input("Type 1 to re train: ")
        if ans == 1:
            mnist=fetch_mldata('MNIST original')
            prepTdata()
            save2file()
        else:
            readfile()
    except:
        readfile()
else:
    mnist=fetch_mldata('MNIST original')
    prepTdata()
    save2file()

clf=svm.SVC(decision_function_shape='ovo', kernel='linear')
clf2=naive_bayes.MultinomialNB()
clf3=neighbors.KNeighborsClassifier(n_neighbors=5)
Lx,Ly=np.shape(L)

clf.fit(L,Y)
clf2.fit(L,Y)
clf3.fit(L,Y)
l,w=np.shape(T_L)
print (Lx,Ly,np.shape(Y),l,w)
correct=0
correct2=0
correct3=0
testnum=l
for i in range(l):
    c=clf.predict([T_L[i]])
    c2=clf2.predict([T_L[i]])
    c3=clf3.predict([T_L[i]])
    c=c[0]
    c2=c2[0]
    c3=c3[0]
    if str(c)==str(T_Y[i]):
	    correct+=1
    if str(c2)==str(T_Y[i]):
        correct2+=1
    if str(c3)==str(T_Y[i]):
        correct3+=1
#print(L)

print("CHaracters learned: ",len(clf.classes_))
print("Total number of training samples: ",Lx)
print("Support Vector MAchine Detection accuracy: ",float(float(correct)/float(testnum))*100)
print("Naive Bayes MultinomialNB Detection accuracy: ",float(float(correct2)/float(testnum))*100)
print("K-nearest neighbor accuracy: ",float(float(correct3)/float(testnum))*100)
