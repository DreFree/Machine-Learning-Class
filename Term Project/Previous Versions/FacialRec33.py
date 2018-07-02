import numpy as np
import dlib
import os
from skimage import io
from sklearn import svm
import sys

TFILE="traindata.csv"
MPATH="Mugshot/"
PPATH="Test/"
NOP=68
L=[]
Y=[]
persons=0

def facialPoints(img):
    predictor_path="shape_predictor_68_face_landmarks.dat"
    detector=dlib.get_frontal_face_detector()
    predictor=dlib.shape_predictor(predictor_path)
    dets=detector(img,1)

    for k, d in enumerate(dets):
        shape=predictor(img,d)
    
    minx=shape.part(0).x
    maxx=shape.part(0).x
    miny=shape.part(0).y
    maxy=shape.part(0).y
    
    for b in range(NOP):
        if shape.part(b).x<minx:
            minx=shape.part(b).x
        if shape.part(b).y<miny:
            miny=shape.part(b).y
        if shape.part(b).x>maxx:
            maxx=shape.part(b).x
        if shape.part(b).y>maxy:
            maxy=shape.part(b).y

    vec=np.empty([NOP,2],dtype=float)
    for b in range(NOP):
        vec[b][0]=float(shape.part(b).x-minx)/float(maxx-minx)
        vec[b][1]=float(shape.part(b).y-miny)/float(maxy-miny)
    ##/(minx, miny, maxx, maxy) IS to Normalize the points to be 
    ##In the range 0. to 1. for all images
    return vec

def loadData ():
    global L
    global Y
    global persons
    global MPATH
    try:
        ##Read in filenames from the mugshot directory
        filenames=os.listdir(MPATH)
    except:
        print("File PAth doesnt exxist")
        return 1
    

    if filenames==[]:
        print("No training data")
        return 1
    prev=[]
    print("Loading training data...")
    print("Setting up training data structure")

    for let in filenames[0]:
        if let == '_':
            break
        else:
            prev.append(let)
    filenames=sorted(filenames)
    for filename in filenames:
        ## from the list of filenames read in each filename one by one
        cur=[]
        for let in filename:
            if let == '_':
                break
            else:
                cur.append(let)
        ##And directory path to filename
        filename=MPATH+filename
        img=io.imread(filename)
        #img = dlib.load_rgb_image(filename)

        points=facialPoints(img)

        ##Checks if the current face image belongs to the same as the previous face image
        if cur != prev:
            prev=cur
            persons+=1 
            
        ##Create the data structure for each point 0- 67 (68) points for the data training
        L.append(points.flatten())
        Y.append(persons)

    return 0

def save2file():
    global L
    global Y
    with  open (TFILE, "w") as file:
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
    return
def readfile():
    global L
    global Y
    global persons

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
                    L.append(tm)
                else:
                    for rt in tm:
                        Y.append(int(rt))
            temp=file.readline()
##Initialize the SVM for 68 points and train each point
clf=svm.SVC(decision_function_shape='ovo', kernel='linear')

########################################################
try: 
    p_filenames=os.listdir(PPATH)
except:
    print("Test file path does not exist")
    sys.exit(1)
if p_filenames==[]:
    print("No data to predict")
    sys.exit(1)

if os.path.isfile(TFILE) ==True:
    print("Pre-training data exists...")
    try:
        ans=input("Type 1 to re train: ")
        if ans == 1:
            if loadData()!=0:
                print("Error Loading/setting up data structure")
                sys.exit(0)
            save2file()
        else:
            readfile()
    except:
        readfile()
else:
    if loadData()!=0:
        print("Error Loading/setting up data structure")
        sys.exit(0)
    save2file()

#print(np.shape(L),np.shape(Y))
##Fitting data for all points and obtaining predicts the result from the test image face for each point
clf.fit(L,Y)

testnum=0
correct=0
for p_filename in p_filenames:
    testnum+=1
    p_file=PPATH+p_filename
    img=io.imread(p_file)
    p_points=facialPoints(img)
    p_fixed_points=[p_points.flatten()]


    c=clf.predict(p_fixed_points)
    pn=""
    for vv in p_filename:
        if vv!='.':
            pn+=vv
        else:
            break
    c=c[0]
    print("Predicted person is: ",c," -- Actual person is: ",pn)
    
    if  str(c)==str(pn):
        correct+=1

Lx,Ly=np.shape(L)
print()
#print(clf.classes_)
print("Persons learned: ",len(clf.classes_))
print("Total number of training samples: ",Lx*Ly)
print("Detection accuracy: ",float(float(correct)/float(testnum))*100)