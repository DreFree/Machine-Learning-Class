import numpy as np
import dlib
import os
from skimage import io
from sklearn import svm
import sys

NOP=68
L=[]
Y=[]
counter=0

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

    vec=np.empty([68,2],dtype=float)
    for b in range(NOP):
        vec[b][0]=float(shape.part(b).x-minx)/float(maxx-minx)
        vec[b][1]=float(shape.part(b).y-miny)/float(maxy-miny)
    ##/(minx, miny, maxx, maxy) IS to Normalize the points to be 
    ##In the range 0. to 1. for all images
    return vec

def loadData ():
    global L
    global Y
    global counter
    PATH="/home/project/Documents/Machine Learning/Project/Code/Mugshot/"
    try:
        ##Read in filenames from the mugshot directory
        filenames=os.listdir(PATH)
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
        filename=PATH+filename
        img=io.imread(filename)
        #img = dlib.load_rgb_image(filename)

        points=facialPoints(img)

        ##Checks if the current face image belongs to the same as the previous face image
        if cur != prev:
            prev=cur
            counter+=1 
            
        ##Create the data structure for each point 0- 67 (68) points for the data training
        i=0
        for x ,y in points: 
            L[i].append([x,y])
            Y[i].append(counter)
            i+=1

        ## Error detection for if the number of points not equalling to 68 points
        if i!= 68:
            print("Points not total to 68 points")
            return 1
    return 0


##Initialize the SVM for 68 points and train each point
clf=[]
for i in range(NOP):
    L.append([])
    clf.append(svm.SVC(decision_function_shape='ovo', kernel='linear',gamma=0.75))
    Y.append([])
########################################################


try: 
    img=io.imread("predict.jpg")
    
except:
    print("Predict image not found")
    sys.exit(1)

if loadData()!=0:
    print("Error Loading/setting up data structure")
    sys.exit(0)

p_points=facialPoints(img)
p_fixed_points=[]
for x ,y in p_points: 
    p_fixed_points.append([[x,y]])

##Fitting data for all points and obtaining predicts the result from the test image face for each point
c=[]
#print(np.shape(L[i]),np.shape(Y[i]),np.shape(p_fixed_points[i]))

for i in range(NOP):
    #print(L[i])
    #print(p_fixed_points[i])
    clf[i].fit(L[i],Y[i])
    c.append(clf[i].predict(p_fixed_points[i]))
    

##TALLy up the r=results from each 68 SVM
c=np.array(c)
c=c.flatten()
print("Persons learned: ",counter+1)

cl=[]
for m in range(counter+1):
    cl.append(0)
for a in c:
    cl[a]+=1
#print(c)
##
##Find max of the TALLY
print("CL:",cl)
max=[0,cl[0]]
i=0
sum=0
for f in cl:
    sum+=f
    if max[1]<f:
        max[0]=i
        max[1]=f
    i+=1

##DIsplay the results with percentage certainty
print("This person is: ",max[0])
print(((float(max[1])/float(sum))*100),"% sure.")

