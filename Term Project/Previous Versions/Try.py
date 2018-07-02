import numpy as np
import dlib
from skimage import io
from sklearn.datasets import make_classification
from imutils import face_utils

def facialPoints(img):
    #USE:
    size=np.shape(img)
    predictor_path="shape_predictor_68_face_landmarks.dat"

    detector=dlib.get_frontal_face_detector()

    predictor=dlib.shape_predictor(predictor_path)

    dets=detector(img,1)
    
    for k, d in enumerate(dets):
        shape=predictor(img,d)
        (x, y, w, h) = face_utils.rect_to_bb(d)
    vec=np.empty([68,2],dtype=float)
    print(x,y,w,h)
    minx=shape.part(0).x
    maxx=shape.part(0).x
    miny=shape.part(0).y
    maxy=shape.part(0).y
    for b in range(68):
        if shape.part(b).x<minx:
            minx=shape.part(b).x
        if shape.part(b).y<miny:
            miny=shape.part(b).y
        if shape.part(b).x>maxx:
            maxx=shape.part(b).x
        if shape.part(b).y>maxy:
            maxy=shape.part(b).y
    for b in range(68):
        vec[b][0]=float(shape.part(b).x-minx)/float(maxx-minx)
        vec[b][1]=float(shape.part(b).y-miny)/float(maxy-miny)
    ##/(size[0]) IS to Normalize the points to be 
    ##In the range 0. to 1. for all images
    return vec
L=[]
img=io.imread("example.jpeg")
#value=facialPoints(img)
#for x ,y in value:
    #L.append([[x,1],[y,-1]])
#print(L[0])
print(make_classification(n_features=4,random_state=0))