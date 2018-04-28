#Created by: Andre Freeman
#Last Updated: 28/4/2018
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier

###############################################################
##Read in image and convert to a 1-D array using flattern function
k1=cv2.imread('key1.png',0)
K1=k1.flatten()

k2=cv2.imread('key1.png',0)
K2=k2.flatten()

i=cv2.imread('I.png',0)
I=i.flatten()

m=cv2.imread('E.png',0)
E=m.flatten()
#######################################################################
##Initialize hieght width from input image
W,H=i.shape[:2]

##Initialize the weight array 'w' to contain decimal (float) values
w=np.array([[0.,0.,0.],[1.,1.,1.]])

##Declare other important variables
max_iter=100
alpha=0.00001
epsilon=alpha
Epoch=1

##Testing read in and flattern function
print ("W:")
print(W)
print ("H:")
print(H)

print ("K1: ")
print (K1[0])
print("K2: ")
print (K2[0])
print ("I: ")
print (I[0])


##Function to calculate the difference between w[Epoch] and w[Epoch-1]
def epi(a,b):
    len=a.size
    dif=0
    for i in range(0,len):
        dif+=abs(a[i]-b[i])
    return dif

##Altered Function to Multiply to Matrix since multiplication is 3 by 1 agaisnt a 1 by 3
def A(w,x):
    a=0
    for i in range(0,3):
        a+=w[i]*x[i]
    return a

##Fuction to subtract a constant from a Matrix
def Ex(B,a):
    result=np.array([0.,0.,0.])
    for i in range(0,3):
        result[i]=B[i]-a
    return result

##Function to multiply a Matrix by a constant
def S(A,a):
    result=np.array([0.,0.,0.])
    for i in range (0,3):
        result[i]=A[i]*a
    return result

##Function to Add two matricies
def plus(A,B):
    C=np.array([0.,0.,0.])
    for i in range(0,3):
        C[i]=A[i]+B[i]
    return C      


while (Epoch==1 and (Epoch<max_iter and epi(w[Epoch],w[Epoch-1])>epsilon)):
    for k in range(0,20):#Range is only 0 - 19 now for testing sake
        x=np.array([K1[k],K2[k],I[k]])#Let x be a matrix combined of K1, K2 and I
        print ("x:")
        print (x)
        a=A(w[Epoch],x)
        print ("a: ")
        print (a)
        e=E[k]-a
        print ("e: ")
        print (e)
        oldw=w.copy()
        w[Epoch]=plus(w[Epoch],S(x,(alpha*e)))
        print (oldw[Epoch])
        print  (w[Epoch])
    Epoch=Epoch+1 #Epoch++ doesnot work yet since this is not set up to be 3-D yet

