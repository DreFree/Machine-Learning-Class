#Created by: Andre Freeman
#Last Updated: 28/4/2018
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random as rand
from sklearn.linear_model import SGDClassifier

###############################################################
##Read in image and convert to a 1-D array using flattern function
k1=cv2.imread('key1.png',0)
K1=k1.flatten()

k2=cv2.imread('key2.png',0)
K2=k2.flatten()

img=cv2.imread('I.png',0)
I=img.flatten()

m=cv2.imread('E.png',0)
E=m.flatten()
#######################################################################
##Initialize height and width from input image
W,H=img.shape[:2]

##Declare other important variables
max_iter=100
alpha=0.00001
epsilon=0.01
Epoch=1

##Initialize the weight array 'w' to have width * height +1 elements
w=[[0]*3 for i in range(H*W+1)]
#Random starting value for the first weight between -1 to 1
a=rand.randint(0,2)-1
b=rand.randint(0,2)-1
c=rand.randint(0,2)-1
w[0][0]=a
w[0][1]=b
w[0][2]=c
#w[0]=[0.,0.,0.]
#w[1]=[1.,1.,1.]


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
print("Starting w:")
print(w[0])

##Function to calculate the difference between w[Epoch] and w[Epoch-1]
def epi(a,b):
    dif=0
    for i in range(0,3):
        dif+=abs(a[i]-b[i])
    return dif

##Altered Function to Multiply to Matrix since multiplication is 3 by 1 agaisnt a 1 by 3
def A(w,x):
    a=0
    for i in range(0,3):
            a+=w[i]*x[i]#abs or no abs???
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

##Testing
min=0.

while (Epoch==1 or (Epoch<max_iter and abs(epi(w[Epoch],w[Epoch-1]))>epsilon)):
    print("Epoch:")
    print(Epoch)
    for k in range(0,H*W):#Range is only 0 - 19 now for testing sake
        
        x=np.array([K1[k],K2[k],I[k]])#Let x be a matrix combined of K1, K2 and I
        a=A(w[k],x)
        e=E[k]-a
        w[k+1]=plus(w[k],S(x,(alpha*e)))
    print("epi: ")
    
    Epoch+=1 #Epoch++ doesnot work yet since this is not set up to be 3-D yet
    print(epi(w[Epoch],w[Epoch-1]))
    

bestw=[w[Epoch-1]]
eprime=cv2.imread('Eprime.png',0)
Eprime=eprime.flatten()
Iprime=np.zeros(H*W)

for i in range (0,H*W):
    if w[i][2]==0:
         Iprime[i]=0
    else:
        Iprime[i]=(Eprime[i]-(w[i][0]*K1[i])-(w[i][1]*K2[i]))/w[i][2]

iprime=np.reshape(Iprime,(-1,400))
print(eprime.shape[:2])
print(iprime.shape[:2])
#print(len(iprime))
#print(len(iprime[0]))
print(Iprime)
cv2.imwrite('Iprime.png',iprime)