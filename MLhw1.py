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

##Testing read in and flattern function
print ("W:")
print(W)
print ("H:")
print(H)
print("Starting w:")
print(w[0])

##Function to calculate the difference between w[Epoch] and w[Epoch-1]
def epi(a,b):
    dif=0
    for i in range(0,3):
        dif+=a[i]-b[i]
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

##################################################################################
##Finding w1, w2 and w3 
while (Epoch==1 or (Epoch<max_iter and abs(epi(w[Epoch],w[Epoch-1]))>epsilon)):
    for k in range(0,H*W):#Range is only 0 - 19 now for testing sake
        x=np.array([K1[k],K2[k],I[k]])#Let x be a matrix combined of K1, K2 and I
        a=A(w[k],x)
        e=E[k]-a
        w[k+1]=plus(w[k],S(x,(alpha*e)))
    Epoch+=1

################################################################3
##Decrypting the Eprime image
##Load Eprime image and flatten to a 1D array
eprime=cv2.imread('Eprime.png',0)
Eprime=eprime.flatten()

##Declare a 1D new data structure for the decrypted image 'Iprime'
Iprime=np.zeros(H*W)

##Mathematical operation for calculating Iprime
for i in range (0,H*W):
    if w[i+1][2]==0: #Cant div by 0
        Iprime[i]=255
    else:
        Iprime[i]=round((Eprime[i]-(w[i+1][0]*K1[i])-(w[i+1][1]*K2[i]))/w[i+1][2],0)

##Restructure Iprime to being a 2D array named 'iprime'
iprime=np.reshape(Iprime,(-1,400))

##Write back Iprime to computer memory
cv2.imwrite('Iprime.png',iprime)
print("Complete")