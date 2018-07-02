import sys
import dlib
import cv2
import openface
import os
from sklearn.svm import SVC
import numpy as np

from PIL import Image
import glob
image_list = []
filename_list = []
labels = []

for x in range (50):
	completepath = ['C:\Python36-32\Facial Recognition\FaceDatabase/', str(x+1)]
	path = ''.join(completepath)
	##print(path)
	
	for filename in glob.glob(os.path.join(path, '*.jpg')): #assuming gif
		im= cv2.imread(filename)
		im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
		image_list.append(im)
		filename_list.append(filename)
		labels.append(x+1)


predictor_model = "shape_predictor_68_face_landmarks.dat"


# Create a HOG face detector using the built-in dlib class
face_detector = dlib.get_frontal_face_detector()
face_pose_predictor = dlib.shape_predictor(predictor_model)
face_aligner = openface.AlignDlib(predictor_model)



# Load the images
for x in range (len(image_list)):
	image = image_list[x]
	##image = cv2.imread("Iprime.png")
	

	# Run the HOG face detector on the image data
	detected_faces = face_detector(image, 1)

	##print("Found {} faces in the image file {}".format(len(detected_faces), filename_list[x]))

	# Loop through each face we found in the image
	for i, face_rect in enumerate(detected_faces):

		# Detected faces are returned as an object with the coordinates 
		# of the top, left, right and bottom edges
		print("- Face #{} found at Left: {} Top: {} Right: {} Bottom: {}".format(i, face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom()))

		# Get the the face's pose
		pose_landmarks = face_pose_predictor(image, face_rect)

		# Use openface to calculate and perform the face alignment
		alignedFace = face_aligner.align(534, image, face_rect, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
		# Save the aligned image to a file
		imageSAVE = Image.fromarray(alignedFace, mode='RGB')
		path = ["C:\Python36-32\Facial Recognition\ALIGNED/aligned_face_", str(int(x/12)+1),"_", str((x%12)+1), ".jpg"]
		path = ''.join(path)
		os.makedirs(os.path.dirname(path), exist_ok=True)
		print(path)
		imageSAVE.save(path)
		##cv2.imwrite("aligned_face_{}{}.jpg".format(i, x), alignedFace)

ftp = open("labels.txt", "w")
for x in labels:
	ftp.write("%s\n" % str(x))
