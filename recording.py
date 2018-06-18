"""
###################################################################################
Python script  used to save facial features of a person using dlib face shape model
and deep neural network model to extract face features 
###################################################################################
"""

import cv2
import os
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.externals import joblib
import dlib
import time
import shutil
import pickle

# Set the output directory of the user's data
database_dir = 'face_db/'

if not os.path.exists(database_dir):
	os.makedirs(database_dir)

# This is the number of maximum person to be stored in the database
maximum_pers=5000

# DLIB's model path for face pose predictor and deep neural network model
predictor_path='shape_predictor_68_face_landmarks.dat'
face_rec_model_path='dlib_face_recognition_resnet_model_v1.dat'


detector = dlib.get_frontal_face_detector()
dlib_shape_predictor = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)\

cap = cv2.VideoCapture(1)

saving_data = False

temp_data=[]

while (True):

	name = str(raw_input("Enter your name:  "))
	directory = tdatabase_dir+name+'/'

	if not os.path.exists(directory):
		dir_list = os.listdir(database_dir)
		
		if len(dir_list) >= maximum_pers:
			print "Too many recorded persons (%i users limitation exceeded)! Choose user's data to delete: "%(maximum_pers)
			
			for user_count in range(0,len(dir_list)):
				print '%i : %s'%(user_count+1,dir_list[user_count])
			
			try:
				delete_person=int(raw_input('Choose user (number) to delete: '))
				chosen_user_dir=dir_list[delete_person-1]
				shutil.rmtree(target_dir+chosen_user_dir)
			except:
				print " Wrong input! Please try again!"
				continue
		os.makedirs(directory)
		break
	else:
		print 'Name already exist! Please try again!'

while (True):
	
	ret, frame = cap.read()
	clone_frame = frame.copy()
	start=time.time()
	is_eligible_frame=False

	detects,scores,idx = detector.run(frame, 0,0)

	for i, d in enumerate(detects):
			
		if len(idx)==1 and saving_data:
			shape = dlib_shape_predictor(frame, d)
			face_descriptor = np.array([facerec.compute_face_descriptor(frame, shape)])
			if len(temp_data)==0:
				temp_data=face_descriptor
			else:
				temp_data=np.append(temp_data,face_descriptor,axis=0)
			is_eligible_frame=True
			color=(0,255,0)
		elif len(idx)!=1 and saving_data:
			color=(0,0,255)
		else:
			color=(255,0,0)
		cv2.rectangle(clone_frame,(d.left(),d.top()),(d.right(),d.bottom()),color,2)

	if is_eligible_frame:
		scaled_frame=cv2.resize(frame,(int(frame.shape[1]/3),int(frame.shape[0]/3)))

		detects,scores,idx = detector.run(scaled_frame, 0,-0.5)

		for i, d in enumerate(detects):
			if len(idx)==1:
				shape = dlib_shape_predictor(scaled_frame, d)
				face_descriptor = np.array([facerec.compute_face_descriptor(scaled_frame, shape)])
				temp_data=np.append(temp_data,face_descriptor,axis=0)


	cv2.imshow('face detect',clone_frame)
	if cv2.waitKey(1) & 0xFF == ord('r'):
		if saving_data==True:
			saving_data=False
			print temp_data
			print len(temp_data)
			with open(directory+'/face_descriptor.pkl', 'wb') as f1:
			
			# Save each face feature descriptor  data  to pickle (.pkl)
				pickle.dump(temp_data,f1, protocol=pickle.HIGHEST_PROTOCOL)
			break
		else:
			saving_data=True
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
	delta=time.time()-start
	fps=float(1)/float(delta)
	print(fps)

# release the capture
cap.release()
cv2.destroyAllWindows()

