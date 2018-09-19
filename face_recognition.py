# -*- coding:utf8 -*-
"""
This script is used to perform face recognition using the computer vision library Dlib 19.4.99 \
and the deep learning framework keras with tensorflow as backend. The face recognition system inlude chinese character
encoding and decoding
"""
from keras.models import Sequential, model_from_json
from keras import backend as K
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.optimizers import SGD
import subprocess
import dlib
import cv2
import os
from sklearn.cross_validation import train_test_split
import h5py
import numpy as np
from sklearn.externals import joblib
import time
import tensorflow as tf
from cPickle import dump, load, HIGHEST_PROTOCOL
import sys
import pickle
import codecs
#import freetype
import copy
import pdb
import threading
import multiprocessing
from subprocess import call
import signal
import time
from threading import Thread
from multiprocessing.pool import ThreadPool
from concurrent.futures import ThreadPoolExecutor
from threading import Timer
import subprocess as sub
from subprocess import PIPE, Popen

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
reload(sys)

sys.setdefaultencoding('utf-8')

# ================================PARAMETER============================================
stop = False


# Select between 'cam','video' and 'image'.
class MyThread(threading.Thread):
	def __init__(self):
		self.stdout = None
		self.stderr = None
		threading.Thread.__init__(self)

def rect_to_bb(rect):
	# take a bounding predicted by dlib and convert it
	# to the format (x, y, w, h) as we would normally do
	# with OpenCV
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
	# return a tuple of (x, y, w, h)
	return (x, y, w, h)

class put_chinese_text(object):
	def __init__(self, ttf):
		self._face = freetype.Face(ttf)

	def draw_text(self, image, pos, text, text_size, text_color):
		'''
		draw chinese(or not) text with ttf
		:param image:     image(numpy.ndarray) to draw text
		:param pos:       where to draw text
		:param text:      the context, for chinese should be unicode type
		:param text_size: text size
		:param text_color:text color
		:return:          image
		'''
		self._face.set_char_size(text_size * 64)
		metrics = self._face.size
		ascender = metrics.ascender / 64.0
		# descender = metrics.descender/64.0
		# height = metrics.height/64.0
		# linegap = height - ascender + descender
		ypos = int(ascender)
		if not isinstance(text, unicode):
			text = text.decode('utf-8')
		img = self.draw_string(image, pos[0], pos[1] + ypos, text, text_color)
		return img

	def draw_string(self, img, x_pos, y_pos, text, color):
		'''
		draw string
		:param x_pos: text x-postion on img
		:param y_pos: text y-postion on img
		:param text:  text (unicode)
		:param color: text color
		:return:      image
		'''
		prev_char = 0
		pen = freetype.Vector()
		pen.x = x_pos << 6  # div 64
		pen.y = y_pos << 6

		hscale = 1.0
		matrix = freetype.Matrix(int(hscale) * 0x10000L, int(0.2 * 0x10000L), \
								 int(0.0 * 0x10000L), int(1.1 * 0x10000L))
		cur_pen = freetype.Vector()
		pen_translate = freetype.Vector()

		image = copy.deepcopy(img)
		for cur_char in text:
			self._face.set_transform(matrix, pen_translate)

			self._face.load_char(cur_char)
			kerning = self._face.get_kerning(prev_char, cur_char)
			pen.x += kerning.x
			slot = self._face.glyph
			bitmap = slot.bitmap

			cur_pen.x = pen.x
			cur_pen.y = pen.y - slot.bitmap_top * 64
			self.draw_ft_bitmap(image, bitmap, cur_pen, color)

			pen.x += slot.advance.x
			prev_char = cur_char

		return image

	def draw_ft_bitmap(self, img, bitmap, pen, color):
		'''
		draw each char
		:param bitmap: bitmap
		:param pen:    pen
		:param color:  pen color e.g.(0,0,255) - red
		:return:       image
		'''
		x_pos = pen.x >> 6
		y_pos = pen.y >> 6
		cols = bitmap.width
		rows = bitmap.rows

		glyph_pixels = bitmap.buffer

		for row in range(rows):
			for col in range(cols):
				if glyph_pixels[row * cols + col] != 0:
					img[y_pos + row][x_pos + col][0] = color[0]
					img[y_pos + row][x_pos + col][1] = color[1]
					img[y_pos + row][x_pos + col][2] = color[2]

input_data = 'camera'

# Set tolerance for face detection smaller means more tolerance for example -0.5 compared with 0
tolerance = -0.5

# Acquire face data 
acquire_data= False
person_name_list = [ ]

database_dir = 'face_db/'

# The directory of the trained neural net model
model_dir = 'models/'

hdf5_filename = 'face_recog_special_weights.hdf5'
json_filename = 'face_recog_special_arch.json'
labeldict_filename = 'label_dict_special.pkl'

# DLIB's model path for face pose predictor and deep neural network model
predictor_path = 'shape_predictor_68_face_landmarks.dat'
face_rec_model_path = 'dlib_face_recognition_resnet_model_v1.dat'

# .pkl file containing dictionary information about person's label corresponding with neural network output data
label_dict = joblib.load(model_dir + labeldict_filename)

# ML Parameters

batch_size = 150
nb_epoch = 200
loss = 'categorical_crossentropy'
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

temp_data = dict()

for person in person_name_list:
	temp_data[person] = dict()
	beg = time.time()
	temp_data[person]['data'] = joblib.load(target_dir + person.decode('utf8') + '/face_descriptor.pkl')
	temp_data[person]['count'] = 0

json_model_file = open(model_dir + json_filename, 'r')
json_model = json_model_file.read()
json_model_file.close()

cnn_model = model_from_json(json_model)
cnn_model.load_weights(model_dir + hdf5_filename)
end = time.time()

cnn_model.compile(loss=loss,
				  optimizer=sgd,
				  metrics=['accuracy'])

tracker = dlib.correlation_tracker()
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)
cap = cv2.VideoCapture(0)
cap.set(3, 340)
cap.set(4, 240)

start = time.time()
facerec = dlib.face_recognition_model_v1(face_rec_model_path)
global trackingFace
if input_data == 'camera':

	while (True):
		ret, frame = cap.read()
		start = time.time()
	
		detects, scores, idx = detector.run(frame, 0, tolerance)
		
		for i, d in enumerate(detects):

			if idx[i] == 0 or idx[i] == 1 or idx[i] == 2 or idx[i] == 3 or idx[i] == 4:
				start_descriptor = time.time()
				(x, y, w, h) = rect_to_bb(d)
				bb1 = tracker.get_position()
				bb1 = dlib.rectangle(int(bb1.left()), int(d.top()), int(d.right()), int(d.bottom()))
				bb = [bb1]
				tracker.start_track(frame, dlib.rectangle(bb[0].left(), bb[0].top(), bb[0].right(), bb[0].bottom()))
				shape = sp(frame, d)
				face_descriptor = np.array([facerec.compute_face_descriptor(frame, shape)])
				end_descriptor = time.time()
				delta_descrip = end_descriptor - start_descriptor
				# print delta_descrip
				start_predict = time.time()
				prediction = cnn_model.predict_proba(face_descriptor)
				end_predict = time.time()
				delta_predict = end_predict - start_predict
				print delta_predict
				highest_proba = 0
				counter = 0
				# print prediction
				for prob in prediction[0]:
					if prob > highest_proba and prob >= 0.45:
						highest_proba = prob
						label = counter
						label_prob = prob
						face_label_name = label_dict[label]

						if face_label_name in person_name_list and acquire_data:
							temp_data[face_label_name]['data'] = np.append(temp_data[face_label_name]['data'], face_descriptor,
																	axis=0)
							temp_data[face_label_name]['count'] += 1
							if temp_data[face_label_name]['count'] == 5:
								dump(temp_data[face_label_name]['data'], target_dir + face_label_name + '/face_descriptor.pkl', rb,
									 HIGHEST_PROOCOL)
								temp_data[face_label_name]['count'] = 0
							cv2.rectangle(frame, (d.left(), d.bottom()), (d.right(), d.bottom()), (0, 0, 255),
										  cv2.FILLED)
							cv2.rectangle(frame, (d.left(), d.top()), (d.right(), d.bottom()), (0, 0, 255), 2)

					if counter == (len(label_dict) - 1) and highest_proba == 0:
						label = label_dict[counter]
						label_prob = prob
						face_label_name = label
					counter += 1

				font = cv2.FONT_HERSHEY_DUPLEX
				l = [face_label_name]
				color_ = (0, 0, 255)
				pos = (d.left() - 240, d.bottom() - 240)
				text_size = 36

				if label != 'UNKNOWN':
					cv2.rectangle(frame, (d.left(), d.top()), (d.right(), d.bottom()), (0, 0, 255), 2)
					cv2.rectangle(frame, (d.left(), d.bottom() - 45), (d.right(), d.bottom()), (0, 0, 255), cv2.FILLED)
					

				thread = MyThread()
				thread.start()
				

		cv2.namedWindow('face detect', flags=cv2.WINDOW_NORMAL)
		cv2.imshow('face detect', frame)
		
cap.release()
cv2.destroyAllWindows()
