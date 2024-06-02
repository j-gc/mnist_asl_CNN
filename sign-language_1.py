import pandas as pd
import numpy as np
import os
from random import shuffle
from tqdm import *
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import SGD

original_limits = [0,255]
new_limits = [0,1]

path = 'sign_mnist_train.csv'

def read_data(path):
	# reads the csv file with pandas
	df = pd.read_csv(path)
	df_copy = df.copy()
	pixel_vals = df_copy.iloc[:,1:].copy()
	if not os.path.isfile("standardized.csv"):
		for pixel_val in pixel_vals.columns:
			pixel_vals[pixel_val] = pixel_vals[pixel_val].apply(lambda x:float(normalize(x=x)))
			
		df_copy.iloc[:,1:] = pixel_vals
		df_copy.to_csv('standardized.csv',index=False)
	x_t = pixel_vals
	y_t = df_copy['label']
	# print(y_t)
	# plt.plot(x_t,y_t)
	# plt.show()
	print("Created standardized.csv")

def get_pixel_vals():
	df = pd.read_csv("standardized.csv")
	pixel_vals = df.iloc[:,1:].values
	return pixel_vals

def get_label_vals():
	df = pd.read_csv("standardized.csv")
	label_vals = df["label"]
	print(label_vals)
	return label_vals


def modeling():
	x_train = get_pixel_vals()
	y_train = get_label_vals()
	model = Sequential()
	model.add(Dense(50, activation='softmax'))
	model.add(Dense(25, activation='relu'))
	model.compile(optimizer=SGD(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
	model.fit(x_train,y_train,epochs=5, verbose=1)
	      # save model to JSON
	model_json = model.to_json()
	with open("mlp_model.json", "w") as json_file:
		json_file.write(model_json)
	#save weights to HDF5
	model.save_weights("mlp_weights.h5")
	model.save("mlp_asl.h5")
	print("saved model and trained weights")


def normalize(x):
	""" Take list of original min and max and map it linearly between desired limits """
	# linearly map new_limits to original_limits
	#              [0,1]         [0,255]
	slope = (new_limits[1] - new_limits[0])/(original_limits[1] - original_limits[0])
	constant = new_limits[0] - slope * (original_limits[0])
	return (slope * x) + constant


read_data(path)
modeling()
