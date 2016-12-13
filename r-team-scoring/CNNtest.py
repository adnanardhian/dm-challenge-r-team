
from __future__ import print_function
import numpy as np

np.random.seed(1337)  # for reproducibility

from datetime import datetime
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from sklearn.metrics import accuracy_score
import pandas as pd
from PIL import Image

import h5py


def dir_to_dataset(mypath, loc_train_labels=""):
    dataset = []
    
    gbr = pd.read_csv(loc_train_labels, sep="\t")
    idxGambar = gbr["filename"].values

    for i in range(0,len(idxGambar)):
        image = Image.open(mypath + idxGambar[i])
        img = Image.open(mypath + idxGambar[i]).convert('LA') #tograyscale
        pixels = [f[0] for f in list(img.getdata())]
        dataset.append(pixels)
    # outfile = glob_files+"out"
    # np.save(outfile, dataset)
    if len(loc_train_labels) > 0:
        df = pd.read_csv(loc_train_labels)
        return np.array(dataset), gbr["class"].values
    else:
        return np.array(dataset)

#Getting Data
Data, y = dir_to_dataset("DM/20Nov/","DM/label20.csv")

#num_of Classes
nb_classes = 2

# input image dimensions
img_rows, img_cols = 224, 224
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

#Check whether the backend is Theano or Tensorflow
if K.image_dim_ordering() == 'th':
    X_Data = Data.reshape(Data.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_Data = Data.reshape(Data.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1) 

#Do Something to these Images
X_Data = X_Data.astype('float32')
X_Data /= 255

# convert class vectors to binary class matrices
Y_Data = np_utils.to_categorical(y, nb_classes)

#Define the Model
model = Sequential()
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], border_mode='valid',input_shape=input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

#Load the saved Weights
model.load_weights('DM/CNNweightsJPG.h5', by_name=True)

model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

#Evaluate the loss and accuracy
score = model.evaluate(X_Data,Y_Data)
print ('Loss : ',score[0])
print ('Accuracy : ',score[1])

