from __future__ import print_function
from PIL import Image
from numpy import genfromtxt
import gzip, cPickle
from glob import glob
import numpy as np
import pandas as pd
import csv

from pandas import DataFrame
import numpy as np

np.random.seed(1337)  # for reproducibility

from datetime import datetime
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

def dir_to_dataset(mypath, loc_train_labels,startP,endP):
    dataset = []
    
    gbr = pd.read_csv(loc_train_labels, sep="\t")
    idxGambar = gbr["filename"].values[startP:endP]

    #for file_count, file_name in enumerate( sorted(glob(glob_files),key=len) ):
    for i in range(0,len(idxGambar)):
        image = Image.open(mypath + idxGambar[i])
        img = Image.open(mypath + idxGambar[i]).convert('LA') #tograyscale
        pixels = [f[0] for f in list(img.getdata())]
        dataset.append(pixels)
    # outfile = glob_files+"out"
    # np.save(outfile, dataset)
    if len(loc_train_labels) > 0:
        df = pd.read_csv(loc_train_labels)
        return np.array(dataset), gbr["class"].values[startP:endP]
    else:
        return np.array(dataset)

batch_size = 50
nb_classes = 2
nb_epoch = 1

# input image dimensions
img_rows, img_cols = 224, 224
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

model = Sequential()
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                    border_mode='valid',
                    input_shape=(1,224,224)))
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

model.compile(loss='categorical_crossentropy',
            optimizer='rmsprop',
            metrics=['accuracy'])

#init_weight = model.get_weights()
#model.set_weights(init_weight)
    

gbar = pd.read_csv('/preprocessedData/metadata/image_labels.csv', sep=",")    
lengam = len(gbar["filename"].values)

num_iter = 50

left_data = lengam%num_iter
if left_data==0:
    n_iter = (lengam/num_iter)
else:
    n_iter = (lengam/num_iter)+1

Total_Tes_Score = 0
Total_Tes_Acc = 0
Total_Data = 0    
for k in range(0,10):
    for i in range(0,n_iter):    
        Data, y = dir_to_dataset("/preprocessedData/images/","/preprocessedData/metadata/image_labels.csv",(i*num_iter),(i*num_iter+num_iter))
        #print (i*50)
        #print (i*50+50)
        # Data and labels are read
        xx = len(Data) 
        train_set_x = Data[:(xx*4)/5]
        val_set_x = Data[(xx*4)/5:]
        train_set_y = y[:(xx*4)/5]
        val_set_y = y[(xx*4)/5:]

        # the data, shuffled and split between train and test sets
        (X_train, y_train), (X_test, y_test) = (train_set_x,train_set_y),(val_set_x,val_set_y)

        if K.image_dim_ordering() == 'th':
            X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
            X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
            input_shape = (1, img_rows, img_cols)
        else:
            X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
            X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
            input_shape = (img_rows, img_cols, 1) 

        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        X_train /= 255
        X_test /= 255
    #print('X_train shape:', X_train.shape)
    #print(X_train.shape[0], 'train samples')
    #print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
        Y_train = np_utils.to_categorical(y_train, nb_classes)
        Y_test = np_utils.to_categorical(y_test, nb_classes)

        model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(X_test, Y_test))
        score = model.evaluate(X_test, Y_test, verbose=0)
        #print('Test score:', score[0])
        #print('Test accuracy:', score[1])
        Total_Tes_Score += score[0]*X_test.shape[0]
        Total_Tes_Acc += score[1]*X_test.shape[0]
        Total_Data += X_test.shape[0]

        upd_weight = model.get_weights()
        model.set_weights(upd_weight)

print('Rata2 Loss: ',Total_Tes_Score*1.0/Total_Data)
print('Rata2 Acc: ',Total_Tes_Acc*1.0/Total_Data)

model.save_weights('/modelState/CNNweightsJPG.h5')
