from __future__ import print_function
from PIL import Image
from numpy import genfromtxt
import gzip, cPickle
from glob import glob
import numpy as np
import pandas as pd

from pandas import DataFrame
import numpy as np

np.random.seed(1337)  # for reproducibility

from datetime import datetime
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K


def dir_to_dataset(loc_train_labels=""):

    gambar = pd.read_csv(loc_train_labels)
    idxGambar = np.array(gambar["filename"])
    
    return len(idxGambar);
    
totData = dir_to_dataset("/preprocessedData/metadata/image_labels.txt")

# Data and labels are read 
xx = totData

print (xx)
