""" Import Modules """

""" Loads data in the new format with new and old processing combined 
#https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

"Building powerful image classification models using very little data"

In our setup, we:
- created a data/ folder
- created train/ and validation/ subfolders inside data/
- created GR2/ and GR3/ subfolders inside train/ and validation/
"""

""" Path to data"""

import preprocessing
import utils

import os

# Machine learning modules
import numpy as np
import keras
import os.path as path
from keras.preprocessing.image import ImageDataGenerator

from keras import optimizers
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

""" Data-augmentation """

class cnn(object):
    def init(self):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        model.compile(loss='binary_crossentropy',
                    optimizer='rmsprop',
                    metrics=['accuracy'])

        model.summary()



from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')


def get_path(computer_str, directory):
    """
    Returns the correct path based on where the program is run
    """

    mac_data_path = '/Users/justinwhatley/Dropbox/FevensLab'
    linux_data_path = '/Users/justinwhatley/Dropbox/FevensLab'
    
    if computer_str.lower() == 'mac':
        return os.path.join(mac_data_path, 'FNAB_raw')

    elif computer_str.lower() == 'linux': 
        return os.path.join(linux_data_path, 'FNAB_raw')

    elif computer_str.lower() == 'colab':
        # Google colab path
        print('Not yet implemented')
        exit(0)

    else: 
        print('Incorrect base path option')
        exit(0)


def train():
    """
    """
    pass

def validate():
    """
    Compute validation score
    """
    pass

def remove_training_validation_files(directory):
    """
    Remove the directory containing the validation and training set files that were assigned for k-fold validation 
    Note. Original files are still available 
    """
    pass



if __name__ == "__main__":

    raw_data_directory_name = FNAB_raw
    raw_data_directory_path = get_path('mac', FNAB_raw)
    # raw_data_directory_path = get_path('linux', FNAB_raw)

    # Prepares data
    height, width = 224, 224
    file_lists_by_class = preprocessing.prepare_datasets(raw_data_path, height, width)

    preprocessed_path = os.path.join(mac_data_path, ' ')

    for _class in file_lists_by_class:
        for i, fold in enumerate(_class)
            validation_fold = i
            # Move the list of selected files to a training and validation folders
            preprocessing.assign_folds_to_training_and_validation(validation_fold, _class)
            # Run training, validation while keeping average
            model = train()
            model_score = validate()
            
            remove_test_files()
            # remove_files
