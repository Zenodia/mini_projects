# -*- coding: utf-8 -*-
"""
Created on Tue May 16 10:38:36 2017

@author: zenodia.charpy
"""

from keras.layers import Input, Dense
from keras.models import Model
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
import pandas as pd
# this is the size of our encoded representations
# Part 1 - Building the CNN
filname='fer2013.csv'
label_map = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
def getData(filname):
    # images are 48x48 = 2304 size vectors
    # N = 35887
    Y = []
    X = []
    first = True
    for line in open(filname):
        if first:
            first = False
        else:
            row = line.split(',')
            Y.append(int(row[0]))
            X.append([int(p) for p in row[1].split()])

    X, Y = np.array(X) / 255.0, np.array(Y) # scaling is already done here 
    return X, Y

X, Y = getData(filname)
print(X.shape)
print(Y.shape)
print(len(set(Y)))
num_class=len(set(Y))
#check if classes are balance

#reshape X to fit keras with tensorflow backend
N,D=X.shape
X=X.reshape(N,48,48,1) # last dimension =1 is because it is black and white image, if colored, it will be 3
X=X.reshape(N,2304,1) # last dimension =1 is because it is black and white image, if colored, it will be 3


#split into training and testing set and rearrange the label y into 7 classes 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
y_train= (np.arange(num_class) == y_train[:,None]).astype(np.float32)
y_test=(np.arange(num_class) == y_test[:,None]).astype(np.float32)


from keras.layers import Input, Dense
from keras.models import Model

# this is the size of our encoded representations
encoding_dim = 48  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# this is our input placeholder
input_img = Input(shape=(2304,))
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_img)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(2304, activation='sigmoid')(encoded)
# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)
# this model maps an input to its encoded representation
encoder = Model(input_img, encoded)


# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))
# compile the model
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')


#manipulate the train and test set shape to fit the autoenconder
X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
X_test = X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))
# fit the model
autoencoder.fit(X_train, X_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(X_test, X_test))


# encode and decode some digits
# note that we take them from the *test* set
encoded_imgs = encoder.predict(X_test)
decoded_imgs = decoder.predict(encoded_imgs)

#plot the reconstructed image
import matplotlib.pyplot as plt

n = 7  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(X_test[i].reshape(48, 48))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(48, 48))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

#save the model weights
# save the model for later use
from keras.models import save_model, load_model
    
encoder.save('encoder_face.h5')  # creates a HDF5 file 
decoder.save('decoder_face.h5')
#del autoenconder  # deletes the existing model

