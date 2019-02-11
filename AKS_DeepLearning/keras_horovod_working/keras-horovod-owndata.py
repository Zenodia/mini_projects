from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

from keras import backend as K
import math
import tensorflow as tf
import horovod.keras as hvd
from keras.models import load_model

# Horovod: initialize Horovod.
hvd.init()

# Horovod: pin GPU to be used to process local rank (one GPU per process)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = str(hvd.local_rank())
K.set_session(tf.Session(config=config))

import numpy as np

### load dataset 

X=np.load('/mnt/azure/pneumonia/X.npy')
y=np.load('/mnt/azure/pneumonia/Y.npy')

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1234)

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, 3, 3, border_mode='same',input_shape = (150, 150, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(output_dim = 128, activation = 'relu')) # the output_dim is chosen by experience
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

# Compiling the CNN
#classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images
#classifier.fit(X_train,y_train, epochs=1, batch_size=20)

# Horovod: adjust learning rate based on number of GPUs.
opt = keras.optimizers.adam()

# Horovod: add Horovod Distributed Optimizer.
opt = hvd.DistributedOptimizer(opt)

classifier.compile(loss=keras.losses.binary_crossentropy,
              optimizer=opt,
              metrics=['accuracy'])

callbacks = [
    # Horovod: broadcast initial variable states from rank 0 to all other processes.
    # This is necessary to ensure consistent initialization of all workers when
    # training is started with random weights or restored from a checkpoint.
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),
]

# Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
if hvd.rank() == 0:
    callbacks.append(keras.callbacks.ModelCheckpoint('/mnt/azure/pneumonia/k8-horovod-{epoch}.h5'))

classifier.fit(x_train, y_train,
          batch_size=20,
          callbacks=callbacks,
          epochs=1,
          verbose=1,
          validation_data=(x_test, y_test))
score = classifier.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
classifier.save('/mnt/azure/pneumonia/k8_pneumonia_horovod_model.h5') 