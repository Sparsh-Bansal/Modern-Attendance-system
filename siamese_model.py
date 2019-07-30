import tensorflow as tf
from tensorflow.python.keras.layers import Conv2D , MaxPool2D , Lambda , Dense , Input , Flatten
from tensorflow.python.keras.layers import BatchNormalization ,Dropout
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.optimizers import rmsprop ,adam
from tensorflow.python.keras.models import Model , load_model ,model_from_json
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.activations import relu , sigmoid ,tanh
import numpy as np
from tensorflow.python import keras
import os
import time
import cv2

def build_model_network():
    model = Sequential()

    model.add(Conv2D(input_shape=(128,128,3) , filters=64 ,kernel_size=(10,10), activation=relu))

    model.add(BatchNormalization())

    model.add(Dropout(0.1))

    model.add(MaxPool2D(pool_size=(2,2)))

    model.add(Conv2D(filters=128 , kernel_size=(7,7) , activation=relu))

    model.add(BatchNormalization())

    model.add(MaxPool2D(pool_size=(2,2)))

    model.add(Conv2D(filters=128 , kernel_size=(4,4) , activation=relu))

    model.add(BatchNormalization())

    model.add(MaxPool2D(pool_size=(2,2)))

    model.add(Conv2D(filters=256 , kernel_size=(4,4) , activation=relu))

    model.add(BatchNormalization())

    model.add(Dropout(rate=0.1))

    model.add(Flatten())

    model.add(Dense(units=4096 , activation=sigmoid))

    return model

model = build_model_network()
input_x1 = Input(shape=(128,128,3))
input_x2 = Input(shape=(128,128,3))

output_x1 = model(input_x1)
output_x2 = model(input_x2)

def euclidean_distance(vects):
    x, y = vects[0] , vects[1]
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))

def contrastive_loss(y_true, y_pred):
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

distance = Lambda(euclidean_distance)([output_x1 , output_x2])
# distance = Lambda( lambda tensors : K.abs( tensors[0] - tensors[1] ))( [output_x1 , output_x2] )
output_ = Dense(1,activation=sigmoid)(distance)

rms = rmsprop()

model = Model([input_x1 , input_x2] , output_)
model.compile(loss=contrastive_loss , optimizer = 'rmsprop')

X1 = np.load('numpy_files/X1.npy',allow_pickle=True)
X2 = np.load('numpy_files/X2.npy' , allow_pickle=True)
Y = np.load('numpy_files/Y.npy' , allow_pickle=True)
print(X1.shape)
print(X1[0])
# print(X1[0][0])
data_dimension = 128
X11 = X1.reshape( ( X1.shape[0]  , 128,128,3 ) ).astype( np.float32 )
X22 = X2.reshape( ( X2.shape[0]  , 128,128,3 ) ).astype( np.float32 )

model.fit([X11,X22] , Y , batch_size=5 , epochs=5 , validation_split=None )
# model.save('siamese.h5')
# model = load_model('siamese.h5')

model_json = model.to_json()
with open("model_num.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("model_num.h5")
# #
json_file = open('model_num.json', 'r')
# #
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# #
# # # load weights into new model
loaded_model.load_weights("model_num.h5")
print("Loaded model from disk")
# #
loaded_model.save('model_num.hdf5')
# loaded_model=load_model('model_num.hdf5')

# checkpoint = keras.callbacks.ModelCheckpoint("Checkpoint/weights.{epoch:02d}-{val_loss:.2f}.hdf5", monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
# model_json = model.to_json()
# with open("model.json", "w") as json_file:
#     json_file.write(model_json)
# # serialize weights to HDF5
# model.save_weights("model_weights.h5")
# print("Saved model to disk")





