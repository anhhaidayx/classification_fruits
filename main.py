import numpy as np
import tensorflow

from sklearn.datasets import load_files
from tensorflow.python.keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras_preprocessing.image import array_to_img, img_to_array,load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D
from tensorflow.keras.layers import Activation, Dense, Flatten, Dropout
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD, Adam, RMSprop

# run_opts = tensorflow.RunOptions(report_tensor_allocations_upon_oom = True)
# Loading data and putting them into training and test sets
#locations setting for training and test datasets
train_data='D:\\Tài liệu học\\Tri tuệ nhân tạo\\Fruit\\train1'
test_data='D:\\Tài liệu học\\Tri tuệ nhân tạo\\Fruit\\test1'
def get_data(path):
    data = load_files(path)
    files = np.array(data['filenames'])
    targets = np.array(data['target'])
    target_labels = np.array(data['target_names'])
    return files,targets,target_labels
X_train, Y_train, labels = get_data(train_data)
X_test, Y_test,_ = get_data(test_data)
Y_train = np_utils.to_categorical(Y_train, 6)
Y_test = np_utils.to_categorical(Y_test, 6)
#nomalizing the pixel values before feeding into a neural network
X_train, X_val = train_test_split(X_train, test_size=0.2,
random_state=33)
Y_train, Y_val = train_test_split(Y_train, test_size=0.2,
random_state=33)
#converting images into array to start computation
def convert_image_to_array(files):
    images_as_array=[]
    for file in files:
        images_as_array.append(img_to_array(load_img(file)))
    return images_as_array
X_train = np.array(convert_image_to_array(X_train))
X_val = np.array(convert_image_to_array(X_val))
X_test = np.array(convert_image_to_array(X_test))
#nomalizing the pixel values before feeding into a neural network
X_train = X_train.astype('float32')/255
X_val = X_val.astype('float32')/255
X_test = X_test.astype('float32')/255

# Building model 1 using customized convolutional and pooling layers
# model = Sequential()
# #input_shape is 100*100 since thats the dimension of each of the fruit images
# model.add(Conv2D(filters = 64, kernel_size = 2,input_shape=(224,224,3),padding='same'))
# model.add(Activation('relu'))
# model.add(tensorflow.keras.layers.BatchNormalization())
# model.add(MaxPooling2D(pool_size=2))
# model.add(Conv2D(filters = 128,kernel_size = 2,activation='relu',padding='same'))
# model.add(tensorflow.keras.layers.BatchNormalization())
# model.add(MaxPooling2D(pool_size=2))
# model.add(Conv2D(filters = 256,kernel_size = 2,activation='relu',padding='same'))
# model.add(tensorflow.keras.layers.BatchNormalization())
# model.add(MaxPooling2D(pool_size=2))
# model.add(Conv2D(filters = 512,kernel_size = 2,activation='relu',padding='same'))
# model.add(tensorflow.keras.layers.BatchNormalization())
# model.add(MaxPooling2D(pool_size=2))
# # specifying parameters for fully connected layer
# model.add(Flatten())
# model.add(Dense(256))
# model.add(Activation('relu'))
# model.add(Dropout(0.3))
# model.add(Dense(128))
# model.add(Activation('relu'))
# model.add(Dropout(0.3))
# model.add(Dense(6,activation = 'softmax'))
# model.summary()
# #importing ootimizers
# from tensorflow.keras.optimizers import SGD, Adam, RMSprop
# optimizer = Adam()
# model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])
# # creating a file to save the trained CNN model
# checkpointer = ModelCheckpoint(filepath =
# 'model11.hdf5', verbose = 1, save_best_only = True)
# # fitting model using above defined layers
# CNN_model = model.fit(X_train,Y_train,batch_size = 10,epochs=50,validation_data=(X_val, Y_val),callbacks = [checkpointer],verbose=2, shuffle=True)
# plt.figure(1, figsize = (10, 10))

from tensorflow.keras.applications.vgg16 import VGG16
vgg_model = VGG16(input_shape=[224,224,3], weights='imagenet',
include_top=False)
#We will not train the layers imported.
for layer in vgg_model.layers:
    layer.trainable = False
transfer_learning_model = Sequential()
transfer_learning_model.add(vgg_model)
# transfer_learning_model.add(Conv2D(1024, kernel_size=3,
# padding='same'))
# transfer_learning_model.add(Activation('relu'))
# transfer_learning_model.add(tensorflow.keras.layers.BatchNormalization())
# transfer_learning_model.add(MaxPooling2D(pool_size=(2, 2)))
# transfer_learning_model.add(Dropout(0.4))
transfer_learning_model.add(Flatten())
transfer_learning_model.add(Dense(4096))
transfer_learning_model.add(Activation('relu'))
transfer_learning_model.add(Dropout(0.4))
transfer_learning_model.add(Dense(4096))
transfer_learning_model.add(Activation('relu'))
transfer_learning_model.add(Dropout(0.4))
transfer_learning_model.add(Dense(6,activation = 'softmax'))
transfer_learning_model.summary()
optimizer = Adam()
transfer_learning_model.compile(loss='categorical_crossentropy',
optimizer=optimizer,
metrics=['accuracy'])
#fitting the new model
checkpointer = ModelCheckpoint(filepath = 'model11.hdf5',verbose = 1, save_best_only = True)
# running
CNN_model = transfer_learning_model.fit(X_train,Y_train,batch_size = 16,epochs=50,validation_data=(X_val, Y_val),callbacks = [checkpointer],
verbose=2, shuffle=True)


