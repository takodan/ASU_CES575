# import keras
import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

# samples size in one iteration
batch_size = 128
# full training cycle times
epochs = 12
# MNIST has 10 digits
num_classes = 10

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# reshaping data format for TensorFlow
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# normalize pixel data
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# print to check the data shape and number
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

# build the model
model = Sequential()
# add a convolutional layer with filter number, kernel size, activation function, and input shape
model.add(Conv2D(2, kernel_size=(5, 5), activation='relu', input_shape=input_shape))
# add a pool layer, max pooling gives better result for MNIST
model.add(MaxPooling2D(pool_size=(2, 2)))
# add another convolutional layer
model.add(Conv2D(4, (5, 5), activation='relu'))
# add another pool layer
model.add(MaxPooling2D(pool_size=(2, 2)))
# flatten the feature maps into a one-dimensional array for the dense layers
model.add(Flatten())
# add fully connected layer (dense layer) with nodes number and activation function
model.add(Dense(120, activation='relu'))
# add another dense layer
model.add(Dense(84, activation='relu'))
# add another dense layer with 10 nodes for output
model.add(Dense(num_classes, activation='softmax'))

# https://keras.io/optimizers/
# loss for measure the error
# optimizer for update the weights
# metrics for choose the metrics to monitor
model.compile(loss= tf.keras.losses.categorical_crossentropy,
              optimizer= tf.keras.optimizers.Adadelta(learning_rate=0.1, rho=0.95),
              metrics=['accuracy'])

# training the model
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

# evaluate the result
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])