from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D
from sklearn.metrics import classification_report,confusion_matrix
from keras.callbacks import EarlyStopping, TensorBoard
from keras.models import Sequential, load_model
from keras import optimizers, regularizers
from keras.utils import np_utils

import data_processing as data 
import numpy as np
from time import time
import os

np.random.seed(123) # for reproducibility
    
# load data
dataset = data.getData() #instance 
X, Y, p = dataset.get()

(X_train, y_train), (X_test, y_test) = (dataset.X_train, dataset.y_train), (dataset.X_test, dataset.y_test)

# preprocess data
X_train = X_train.reshape(X_train.shape[0], 1, 430, 1)
X_test = X_test.reshape(X_test.shape[0], 1, 430, 1)
print(X_train.shape)
print(X_test.shape)

# normalize data values to range [0, 1]
X_train /= 255
X_test /= 255

# convert flat array to [Person1 .. Person90] one-hot coded array
y_train = y_train - 1
y_test = y_test - 1
y_train = np_utils.to_categorical(y_train, 90)
y_test = np_utils.to_categorical(y_test, 90)

#model architecture
model = Sequential()
model.add(Convolution2D(32, 1, 5, activation='tanh', input_shape=(1,430,1), kernel_regularizer=regularizers.l2(0.001)))
model.add(MaxPooling2D(pool_size=(1,3)))

model.add(Convolution2D(64, 1, 5, activation='tanh', kernel_regularizer=regularizers.l2(0.001)))
model.add(MaxPooling2D(pool_size=(1,3)))

model.add(Flatten())
model.add(Dense(128, activation='tanh'))
model.add(Dense(90, activation='softmax'))

# compile model
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# fit model on training data
tensorboard = TensorBoard(log_dir="logs_personid/{}".format(time()))
earlystopping = EarlyStopping(monitor='val_loss', patience=10)
history = model.fit(X_train, y_train, batch_size=10, validation_data=(X_test, y_test), epochs=10, verbose=1, callbacks = [earlystopping, tensorboard])

# evaluate model on test data
print("Evaluating model")
score = model.evaluate(X_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# save model to dir
model.save_weights(os.path.join('saved_models', 'rsampled_.h5'))

# plot performance graph
plot_fn = data.plotHelper()
plot_fn.plot_keys(history)

# confusion matrix
y_pred = model.predict(X_test)
classes_x=np.argmax(y_pred,axis=1)


# p=model.predict_proba(X_test) # to predict probability

inst = data.Setup()
feats, personid, info = inst.get_data()
names, age, gender = inst.dissect_labels(info)
target_names = np.asarray(names)
# print(target_names)

# Y = np.argmax(y_test,axis=1)
# print(Y)
# print(classification_report(np.argmax(y_test,axis=1), y_pred,target_names=target_names))
# cm = confusion_matrix(Y, y_pred)
# plot_fn.plot_confusion_matrix(cm, classes=['m', 'f'], title='Confusion matrix')

# load model from dir
model.save(os.path.join('saved_models', 'person_model.h5'))
model = model.load_weights(os.path.join('saved_models', 'rsampled_.h5'))