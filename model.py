from keras.models import Sequential
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
from keras.models import model_from_json
import matplotlib.pyplot as plt
import csv

training_file = 'train.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)

X_train, Y_train = train['image_set'], train['angles_set']

#print (Y_train[0])
#plt.contourf(X_train[0,:,:,0],255, cmap='gray')
#plt.show()

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.01, random_state=42)

model = Sequential()
model.add(Lambda(lambda x: x/127.5 - 1.,
            input_shape=(40, 80, 3)))
model.add(Convolution2D(16, 8, 8, border_mode='same', subsample=(1, 1)))
model.add(ELU())
model.add(Convolution2D(32, 5, 5, border_mode='same', subsample=(2, 2)))
model.add(ELU())
model.add(Convolution2D(64, 5, 5, border_mode='same', subsample=(2, 2)))
model.add(Flatten())
model.add(Dropout(.2))
model.add(ELU())
model.add(Dense(512))
model.add(Dropout(.5))
model.add(ELU())
model.add(Dense(1))

model.summary()

# TODO: Compile and train the model.
batch_size = 128

load_model = True
if load_model:
	json_file = open('model.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	model = model_from_json(loaded_model_json)
	model.load_weights("model.h5")
	print("Loaded model from disk")

model.compile(optimizer="adam", loss="mse")

history = model.fit(X_train, Y_train,
                    batch_size=batch_size, nb_epoch=5,
                    verbose=1, validation_data=(X_val, Y_val), shuffle=True)

Y_train_pred = np.transpose(model.predict(X_train))[0]
print (np.amin(Y_train_pred ))
print (np.amax(Y_train_pred ))
print (np.mean((Y_train_pred -Y_train)**2))

#myfile = open('Y_train_pred.csv','w')
#wr = csv.writer(myfile)
#wr.writerow(Y_train_pred)
#
#myfile = open('Y_train_act.csv','w')
#wr = csv.writer(myfile)
#wr.writerow(Y_train)

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")