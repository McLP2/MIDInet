import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import LSTM, Dense, Dropout, InputLayer
import numpy as np
from os import listdir

# import midis

read_from = 'midi_arrays/'
look_back = 10

# TODO: figure out how this could work with multiple files
midis = np.ndarray((0, look_back, 128))
midis_y = np.ndarray((0, 128))
# go through folder and add all arrays
for file_name in listdir(read_from):
    arr = np.load(read_from + file_name)
    arr = arr[arr.keys()[0]]
    arr = np.reshape(arr[:look_back], (1, -1, 128))
    midis = np.concatenate((midis, arr), axis=0)
for file_name in listdir(read_from):
    arr = np.load(read_from + file_name)
    arr = arr[arr.keys()[0]]
    arr = np.reshape(arr[look_back], (1, 128))
    print(midis_y.shape, arr.shape)
    midis_y = np.concatenate((midis_y, arr))
midi, midi_y = 0, 0

# load just the first midi
for file_name in listdir(read_from)[:1]:
    arr = np.load(read_from + file_name)
    arr = arr[arr.keys()[0]]
    midi = (np.reshape(arr[:arr.shape[0] - 1], (-1, 128, 1)))
for file_name in listdir(read_from)[:1]:
    arr = np.load(read_from + file_name)
    arr = arr[arr.keys()[0]]
    midi_y = (np.reshape(arr[1:], (-1, 128)))

print(midi.shape, midi_y.shape)

model = keras.models.Sequential()

model.add(InputLayer(input_shape=(look_back, 128)))
model.add(LSTM(160, activation='tanh', dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(128, activation='sigmoid'))

model.compile(optimizer=keras.optimizers.Adam(lr=0.001),
              loss=keras.losses.mean_squared_error,
              metrics=[keras.metrics.mean_absolute_error])

model.fit(midis, midis_y, batch_size=256, epochs=50, shuffle=False, verbose=2)

np.set_printoptions(threshold=np.nan)
print(midis[1][:look_back])
print(model.predict(midis[1][:look_back].reshape(1, 10, 128)))

model.save('MIDInet.h5')
