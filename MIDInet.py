import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import LSTM, Dense, Dropout, TimeDistributed, Embedding, InputLayer, Flatten
import numpy as np
from os import listdir

# import midis

read_from = 'midi_arrays'

midis = []
midis_y = []
# go through folder and add all arrays
for file_name in listdir(read_from)[:10]:
    arr = np.load(read_from + file_name)
    arr = arr[arr.keys()[0]]
    midis.append(np.reshape(arr[:arr.shape[0] - 1], (-1, 128, 1)))
for file_name in listdir(read_from)[:10]:
    arr = np.load(read_from + file_name)
    arr = arr[arr.keys()[0]]
    midis_y.append(np.reshape(arr[1:], (-1, 128, 1)))

midi, midi_y = 0, 0

for file_name in listdir(read_from)[:1]:
    arr = np.load(read_from + file_name)
    arr = arr[arr.keys()[0]]
    midi = (np.reshape(arr[:arr.shape[0] - 1], (-1, 128, 1)))
for file_name in listdir(read_from)[:1]:
    arr = np.load(read_from + file_name)
    arr = arr[arr.keys()[0]]
    midi_y = (np.reshape(arr[1:], (-1, 128, 1)))

print(midi.shape, midi_y.shape)

model = keras.models.Sequential()

model.add(InputLayer(input_shape=(128, 1)))
model.add(LSTM(1024, activation='relu', dropout=0.2, recurrent_dropout=0.2))
model.add(TimeDistributed(Flatten()))
model.add(TimeDistributed(Dense(256, activation='relu')))
# model.add(TimeDistributed(Dropout(rate=0.2)))
model.add(TimeDistributed(Dense(128, activation='softmax')))

model.compile(optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0 - 9, beta_2=0.999, epsilon=None),
              loss=keras.losses.binary_crossentropy,
              metrics=['accuracy'])

model.fit(midi, midi_y, batch_size=32, epochs=10, shuffle=True)
