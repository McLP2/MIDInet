import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import LSTM, Dense, Dropout, InputLayer, TimeDistributed
import numpy as np
from os import listdir

# import midis

read_from = 'midi_arrays/'
look_back = 20
step_size = 1


def make_dataset_from_midi(midi_array: np.array) -> (np.array, np.array):
    result_x = np.ndarray((0, look_back, 128))
    result_y = np.ndarray((0, look_back, 128))
    for i in range(0, midi_array.shape[0] - look_back - 1, step_size):
        x = midi_array[i:i + look_back]
        y = midi_array[i + 1:i + look_back + 1]
        result_x = np.concatenate((result_x, x.reshape(1, look_back, 128)))
        result_y = np.concatenate((result_y, y.reshape(1, look_back, 128)))
    return result_x, result_y


midis = np.ndarray((0, look_back, 128))
midis_y = np.ndarray((0, look_back, 128))
# go through folder and add all arrays
for file_name in listdir(read_from):
    arr = np.load(read_from + file_name)
    arr = arr[arr.keys()[0]]
    arr_x, arr_y = make_dataset_from_midi(arr)
    midis = np.concatenate((midis, arr_x))
    midis_y = np.concatenate((midis_y, arr_y))

model = keras.models.Sequential()

model.add(InputLayer(input_shape=(look_back, 128)))
model.add(LSTM(units=512, activation='tanh', dropout=0, recurrent_dropout=0, return_sequences=True))
model.add(LSTM(units=512, activation='tanh', dropout=0, recurrent_dropout=0, return_sequences=True))
model.add(LSTM(units=512, activation='tanh', dropout=0, recurrent_dropout=0, return_sequences=True))
model.add(TimeDistributed(Dense(units=256, activation='tanh')))
model.add(TimeDistributed(Dropout(rate=0)))
model.add(TimeDistributed(Dense(units=128, activation='sigmoid')))

model.compile(optimizer=keras.optimizers.Adam(lr=0.002),
              loss=keras.losses.mean_squared_error,
              metrics=[keras.metrics.mean_absolute_error])

model.fit(midis, midis_y, batch_size=512, epochs=50, shuffle=True, verbose=1)

# np.set_printoptions(threshold=np.nan)
print(midis[look_back][:look_back])
print(model.predict(midis[look_back][:look_back].reshape(1, look_back, 128)))

model.save('MIDInet.h5')
