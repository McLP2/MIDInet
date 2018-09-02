
print("Initializing...")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import LSTM, Dense, Dropout, InputLayer, TimeDistributed
import numpy as np
from os import listdir
from os.path import isfile
import MIDIgenerator
import generator_callback

# import midis

read_from = 'midi_arrays/'
look_back = 50
step_size = look_back // 3 * 2
batch_size = 100
epochs = 256
approximate_file_count = 100


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

print("\nLoading training data...")

if not isfile("prepared_training_data_" + str(approximate_file_count) + ".npz"):
    # go through folder and add all arrays
    files = listdir(read_from)
    mask = np.random.random((len(files),)) < approximate_file_count / len(files)
    files = [files[i] for i in range(len(files)) if mask[i]]
    i = 0
    for file_name in files:
        i = i + 1
        print("  Loading: " + file_name + " (" + str(i) + "/" + str(len(files)) + ")")
        arr = np.load(read_from + file_name)
        arr = arr[arr.keys()[0]]
        arr_x, arr_y = make_dataset_from_midi(arr)
        midis = np.concatenate((midis, arr_x))
        midis_y = np.concatenate((midis_y, arr_y))
    np.savez_compressed("prepared_training_data_" + str(approximate_file_count) + ".npz", x=midis, y=midis_y)
else:
    # load prepared file
    loaded = np.load("prepared_training_data_" + str(approximate_file_count) + ".npz")
    midis = loaded['x']
    midis_y = loaded['y']

print("\nBuilding model...")

model = keras.models.Sequential()

model.add(InputLayer(input_shape=(None, 128)))
model.add(LSTM(units=512, activation='tanh', dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
model.add(LSTM(units=128, activation='sigmoid', dropout=0, recurrent_dropout=0.2, return_sequences=True))

model.compile(optimizer=keras.optimizers.Adam(lr=0.001),
              loss=keras.losses.mean_squared_error,
              metrics=[keras.metrics.mean_absolute_error])

print("\nStarting training process...")

checkpoint = keras.callbacks.ModelCheckpoint("checkpoints/MIDInet_epoch{epoch:03d}.hdf5")
generator = generator_callback.GeneratorCallback("checkpoints/prediction_at_epoch_{epoch:03d}", 100)

model.fit(midis, midis_y,
          batch_size=batch_size,
          epochs=epochs,
          shuffle=True,
          verbose=1,
          callbacks=[checkpoint, generator]
          )

print("\nSaving model...")

model.save('MIDInet.hdf5')

print("\nRunning final prediction:\n\n")

MIDIgenerator.generate(model, output_length=600)
