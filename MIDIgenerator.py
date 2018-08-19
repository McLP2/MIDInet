from tensorflow.keras import models
from tensorflow.keras.layers import LSTM, Dense, Dropout, InputLayer
import onehotarray_to_midi
import numpy as np

# not working
# model = models.load_model('MIDInet.h5')

# may work
model = models.Sequential()

model.add(InputLayer(input_shape=(128, 1)))
model.add(LSTM(160, activation='relu', dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(128, activation='sigmoid'))

model.load_weights('MIDInet.h5')

sequence = np.array([
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
])

output_length = 100  # in deciseconds

# generate & export data
output = sequence
for i in range(output_length):
    output = model.predict(output.reshape((1, 128, 1)), verbose=1)[0]
    mask = output > 0.5
    output = np.zeros(128)
    output[mask] = 1
    print(output)
    sequence = np.concatenate((sequence, output.reshape((1, 128))))
onehotarray_to_midi.convert(sequence, 'prediction.mid')
