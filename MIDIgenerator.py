import tensorflow as tf
from tensorflow import keras
import numpy as np
import mido

model = keras.models.load_model('MIDInet.h5')

initial_sequence = [
    [0, 0, 0, 0]
]

output_length = 200  # in deciseconds

# TODO: generate data
