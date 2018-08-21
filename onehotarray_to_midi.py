import mido
import numpy as np


def convert(onehotarray_sequence, filename):
    midi_file = mido.MidiFile()
    track = mido.MidiTrack()
    midi_file.tracks.append(track)

    last_array = np.zeros(128)
    timer = 0

    for onehotarray in onehotarray_sequence:
        # get changed indices via xor
        indices = np.arange(0, 128, 1)[np.logical_xor(last_array, onehotarray)]

        for index in indices:
            # if new value == 0: add note-off and reset timer
            if onehotarray[index] == 0:
                track.append(mido.Message('note_off', note=index, velocity=127, time=timer))
                timer = 0
            # if new value == 1: add note-on and reset timer
            if onehotarray[index] == 1:
                track.append(mido.Message('note_on', note=index, velocity=127, time=timer))
                timer = 0

        # increase timer by array sample-length in midi ticks (default: 960 (tps) * 0.1 (s))
        timer = timer + 96

        last_array = onehotarray

    for index in np.arange(0, 128, 1)[last_array == 1]:
        track.append(mido.Message('note_off', note=index, velocity=127, time=timer))
    
    midi_file.save(filename)
