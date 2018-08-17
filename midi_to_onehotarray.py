import numpy as np
import math
import mido
import time as duration
from os import listdir
from os.path import isfile, join

save_to = 'midi_arrays/'
read_from = 'midis/'


def convert(filename):
    relevant_types = [
        'note_on',
        'note_off'
    ]
    print("\nReading File...")
    start = duration.time()
    midi_file = mido.MidiFile(read_from + filename)
    end = duration.time()
    print("Done. Duration:", end - start)
    print("\nConverting MIDI...")
    start = duration.time()
    timestamp = 0
    notes = []
    for msg in midi_file:
        timestamp += msg.time
        if msg.type in relevant_types:
            notes.append([msg.type, msg.note, timestamp])
    time = 0
    i = 0
    samples = []
    current_situation = np.zeros(128, dtype='int')
    for sample in np.arange(0, math.ceil(timestamp), 0.1):
        if i < len(notes) and notes[i][2] < sample:
            while time < sample:
                action, pitch, time = notes[i]
                if action == 'note_on':
                    current_situation[pitch] = 1
                elif action == 'note_off':
                    current_situation[pitch] = 0
                i = i + 1
                if not i < len(notes):
                    break
        # print("Sample",sample,"is",current_situation)
        samples.append(current_situation.copy())

    end = duration.time()
    data = np.array(samples)
    print("Done. Duration:", end - start)
    print("\nVerifying data:")
    if data.sum() > 0:
        print("Success!")
    else:
        print("Failed.")
        exit(-1)
    print("\nSaving File...")
    start = duration.time()
    np.savez_compressed(save_to + filename + '.npz', data)
    end = duration.time()
    print("Done. Duration:", end - start)


# read all files and convert them
for file_name in listdir(read_from):
    if isfile(join(read_from, file_name)):
        if not isfile(join(save_to, file_name + '.npz')):
            print("\n\n\n", file_name, ":")
            try:
                convert(file_name)
            except Exception:
                print("Error!")
                pass
