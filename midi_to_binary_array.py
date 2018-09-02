import numpy as np
import math
import mido
import time as duration
from os import listdir
from os.path import isfile, join

# settings

save_to = 'midi_arrays/'  # output folder
read_from = 'midis/'  # input folder


def convert(filename):
    # only care about notes
    relevant_types = [
        'note_on',
        'note_off'
    ]
    print("\nReading File...")
    start = duration.time()  # duration calculation
    # read midi file
    midi_file = mido.MidiFile(read_from + filename)
    end = duration.time()  # duration calculation
    print("Done. Duration:", end - start)  # duration calculation
    print("\nConverting MIDI...")
    start = duration.time()  # duration calculation
    timestamp = 0
    # create list of notes
    notes = []

    for msg in midi_file:
        timestamp += msg.time  # msg.time is the delta
        if msg.type in relevant_types:
            # add relevant data to notes-array (on/off, pitch, timestamp)
            notes.append([msg.type, msg.note, timestamp])

    i = 0
    samples = []
    # begin with zeros (assume that there were no notes playing before the beginning of the midi)
    current_situation = np.zeros(128, dtype='int')
    # make samples of the whole midi fil with a time-delta-resolution of 0.1 seconds
    for sample in np.arange(0, math.ceil(timestamp), 0.1):
        # if non-processed notes are available to the current sample, process them
        if i < len(notes) and notes[i][2] <= sample:
            while notes[i][2] <= sample:
                action, pitch, time = notes[i]
                # print(notes[i])
                if action == 'note_on':
                    current_situation[pitch] = 1
                elif action == 'note_off':
                    current_situation[pitch] = 0
                i = i + 1
                if not i < len(notes):
                    break
        # after each time-step, the current sample is saved
        samples.append(current_situation.copy())

    end = duration.time()  # duration calculation
    data = np.array(samples)
    print("Done. Duration:", end - start)  # duration calculation
    print("\nVerifying data:")
    if data.sum() > 0:
        print("Success!")
    else:
        print("Failed.")
        exit(-1)
    print("\nSaving File...")
    start = duration.time()  # duration calculation
    np.savez_compressed(save_to + filename + '.npz', data)
    end = duration.time()  # duration calculation
    print("Done. Duration:", end - start)  # duration calculation


def main():
    # read all files and convert them
    for file_name in listdir(read_from):
        if isfile(join(read_from, file_name)):
            # but only, if they are not converted yet
            if not isfile(join(save_to, file_name + '.npz')):
                print("\n\n\n", file_name, ":")
                try:
                    convert(file_name)
                except Exception as ex:
                    # print all errors but do not care about them further
                    print("Error:", ex)
                    pass


if __name__ == "__main__":
    main()
