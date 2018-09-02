from tensorflow import keras
import MIDIgenerator


class GeneratorCallback(keras.callbacks.Callback):
    def __init__(self, filename, output_length):
        super().__init__()
        # save parameters
        self.filename = filename
        self.output_length = output_length

    def on_epoch_end(self, epoch, logs=None):
        # call generator
        MIDIgenerator.generate(self.model,
                               self.filename.format(epoch=epoch + 1),
                               verbose=0,
                               output_length=self.output_length)
