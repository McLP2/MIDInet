# MIDInet
Project to try out LSTM with tensorflow and keras


## Motivation
The main goal of MIDInet is to train an LSTM-Neural-Network to compose its own music.
I thought, this might be a fun and exciting way to find out, how LSTMs work and where
the difficulties are with Time-Series-Prediction.


## Concept
The neural model is trained with specially encoded Midi-files. 

### Midi
Midi is a message-oriented protocol for communication between digital instruments such
as keyboards. Amongst other things, these messages contain information about notes 
being played and released. Messages are provided with a delta-time corresponding to the
time which has passed since the last one.

### Encoding
The Midis are encoded as sampled binary arrays. This means, the files no longer consist
of messages, but of samples. Samples can be described as special photos of the virtual 
keyboard that are taken in equal intervals. These "photos" are much simpler than a real
picture. They just contain information about which keys are pressed down (1) an which are 
not (0). This saves a lot of space and should make it easier for the model to learn,
because all time intervals are equal and there are not many options.

### Model
I chose an LSTM-Neural-Network for this task, because this Machine-Learning-algorithm
is said to be the best for long and far-past-dependent sequences such as music. The
network currently consists of one hidden layer with 512 cells and dropouts of 20%. The
output Layer is also LSTM. Its reason is depending of the training concept I
describe in the next paragraph.

### Training
Adam (my optimizer of choice) computes the errors to train the model after every
time-step for best result, even though this uses a lot of training resources. The
alternative would have been, to just compute the error only after a full sequence-block
has been shown to the model which does not produce great results at all.

To be able to compute the gradients, the network needs the error which can not be
computed without the comparison to the expected value, because the training is a
supervised paradigm. So, how do we know what the network expects?
We first split our Midi-samples into blocks of 5 seconds. This means, the model can only
learn patterns that are shorter than this, but because the blocks overlap each other,
it should be able to combine them to greater. The expected output for the sample-block
has a very simple definition: It is a block of samples with the same length, but one
time-step/sample ahead.

### Generate
Once the model is trained, it can be used to compose music. This is done by feeding it
a predefined sequence of samples and asking it, what it predicts next. The prediction is
then processed to a binary array and appended to the sequence. This binary array is then
fed to the network again and again the output is added to the sequence. In the end, the
final sequence is then converted to Midi-messages and saved to the disk, ready to be
listened to - or to cover your ears.


## Results
The results are rather bad. There are midis produced, but they are horrible. It can't
be called music, it is not even real key-mashing. I think this error is produced because
most of the time a key is not being pressed, so the network just learns this. But
sometimes, this is not the case, but it plays one or two funky chords or clusters. So
maybe the mistake is in the interpretation of the neural network's output via a threshold
and not in the network itself.

Further research is needed.


## Ideas
Use one-hot-encoding for each note by introducing a new dimension instead of onedimensional binary encoding for the whole spectrum. This would make the threshold-problem
obsolete and therefore reduce the possibility of error.
