## Deep Learning application to identify modulators in the monkey brain network

This project aims to use Deep Learning tools on the monkey brain neural signal to identify neural modulators of brain networks. Given a causal brain network of sender and receivers electrode (see below for their definition), modulators are electrodes which act as orchestrators of the receiver response given a sender stimulus: modulators act as a gate which opens or closes the channel of communication between senders and receivers.

**Sender and Receivers**

Senders are defined as the electrodes which are stimulated with a single-pulse microstimulation. While a sender electrode is stimulated, we simultaneously record from ~100 electrodes placed in different brain regions. We stimulated one electrode at the time and recorded simultaneously from different sites.

The stimulation pulse-evoked responses were detected using an optimal signal detection algorithm, named
stimulation-based accumulating log-likelihood ratio (stimAccLLR; [1,2]). We tested 6,910 pairs of
electrodes, involving 132 brain region pairs (edges), among these, we detected and characterized 110 directed _sender-receiver communication_ involving 21 network edges.

![Screenshot from 2022-03-23 17-56-29](https://user-images.githubusercontent.com/45543977/159802711-1a570637-f60f-4301-ae22-5863ef781adb.png)



**Modulators**

When recording the receiver’s response to the sender’s stimulation for a given sender-receiver pair we observe that such response is characterized by “hit” and “miss” events, i.e. the receiver either does or does not respond to the sender input, respectively, trial by trial. We hypothesize that such variability in the sender-receiver interaction may be controlled or influenced by a network mechanism involving other brain sites (modulators) responsible for strengthening or weakening the sender-receiver excitability pulse-by-pulse.

To find such modulator sites, we use Machine and Deep Learning tools on the modulator baseline signal (prior to sender stimulation) to predict the receiver's response (i.e. hit or miss) from the modulator signal alone. From the Local Field Potential of the modulator baseline period (1 sec prior sender stimulation) we construct the spectrogram of the modulator using a multi-taper analysis. The spectrogram refers to the frequency-time activity of the mudulator 1 sec prior to the sender stimulation. This frequency-time signal is used as an input for the artificial neural network models used to predict the receiver's response.

**Artificial Neural Network models**

We use three different types of models to predict the receiver's response to the sender stimulus from the other electrodes' activity only. These are:

1. Logistic Regression
2. Multilayer Perceptron (MLP)
3. Convolutional neural network (CNN)

For MLP and CNN we test different architectures with different dropouts and batch normalization in order to reduce the overfitting.  

The problem is a binary classification supervised problem where the labels are the hits and misses of the receiver response. Electrodes which have predictive accuracy of the receiver's response over 70% are classified as modulators.


**Training/ testing**

The data set is split randomly into 70% training and 30% testing set. Each data set is made of the recorded electrodes spectrograms for each trial and the corresponding receiver response for that trial (hit/miss) which acts as a label.
