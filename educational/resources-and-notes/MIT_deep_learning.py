# MIT Introduction to Deep Learning Course Notes

# Perceptron / Neuron
'''
How does it work?
Take your inputs, apply dot product with weights, add a bias, and apply a non-linearity
'''

# Working with weights
'''
Explanation around 33:00 

Essentially weights are randomized and slowly creep in the right direction until finding the optimal weight that
creates optimal output


Code:

import tensorflow as tf

weights = tf.Variable([tf.random.normal()])

while True:
    with tf.GradientTape() as g:
        loss = compute_loss(weights)
        gradient = g.gradient(loss,weights)
    
    weights = weights - lr * gradient

'''

# Setting the learning rate
'''
Too big: constantly overshooting
Too small: being stuck in current position

How to set a learning rate that is 'just right'?
'Adaptive learning rate':
Non-fixed learning rate. Made smaller or larger based on:
1. how large gradient is
2. how fast learning is happening
3. size of particular weights
4. etc.

Implementation:
SGD: tf.keras.optimizers.SGD() -- (NON-ADAPTIVE)
Adam: tf.keras.optimizers.Adam() -- (ADAPTIVE)
Adadelta: tf.keras.optimizers.Adadelta() -- (ADAPTIVE)
Adagrad: tf.keras.optimizers.Adagrad() -- (ADAPTIVE)
RMSProp: tf.keras.optimizers.RMSProp() -- (ADAPTIVE)

'''

# 'Putting it all together'
'''
import tensorflow as tf

model = tf.Keras.Sequential([...])

# Optimizer
optimizer = tf.keras.optimizer.SGD()

while True:

    # Forward pass through network
    prediction = model(x)

    with tf.GradientTape() as tape:
        # Compute loss
        loss = compute_loss(y, prediction)

    # Update weights using gradient
    grads = tape.gradient(loss,model.trainable_variables)
    optimizer.apply_gradients(zip(grads,model.trainable_variables))
    '''

# How to avoid 'memorization'
'''
"Regularization" techniques are used to avoid model memorization of training data. This memorization creates an
'over-fitting' of the data. (seen in the graph around 48:20)

Regularization I: During training, randomly set some activations to 0.
Typically drop 50% of activations in layer
Forces network to not rely on any 1 node.
Works best with multiple iterations (one round of randomly selected activations, another round of another set, etc)
Implementation:
tf.keras.layers.Dropout(p=0.5)

Regularization 2: Early Stopping
Stop training before the model has a chance to overfit (memorize data)
e.g at a certain point the training data will start to outperform the testing data
Helpful graph in the video: 51:41



'''

# Lesson 1 Summary
'''
Core foundation:
Structural building blocks
nonlinear activation functions
stacking perceptrons to form neural networks
Optimization through backpropagation
adaptive learning
batching
regularization
'''

# Sequence modeling
'''
NLP: 

One of the problems with NLP is that the input can have varying length when trying to predict future words.
e.g i took my cat for a walk (7 words)
vs i like calculus (3 words)

'Ideas' to solve this problem:
1. Use a fixed window. e.g only look at a certain number of words prior to the word we are predicting. (2, 3, etc)
However, because we are using this fixed window it is very limiting to our prediction. Some data is only available
earlier in sentences. This is not a good option.
2. Bag of words: each slot in input vector represents a word and the value in the slot represents
the number of times a word appears in the sentence. (6:40 video #2)
However, their is another problem with this. The count does not necesarily denote a proper meaning for the sentence.
3. extend fixed window, look at more words. however, this causes problems with the networks learning. 

To model sequences we need to:
1. develop models that handle variable length, track long term dependencies in data, maintain information about sequences order,
share parameters across the entirety of the sequence.

Solution: Recurrent neural network. (RNN)

'''

# Standard Feed-Forward NN
'''
Cannot handle sequential data. as explained in prior notes about why we must use recurrent neural networks instead.
'''

# Recurrent Neural Network
'''
Ideal for sequential data. 

Sequential data: text, audio, etc.

How is the data looped?
Apply a recurrence relation at every time step to process a sequence: h<t>=f<w>(h<t-1>,x<t>) <-- "<>" for subscript

h<t> == cell state

f<w> == function paramaterized by W (weights) <-- same set of weights and parameters are used in every step of the process

h<t-1> == old state

x<t> == input vector at time step t

Example psuedo code:

my_rnn = RNN()

hidden_state = [0,0,0,0]

sentence = ['I','love','recurrent','neural']

for word in sentence:
    prediction, hidden_state = my_rnn(word,hidden_state)

next_word_prediction = prediction


Back to notes:
Loss:
loss may be computed at every iteration, which will then be summed up into a total loss.


'''

# Implementing RNN from scratch in Tensorflow
'''
class MyRNNCell(tf.keras.layers.Layer):
    def __init__(self,rnn_units,input_dim,_output_dim):
        super(MyRNNCell,self).__init__()
    
    # Initialize weight matrices (Graph at video 2 17:59)
    self.W_xh = self.add_weight([rnn_units,input_dim])
    self.W_hh = self.add_weight([rnn_units,rnn_units])
    self.W_hy = self.add_weight([_output_dim,rnn_units])

    # Initialize hidden state to zeros
    self.h = tf.zeros([rnn_units, 1])

    def call(self,x):
        # Update the hidden state
        self.h = tf.math.tanh(self.W_hh*self.h+self.W_xh*x)

        # Compute the output
        # transformed version of hidden state -- at each time step we return current output and updated hidden state
        output = self.W_hy * self.h

        # Return the current output and hidden state
        return output,self.h
'''

# Intuition turning into implementation
'''
14:00 - 20:00 area on video 2 goes through this.

Graph at 17:56 (and information directly before it) describe weight matrices

h of t == internal state

W xh == transform input to hidden state (rnn_units == hidden state/h, input_dim == input/x)

W hh == transform previous hidden state to current hidden state (rnn_units == hidden state/h)

W hy == hidden state output (rnn_units == hidden state/h, y == output_dim)


'''

# More implementation
'''
tf.Keras.layers.SimpleRNN(rnn_units)
^^
implemented these type of RNN layers for us (code in 2 prior note segments ago), and it is called SimpleRNN (see above)

tf.Keras.layers.LSTM(num_units)
^^
LSTM implementation. -- LSTM cells are able to track information throughout many timesteps
'''

# Long Short Term Memory (LSTM) Networks
'''
Well suited to learning long-term dependencies to overcome vanishing gradient problem. (e.g long sentences that require
knowledge of multiple sections of the sentence)

Information is added or removed through structures called gates. (video #2 31:30) Gates optionally let information
through. For example via a sigmoid neural net later and pointwise multiplication (video #2 31:43 graph)


'''