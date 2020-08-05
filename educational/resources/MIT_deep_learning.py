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