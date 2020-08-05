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

# 