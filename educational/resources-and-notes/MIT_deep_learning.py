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

# 
'''

'''