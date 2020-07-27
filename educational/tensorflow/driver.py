import tensorflow as tf

# Version
'''
print(tf.version)
'''

# Creating Tensors
'''
string = tf.Variable("this is a string", tf.string)
number = tf.Variable(333,tf.int16)
float = tf.Variable(333.33,tf.float64)
'''

# Tensor ranks
'''
Rank 0: 1 object. (i.g number = tf.Variable(333,tf.int16))
Rank 1: 1 list/array (i.g array = tf.Variable([333,356,222],tf.int16))
Rank 2: List/array within list/array (i.g array = tf.Variable([[333,356,222],[333,356,222]],tf.int16))
'''