import tensorflow as tf

# Important resources
'''
https://www.tensorflow.org/
https://www.youtube.com/watch?v=tPYj3fFJGjk
'''

# Version
'''
print(tf.version)
'''

# Tensor types:
'''
Variable --> Mutable
Constant --> Immutable
Placeholder --> Immutable
Sparsetensor --> Immutable
'''

# Creating Variable Tensors
'''
string = tf.Variable("this is a string", tf.string)
number = tf.Variable(333,tf.int16)
float = tf.Variable(333.33,tf.float64)
'''

# Tensor ranks
'''
Rank 0: 1 object. (i.g: number = tf.Variable(333,tf.int16))
Rank 1: 1 list/array (i.g: array = tf.Variable([333,356,222],tf.int16))
Rank 2: List/array within list/array (i.g: array = tf.Variable([[333,356,222],[333,356,222]],tf.int16))
'''

# Test rank of Tensor
'''
tf.rank(number)
'''

# Shape of Tensor
'''
print(number.shape)
Describes number of elements within array/list of tensor. Also describes number of lists/arrays within tensor
i.g:
array = tf.Variable([[333,356,222],[333,356,222][22,56,12]],tf.int16) == [3,3]

first 3: number of interior lists
second 3: number of elements within the lists
'''
# Reshaping Tensors
'''
tensor1 = tf.ones([1,2,3]) <-- 'ones' creates a tensor full of ones in the described shape. in this case 1,2,3
i.g [[[1,1,1],[1,1,1]]]
One interior list containing our two lists of elements. Those lists contain three elements each. This explains 1,2,3
tensor2 = tf.reshape(tensor1,[2,3,1]) <-- reshapes data, this produces:
[
    [1]
    [1]
    [1]]
[
    [1]
    [1]
    [1]
]
i.g 2 lists containing 3 interior lists containing 1 element each
tensor3 = tf.reshape(tensor2,[3,-1]) <-- reshapes data in a unique way. the -1 infers what the next number should be. i.g:
[[1,1,1],[1,1,1]]

this is probably the most simple/helpful method in this circumstance

print(tensor1)
print(tensor2)
print(tensor3)
'''
# Evaluating Tensors
'''
Basic template for evaluation:
with tf.Session() as sess: <-- creates session using default graph
    tensor.eval() <-- tensor is the name of your tensor

tensorflow automatically evaluates tensor value
more information in documentation
'''

# Other functions:
'''
t = tf.zeros([5,5,5,5,5]) <-- Same as 'ones' but with zeros -- creates Tensor of zeros following specified shape pattern
'''

# Basic code in use
'''
t = tf.zeros([5,5,5,5])
print(t)
t=tf.reshape(t,[625])
print(t)
'''