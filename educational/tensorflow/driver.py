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
tensor3 = tf.reshape(tensor2,[3,-1]) <-- reshapes data, this produces:


print(tensor1)
print(tensor2)
print(tensor3)
'''