import tensorflow as tf
import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib
import tensorflow.compat.v2.feature_column as fc

# Important resources
'''
https://www.tensorflow.org/
https://www.youtube.com/watch?v=tPYj3fFJGjk
https://colab.research.google.com/drive/1F_EWVKa8rbMXi3_fG0w7AtcscFq7Hi7B#forceEdit=true&sandboxMode=true&scrollTo=duDj86TfWFof
'''

# Version
'''
print(tf.version)
print(sklearn.__version__)
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

# Machine learning algorithms using Tensorflow
'''
Linear Regression:
y=mx+b
Using line of best fit to predict future x,y,z,etc values

'''

# Misc notes
'''
.csv file extension == comma seperated values
'''

# Pandas
'''
head() method print first 5 columns of dataset
describe() gives statistical analysis of dataset (mean, std, min, max, etc)
linear_est.predict() gives dictionary data that represents predictions for all data points. 'probabilities' gives value of
percent chance of y/n on given data point
'''

# Numerical vs Categorical data
'''
Categorical == non numeric. i.g different categories such as male/female
Numerical == Always represented in numbers

All categorical data must be eventually changed into numeric data. So M/F could be turned into 0 or 1
'''


# ***************************************
# Lines below indicate working code
# ***************************************



# Load dataset
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv') # Training data
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv') # Testing data
# Removing result data from metric data
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

# Graphs
'''
dftrain.age.hist(bins=20) # Histogram of age values
dftrain.sex.value_counts().plot(kind='barh') # Bar graph of sex values
dftrain['class'].value_counts().plot(kind='barh') # Bar graph of class values
pd.concat([dftrain,y_train], axis=1).groupby('sex').survived.mean().plot(kind='barh').set_xlabel('% survive') # Bar graph of survival
# ... rate by gender
plt.show(block=False)
plt.pause(10)
plt.close()
'''

CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck',
                       'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']

feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
    # gets a list of all unique values from given feature column
  vocabulary = dftrain[feature_name].unique()
  # Fills feature_columns list with categorical column values represented via numeric data
  feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))
# Adds original numeric data to feature_columns list
for feature_name in NUMERIC_COLUMNS:
  feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

print(feature_columns)

def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
  def input_function():  # inner function, this will be returned
    ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))  # create tf.data.Dataset object with data and its label
    if shuffle:
      ds = ds.shuffle(1000)  # randomize order of data
    ds = ds.batch(batch_size).repeat(num_epochs)  # split dataset into batches of 32 and repeat process for number of epochs
    return ds  # return a batch of the dataset
  return input_function  # return a function object for use

# Create objects of make_input_fn for training
train_input_fn = make_input_fn(dftrain, y_train)  # here we will call the input_function that was returned to us to get a dataset object we can feed to the model
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)

# Creating the model
linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)

# Training the model
linear_est.train(train_input_fn)  # train
result = linear_est.evaluate(eval_input_fn)  # get model metrics/stats by testing on testing data

clear_output()  # clears consoke output
print(result['accuracy'])  # the result variable is simply a dict of stats about our model