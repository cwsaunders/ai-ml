import tensorflow as tf

# Hidden Markov Models
'''
Hidden Markov Models
"The Hidden Markov Model is a finite set of states, each of which is associated with a (generally multidimensional)
probability distribution []. Transitions among the states are governed by a set of probabilities called transition probabilities."
(http://jedlik.phy.bme.hu/~gerjanos/HMM/node4.html)
A hidden markov model works with probabilities to predict future events or states.
In this section we will learn how to create a hidden markov model that can predict the weather.
https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/HiddenMarkovModel
'''

# States
'''
Warm vs cold, high vs low, etc.
'''

# Observation
'''
Each state has a particular outcome or observation associated with it based on a probability distribution. i.g on a hot day Tim has
a 60% chance of being happy and a 40% chance of being sad.
'''

# Transition
'''
Each state has a probability that defining the likelihood of transitioning to a different state. i.g If you have three hot days in a row
their is a 57% chance of a fourth hot day and a 43% chance of a cold day.
'''