import numpy as np
import matplotlib.pyplot as plt

X = 2*np.random.rand(100,1) # generate random data
y = 4 + 3 * X + np.random.rand(100,1) # generate random data

X_b = np.c_[np.ones((100,1)), X] # add xTHETA = 1 to each instance
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y) # calculate theta hat using normal equation

print(theta_best)


# make predictions using theta hat

X_new = np.array([[0],[2]])
X_new_b = np.c_[np.ones((2,1)), X_new]
y_predict = X_new_b.dot(theta_best)
print(y_predict)

# plot predictions
'''
plt.plot(X_new, y_predict, "r-")
plt.plot(X,y,"b.")
plt.axis([0,2,0,15])
plt.show()
'''
# linear regression
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression() # create model
lin_reg.fit(X,y) # fit model
print("lin_reg:")
print(lin_reg.intercept_)
print(lin_reg.coef_)
print("predict")
print(lin_reg.predict(X_new))

# least squares class
theta_best_svd, residuals, rank, s = np.linalg.lstsq(X_b,y,rcond=1e-6) # computes theta hat = (X^+)y --> X^+ is the pseudoinverse of X
print("theta best svd\n", theta_best_svd)

# compute using np.linalg.pinv() to gather psuedoinverse directly
print("np.linalg.pinv")
print(np.linalg.pinv(X_b).dot(y))

# Implementation of gradient descent
eta = 0.1
n_iterations = 1000
m = 100

theta = np.random.randn(2,1)

for iteration in range(n_iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - eta * gradients

print("theta:\n", theta)

# Stochastic gradient descent (e.g randomized gradient descent)
n_epochs = 50
t0,t1 = 5,50

def learning_schedule(t):
    return t0/(t+t1)

theta = np.random.randn(2,1)

for epoch in range(n_epochs):
    for i in range(m):
        random_index = np.random.randint(m)
        xi = X_b[random_index:random_index+1]
        yi = y[random_index:random_index+1]
        gradients = 2*xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule(epoch*m+i)
        theta = theta - eta*gradients

print("Stochastic theta:")
print(theta)