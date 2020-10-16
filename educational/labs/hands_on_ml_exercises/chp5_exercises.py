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
theta_best_svd, residuals, rank, s = np.linalg.lstsq(X_b,y,rcond=1e-6)
print("theta best svd\n", theta_best_svd)