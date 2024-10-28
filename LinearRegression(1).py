import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
np.random.seed(0)
num_samples = 100
num_features = 2

x = np.arange(-2,2,0.1)
y = np.arange(-2,2,0.1)
X,Y = np.meshgrid(x,y)
coefficients = np.random.randn(num_features)
noise = 0.1 * np.random.randn(num_samples)
Z = coefficients[0]*X + coefficients[1]*Y
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(X, Y, Z)
# Generate random data
X = np.random.randn(num_samples, num_features)
y = np.dot(X, coefficients) + noise
ax.scatter(X[:,0], X[:,1], y,c='r')
plt.show()

# Create and fit linear regression model
class linearRegression:
 def __init__(self,X,y,learning_rate=0.01,iterations=100):
      self.learning_rate=learning_rate
      self.iterations=iterations
      self.X=X
      self.y=y
      self.coef_=np.zeros(num_features)
      self.intercept_=0

      for i in range(self.iterations):
          yp=np.dot(self.X,self.coef_)+self.intercept_
          self.coef_=self.coef_+self.learning_rate*(2/num_samples)*np.dot(self.X.T,(self.y-yp))
          self.intercept_=self.intercept_+self.learning_rate*(2/num_samples)*np.sum(self.y-yp)
 def predict(self,X):
     return np.dot(X, self.coef_) + self.intercept_

model=linearRegression(X,y,0.01,100)
prediction=model.predict(X)


# Predict a new data point
new_data_point = np.random.randn(1, num_features)
prediction = model.predict(new_data_point)
print("Real_Coefficients:", coefficients)
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("Predicted value for new data point:", prediction)