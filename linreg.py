import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd
import array
import matplotlib.pyplot as plt


def standard_normalizer(X):
    # Standardize the data
    scaler = StandardScaler()
    # Fit the scaler to the data and transform the data
    scaled_X = scaler.fit_transform(X)
    return scaled_X

def cost_function(X, y, theta, m_samples):
    # Calculate the cost function
    cost = np.sum((X.dot(theta) - y)**2)
    return (1/(2 * m_samples)) * cost

def gradient_descent(X, y, theta, alpha, num_iters):
    y_length = len(y)
    J_history = []
    for iteration in range(num_iters):
        # Calculate the cost function
        cost = cost_function(X, y, theta, y_length)
        J_history.append(cost)
        # Calculate the gradient. T is the transpose of the matrix. Which was why we were getting the error for mismatched dimensions.
        gradient = -2/y_length * X.T.dot(y-X.dot(theta))
        # Update the parameters
        theta = theta - alpha * gradient
        if iteration < 1000:
            print(f'Cost at iteration {iteration}: {cost}')
        

    return theta, J_history


#===========================================================
# Turn the CSV file into a pandas dataframe
df = pd.read_csv("d3.csv")
df.to_numpy()

# Create a numpy array of the x values
X = np.array(df[['x1', 'x2', 'x3']])
#How many rows are in the dataframe?
#print(len(X))

# Create a numpy array of the y values
y = np.array(df['y'])
#How many rows are in the dataframe?
#print(len(y))

# Standardize the data
scaled_X = standard_normalizer(X)
#print(scaled_X)

# Add a column of 1 to the x values for the bias intercept term
ones_column = np.ones(len(X), dtype=int).reshape(-1, 1)
scaled_X = np.append(scaled_X, ones_column, axis=1)
#print(scaled_X[1:5])

# Initialize theta (coefficients) to zeros.
theta = np.zeros(scaled_X.shape[1])

# Initialize Parameters
learning_rate = 0.01
num_iterations = 100000

# Run gradient descent
theta, J_history = gradient_descent(scaled_X, y, theta, learning_rate, num_iterations)

plt.plot(J_history)
plt.show()



