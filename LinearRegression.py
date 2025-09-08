from uuid import SafeUUID
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd
import array
import matplotlib.pyplot as plt
import sys


def standard_normalizer(X):
    # Standardize the data
    scaler = StandardScaler()
    # Fit the scaler to the data and transform the data
    scaled_X = scaler.fit_transform(X)
    return scaled_X

#Show the cost function derivation
# 

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
        if iteration % 100 == 0:
            print(f'Cost at iteration {iteration}: {cost}')
        # if the cost difference is less than 0.0003
        if abs(J_history[-1] - cost) < 10 and iteration > 1000:
            break
        #print(theta)

    return theta, J_history

def predict(theta, points):
    # Predict the y values for the new points array
    new_X = np.array([1.0, points[0], points[1], points[2]])
    return float(np.dot(theta, new_X))
   




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
scaled_X = np.append(ones_column ,scaled_X, axis=1)
#print(scaled_X[1:5])

# Initialize theta (coefficients) to zeros.
theta = np.zeros(scaled_X.shape[1])

# Initialize Parameters
learning_rate = 0.01
num_iterations = 10000

# Run gradient descent
theta, J_history = gradient_descent(scaled_X, y, theta, learning_rate, num_iterations)

#plt.plot(J_history)
#plt.show()

#Predict the y values with the new inputs
new_points = [[1,1,1], [2,0,4], [3,2,1]]


predictions = []
for i in new_points:
    prediction = predict(theta, i)
    predictions.append(prediction)

print("Prediction: ",predictions)
print("Theta", theta)



#====================================================================
# Question 2:
# Let the first three columns of the data set be separate explanatory variables 𝑥1 , 𝑥2 , 𝑥3 , and the
# fourth column be the dependent variable y. Compute the closed-form solution (or analytical
# solution) for the hypothesis function parameters: 𝛉 = [𝜃0 , 𝜃1 , 𝜃2 , 𝜃3 ]. Show each step. Use the
# normal equation to obtain 𝛉. You can use Python to compute the matrix inverse in deriving your
# solution.

df = pd.read_csv("question2.csv")

# Create a numpy array of the x values
X = np.array(df[['x1', 'x2', 'x3']])
# Create a numpy array of the y values
y = np.array(df['y'])
# Add a column of 1 to the x values for the bias intercept term
ones_column = np.ones(len(X), dtype=int).reshape(-1, 1)
X = np.append(ones_column ,X, axis=1)
# Calculate the closed-form solution using the normal equation
theta = np.zeros(scaled_X.shape[1])

# Initialize Parameters
learning_rate = 0.01
num_iterations = 10000

#Derivation and explanation (10 points)
#o Write the model in matrix form y= X 𝛉, state the MSE objective, and derive the
#normal equation.

