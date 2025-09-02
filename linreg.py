import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd
import array

def cost_function(X, y, theta):
    y_length = len(y)
    h_dot_product = X.dot(theta)
    J = 1/(2*y_length) * np.sum(np.square(h_dot_product - y_length))
    return J

def standard_normalizer(X):
    # Standardize the data
    scaler = StandardScaler()
    # Fit the scaler to the data and transform the data
    scaled_X = scaler.fit_transform(X)
    return scaled_X

def gradient_descent(X, y, theta, alpha, num_iters):
    y_length = len(y)
    J_history = []
    for i in range(num_iters):
        h_dot_product = X.dot(theta)
        theta = theta - (alpha/y_length) * (X.T.dot(h_dot_product - y))
        J_history.append(cost_function(X, y, theta))
    return theta, J_history


#===========================================================
# Turn the CSV file into a pandas dataframe
df = pd.read_csv("d3.csv")

df.to_numpy()

# Create a numpy array of the x values
X = np.array(df[['x1', 'x2', 'x3']])

# Create a numpy array of the y values
y = np.array(df['y'])

#===========================================================
# Add a column of 1 to the x values for the bias intercept term

ones_column = np.ones(len(X), dtype=int).reshape(-1, 1)
X = np.append(X, ones_column, axis=1)

#===========================================================
# Time to train the model

model = LinearRegression()

# Set the learning rate and number of iterations
model.learning_rate = 0.01
model.n_iter = 1000

#Standardize the data. Subtract the mean and divide by the standard deviation
x_standardized = standard_normalizer(X)

# Train the model
model.fit(x_standardized, y)

# Make predictions with the model with new data
x_new = [[1,1,1,1], [2,0,4,1], [3,2,1,1]]
x_new = np.array(x_new)
y_pred = model.predict(x_new)
#print(y_pred)


#===========================================================


