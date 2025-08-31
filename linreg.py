import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd
import array

def cost_function(X, y, theta):
    m = len(y)
    h = X.dot(theta)
    J = 1/(2*m) * np.sum(np.square(h - y))
    return J

def standard_normalizer(X):
    # Standardize the data
    scaler = StandardScaler()
    # Fit the scaler to the data and transform the data
    scaled_X = scaler.fit_transform(X)
    return scaled_X

#print("Hello World!")

# Turn the CSV file into a pandas dataframe
df = pd.read_csv("d3.csv")

#df.columns = ["x1", "x2", "x3", "y"]


df.to_numpy()

# data = {'x1': [1, 2, 3], 'x2': [4, 5, 6], 'y': [7, 8, 9]}
# df = pd.DataFrame(data)

# Print the first 5 rows of the dataframe to see if it worked
#print(df.head())

# Create a numpy array of the x values
X = np.array(df[['x1', 'x2', 'x3']])

# Create a numpy array of the y values
y = np.array(df['y'])

# Print the x and y values to see if it worked
#print(X)
#print(y)
#print(len(df.axes[0])) # number of rows ; 100
#print(len(df.axes[1])) # number of columns ; 4

# Add a column of 1 to the x values for the bias intercept term
ones_column = np.ones(len(X), dtype=int).reshape(-1, 1)
#print(ones_column)
#print(df.loc[0])
X = np.append(X, ones_column, axis=1)
#print(X,y)


model = LinearRegression()
# set the learning rate and number of iterations
model.learning_rate = 0.01
model.n_iter = 10000
# Train the model
model.fit(X, y)
# Make predictions
x_new = [[1,1,1,1], [2,0,4,1], [3,2,1,1]]
x_new = np.array(x_new)
y_pred = model.predict(x_new)
#print(y_pred)

#Standardize the data
x_standardized = standard_normalizer(X)

#Cost function
Cost = cost_function(X, y, model.coef_)
#print(Cost)


