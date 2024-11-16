from matplotlib import pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.preprocessing import PolynomialFeatures

class LinearRegression(BaseEstimator):
    """
    A class that implements linear regression (fit and predict)
    """

    def __init__(self, coefficients=None):
        self.coefficients = coefficients

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit a linear regression model to the given data.
        Add a column of ones to X for the intercept term.
        Use the closed-form equation to find the coefficients of the linear regression model.
        Store the coefficients in the coefficients attribute.

        - Input:
            X: a numpy array of shape (n, d) where n is the number of data points and d is the number of features
            y: a numpy array of shape (n,) containing the true labels
        """
        # TODO : Fit a linear regression model
        X_1 = X
        if X.ndim == 1:
            X_1= X_1.reshape(-1,1)

        X_1 = np.column_stack((np.ones(X_1.shape[0]), X_1))
        coefficients = np.linalg.inv(X_1.T.dot(X_1)).dot(X_1.T).dot(y)
        self.coefficients = coefficients
        return self

    def predict(self, X):
        """
        Predict the labels for the given data using the trained model.
        
        - Input:
            X: a numpy array of shape (n, d) where n is the number of data points and d is the number of features
        - Output:
            y_pred: a numpy array of shape (n,) containing the predicted labels
        """
        #add the bias
        X_1 = X
        
        if X.ndim == 1:
            X_1 = X.reshape(-1,1)
        

        X_1 = np.column_stack((np.ones(X_1.shape[0]),X_1))

        y_pred = X_1.dot(self.coefficients)
        return y_pred
        

    def mse(self, X, y_true):
        """
        Compute the mean squared error of the model on the given data.
        Use predict to compute y_pred and compute the MSE.

        - Input:
            X: a numpy array of shape (n, d) where n is the number of data points and d is the number of features
            y_true: a numpy array of shape (n,) containing the true labels
        - Output:
            The mean squared error of the model on the given data.
        """
        y_pred = self.predict(X)
        #subtract predictions from observations and square it, then sum it all, and divided by the shape

        diff = y_true - y_pred
        diff_2 = diff ** 2 
        mse = np.mean(diff_2)
        return mse 
        

    def get_coefficients(self):
        """
        Return the coefficients of the linear regression model.
        """
        return self.coefficients
    
    def plot_model(self, X, y, title="", xlabel="", ylabel=""):
        """
        Plot the data points and the model.

        - Input:
            X: a numpy array of shape (n, d) where n is the number of data points and d is the number of features
            y: a numpy array of shape (n,) containing the true labels
            title: a string containing the title of the plot
            xlabel: a string containing the label for the x-axis
            ylabel: a string containing the label for the y-axis
        """
        plt.plot(X, y, 'o')
        xs = np.linspace(X.min(), X.max(), 100)
        ys = self.predict(xs)
        plt.plot(xs, ys, label="Model")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.show()

class PolynomialRegression(BaseEstimator):
    """
    A class that implements polynomial regression (fit and predict)
    """
    
    def __init__(self, degree=2, coefficients=None, ridge_regression=False):
        self.degree = degree
        if ridge_regression:
            self.linear_regression = RidgeRegression(coefficients=coefficients)
        else:
            self.linear_regression = LinearRegression(coefficients=coefficients)

    def fit(self, X, y):
        """
        fit the model to the data (find the coefficients of underlying model)
        Transform X to polynomial features and then fit a linear regression model to the transformed data.

        X needs to be transformed into X_poly where each column of X_poly is X raised to the power of i, where i is the column index.
        fit the parameters of self.linear_regression to X_poly and y.

        - Input:
            X: a numpy array of shape (n, d) where n is the number of data points and d is the number of features
            y: a numpy array of shape (n,) containing the true labels
        """

        poly_x = np.array([X ** i for i in range(1,self.degree+ 1)])
        self.linear_regression.fit(poly_x.T,y)
      

    def predict(self, X):
        """
        Get predictions from underlying model.

        Again, X must be transformed into polynomial features before being passed to the underlying model.
        Once X has been transformed, you can call predict on X_poly.

        - Input:
            X: a numpy array of shape (n, d) where n is the number of data points and d is the number of features
        - Output:
            y_pred: a numpy array of shape (n,) containing the predicted labels
        """
        # TODO: Get predictions from underlying model.
        
        
        poly_x = np.array([X ** i for i in range(1,self.degree+ 1)])
        return self.linear_regression.predict(poly_x.T)
        

    def mse(self, X, y_true):
        """
        Compute the mean squared error of the model on the given data.
        Use the underlying model to compute the MSE.

        - Input:
            X: a numpy array of shape (n, d) where n is the number of data points and d is the number of features
            y_true: a numpy array of shape (n,) containing the true labels
        - Output:
            The mean squared error of the model on the given data.
        """
        # TODO: Compute the mean squared error
        #subtract predictions from observations and square it, then sum it all, and divided by the shape

        y_pred = self.predict(X)
        #subtract predictions from observations and square it, then sum it all, and divided by the shape

        diff = y_true - y_pred
        diff_2 = diff ** 2 
        mse = np.mean(diff_2)
        return mse 
        
    
    def plot_model(self, X, y, title="", xlabel="", ylabel=""):
        self.linear_regression.plot_model(X, y, title, xlabel, ylabel)
    
    def get_coefficients(self):
        return self.linear_regression.get_coefficients()

class RidgeRegression(LinearRegression):
    def __init__(self, lam=0.5, coefficients=None):
        self.lam = lam
        self.coefficients = coefficients
  
    def fit(self, X, y):
        """
        Fit a ridge regression model to the given data.
        Add a column of ones to X for the intercept term.
        Use the closed-form equation to find the coefficients of the ridge regression model.
        Store the coefficients in the coefficients attribute.

        - Input:
            X: a numpy array of shape (n, d) where n is the number of data points and d is the number of features
            y: a numpy array of shape (n,) containing the true labels
        """
        # TODO: Fit a ridge regression model
        #use numpy function to find the identity matrix
        X_1 = X
        if X.ndim == 1:
            X_1= X_1.reshape(-1,1)

        X_1 = np.column_stack((np.ones(X_1.shape[0]), X_1))
        i = np.eye(X_1.shape[1])
        coefficients = np.linalg.inv(X_1.T.dot(X_1) + self.lam*i).dot(X_1.T).dot(y)
        self.coefficients = coefficients
        pass
