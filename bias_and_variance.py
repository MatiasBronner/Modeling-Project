import argparse
from matplotlib import pyplot as plt
import numpy as np
from models import PolynomialRegression
import argparse

def f_x(x: float) -> float:
    """
    The function to be approximated
    -Inputs:
        x: input
    -Outputs:
        y: output
    """
    return np.sin(np.pi * x)

def sample_function(func, noise_std=0.25, n_points=10, xlim=(-1, 1)) -> np.ndarray:
    """
    Sample a function with noise. 
    1. Sample n x's uniformly from bounds of x (xlim).
    2. Sample y from a normal distribution with mean func(x) and std noise_std.
    3. Return x and y.

    You can assume that the function is univariate (i.e., x is a scalar and y is a scalar, f: R -> R)

    -Inputs:
        func: function to sample. Function takes in a single input and returns a single output
        noise_std: standard deviation of noise
        n_points: number of samples to draw
        xlim: bounds of x
    -Outputs:
        x: inputs
        y: outputs
    """
    # TODO: Sample from f
    x = np.random.uniform(xlim[0],xlim[1],n_points)
    y = np.random.normal(func(x),noise_std)
    return x,y 
    

def visualize_data(original_function, sampled_points=[], xlim=(-1, 1)):
    """
    Visualize the original function and the sampled points (if there are any)

    -Inputs:
        original_function: function to visualize
        sampled_points: list of sampled points
        xlim: bounds of x
    """
    plt.plot(np.linspace(xlim[0], xlim[1], 100), original_function(np.linspace(xlim[0], xlim[1], 100)), label="f(x)")
    if sampled_points:
        plt.scatter(sampled_points[0], sampled_points[1], label="Observed Data", color="orange")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()

def calculate_mse(X, y, estimators):
    """
    Calculate the expected MSE of the provided estimators.
    MSE is the average squared error of an estimator.
    The expected MSE is the average of MSE over all provided estimators.

    - Inputs:
        X: input features
        y: true output feature
        estimators: list of estimators
    - Outputs:
        mse: mean squared error
    """
    mse = np.mean([estimator.mse(X, y) for estimator in estimators])
    return mse

def build_sample_estimator(original_function, degree, num_points=10, noise_std=0.25, ridge_regression=False) -> tuple[PolynomialRegression, float]:
    """
    build sample estimator
    1. Sample n points from the original function with desired noise
    2. Fit a polynomial regression model to the sampled points
    3. Return the estimator and the training MSE

    -Inputs:
        original_function: function to sample
        degree: degree of the polynomial
        num_points: number of points to sample
        noise_std: standard deviation of noise
        ridge_regression: whether to use ridge regression
    -Outputs:
        estimator: estimator (PolynomialRegression) fit to the sampled points
        training_MSE: training MSE of the estimator on the sampled points
    """
    # TODO: Build you sample estimator
    x,y = sample_function(original_function,noise_std,num_points)
    model = PolynomialRegression(degree=degree,ridge_regression=ridge_regression)
    model.fit(x,y)    
    error = model.mse(x,y)
    return model, error
    

def build_many_estimators(original_function, degree: int, num_estimators=10, num_points=10, noise_std=0.25, ridge_regression=False) -> tuple[list[PolynomialRegression], list[float]]:
    """
    Build many estimators (num_estimators).
    Return a list of estimators and their training errors.

    -Inputs:
        original_function: function to sample
        degree: degree of the polynomial
        num_estimators: number of estimators to build
        num_points: number of points to sample
        noise_std: standard deviation of noise
        ridge_regression: whether to use ridge regression
    -Outputs:
        estimators: list of estimators
        errors: list of training errors. Each element in the list is the training error of the corresponding estimator
    """
    # TODO: Make your list of estimators
    models = []
    errors =[]
    for _ in range(num_estimators):
        model, error = build_sample_estimator(original_function, degree, num_points, noise_std,ridge_regression)
        models.append(model)
        errors.append(error)
    return models,errors
    

def calculate_bias_squared(X, y_true, sampled_estimators):
    """
    Calculate the bias of the provided average estimator
    Bias is the difference between the expected value of the estimator (i.e., average_estimator)
    and the observed data label y_true for some sample x.

    Bias is defined for a single point x. To get the bias of the estimator, we average the bias over all x.

    -Inputs:
        X: input features
        y_true: true output feature
        average_estimator: average estimator
    -Outputs:
        bias_sq: mean squared bias
    """
    # TODO: Calculate bias^2
    #to calculat bias you find the expected value of the function and divide it by 
    
    #over all sampled estimators, run the predictions with x, find the average predictions, subtract the average predictions 
    # from the actual prediction, square that bitch
    predictions = []
    for model in sampled_estimators:
        prediction = model.predict(X)
        predictions.append(prediction)
    predictions = np.array(predictions)
    expected_val = np.mean(predictions, axis=0)
    bias = expected_val - y_true
    bias_sq = bias ** 2
    bias_sq = np.mean(bias_sq)
    return bias_sq

def calculate_variance(X, sampled_estimators):
    """
    Calculate the variance of the provided sampled estimators.
    Variance is the average squared deviation of the estimator from its mean.
    For each each sample in X, we want the variance of the sampled_estimators predictions.
    Variance is defined for a single point x. To get the variance of the estimator, we average the variance over all x.

    -Inputs:
        X: input features
        sampled_estimators: list of estimators
    -Outputs:
        var: mean variance
    """
    # TODO: Calulate varience
    #to find variance we do E(x^2 ) - E(x)^2
    predictions = []
    for model in sampled_estimators:
        prediction = model.predict(X)
        predictions.append(prediction)
    predictions = np.array(predictions)

    var = np.var(predictions,axis=0)
    var = np.mean(var)

    return var 
    

def calculate_bias_variance_mse(X, y_true, sampled_estimators):
    """
    Calculate bias, variance, and MSE correctly for multiple estimators, print results, and return bias, variance, and MSE

    -Inputs:
        X: input features
        y_true: true output feature
        sampled_estimators: list of estimators
        average_estimator: average estimator (E[\\hat{f}(x)])])
    -Outputs:
        bias: squared bias
        variance: variance
        mse: mean squared error
    """
    bias_sq = calculate_bias_squared(X, y_true, sampled_estimators)
    
    variance = calculate_variance(X, sampled_estimators)
    
    mse = calculate_mse(X, y_true, sampled_estimators)

    print("Degree: ", len(sampled_estimators[0].get_coefficients())-1, "Bias^2:", bias_sq, "Variance:", variance, "MSE:", mse, "Bias^2 + Variance", bias_sq + variance)

    return bias_sq, variance, mse

def visualize_estimator(original_function, estimator, xlim=(-1, 1)):
    """
    Visualize the original function and the estimator
    """
    plt.plot(np.linspace(xlim[0], xlim[1], 100), original_function(np.linspace(xlim[0], xlim[1], 100)), label="f(x)")
    plt.plot(np.linspace(xlim[0], xlim[1], 100), estimator.predict(np.linspace(xlim[0], xlim[1], 100)), label="Estimator")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.ylim(-2, 2)
    plt.legend()
    plt.show()

def visualize_estimators(original_function, estimators, xlim=(-1, 1), blocking=False):
    """
    Visualize the original function and the estimators
    """
    X = np.linspace(xlim[0], xlim[1], 100)
    mean_predictions = np.mean([estimator.predict(X) for estimator in estimators], axis=0)
    plt.figure(figsize=(12, 8))
    plt.plot(X, original_function(X), label="f(x)")
    for estimator in estimators:
        plt.plot(X, estimator.predict(X), alpha=0.1, color="orange")
    plt.plot(X, mean_predictions, label="Average Estimator", color="orange")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.ylim(-2, 2)
    plt.legend()
    plt.savefig(f"figs/estimators_d_{len(estimators[0].get_coefficients()) - 1}.png")
    plt.show(block=blocking)

def make_train_test_plot(args):
    """
    Make a plot comparing train error, test error, bias, and variance for different degrees of polynomial regression
    This requires completion of the following functions:
        - build_many_estimators
        - calculate_bias_variance_mse
        - visualize_estimators

    -Inputs:
        args: command line arguments arguments
    """
    all_train_error = []
    all_test_error = []
    biases = []
    variances = []
    mses = []
    for i in range(1, 10):
        # Generate multiple estimators
        estimators, train_errors = build_many_estimators(f_x, degree=i, num_estimators=args.num_estimators, num_points=args.num_points, noise_std=.25, ridge_regression=args.ridge_regression)
        visualize_estimators(f_x, estimators)

        # Plot bias^2, variance, and MSE
        x_test, y_test = sample_function(f_x, noise_std=0.25, n_points=1000, xlim=(-1, 1))
        bias, variance, mse = calculate_bias_variance_mse(x_test, y_test, sampled_estimators=estimators)

        test_error = mse
        all_test_error.append(test_error)
        all_train_error.append(np.mean(train_errors))
        biases.append(bias)
        variances.append(variance)
        mses.append(mse)
    
    plt.clf()
    plt.plot(range(1, 10), all_train_error, label="Train MSE", linestyle="--")
    plt.plot(range(1, 10), all_test_error, label="Test MSE", linestyle="-.")
    plt.plot(range(1, 10), biases, label="Bias^2", linestyle=":")
    plt.plot(range(1, 10), variances, label="Variance", linestyle="-")
    plt.ylim(0, 2)
    plt.xlabel("Degree")
    plt.ylabel("MSE")
    plt.legend()
    plt.savefig(f"figs/train_test_error.png")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Polynomial Regression with Bias-Variance Analysis")
    parser.add_argument('--visualize-sample', action='store_true', help='Visualize a sampled set of data')
    parser.add_argument('--num_points', type=int, default=15, help='Number of points to sample and fit')
    parser.add_argument('--num_estimators', type=int, default=100, help='Number of estimators to generate')
    parser.add_argument('-d', '--degree', type=int, default=1, help='Degree of the polynomial')
    parser.add_argument('--noise_std', type=float, default=0.25, help='Standard deviation of the noise')
    parser.add_argument('--visualize-estimators', action='store_true', help='Visualize the estimators')
    parser.add_argument('--visualize-loss', action='store_true', help='Visualize the loss curves')
    parser.add_argument('--ridge-regression', action='store_true', help='Use ridge regression')
    parser.add_argument('--lam', type=float, default=0.1, help='Lambda for ridge regression')
    return parser.parse_args()


def main():
    args = parse_arguments()
    noise_std = args.noise_std
    n = args.num_points
    d = args.degree
    num_estimators = args.num_estimators
    if args.visualize_sample:
        visualize_data(f_x, sample_function(f_x, noise_std, xlim=(-1, 1), n_points=n))
    if args.visualize_estimators:
        sampled_estimators, errors = build_many_estimators(f_x, degree=d, num_points=n, num_estimators=num_estimators, noise_std=0.25, ridge_regression=args.ridge_regression)
        visualize_estimators(f_x, sampled_estimators, blocking=True)
    if args.visualize_loss:
        make_train_test_plot(args)
    
if __name__ == "__main__":
    main()
