import sys
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from models import LinearRegression, RidgeRegression
import argparse
import numpy as np
from sklearn.linear_model import Lasso

def corr(X: np.ndarray, feature_names: list[str]):
    """
    Calculate the correlation matrix for the given dataset.
    X is a n x d matrix, where n is the number of samples and d is the number of features.
    feature_names is a list of feature names (length d) corresponding to the columns of X.

    How you process this data is up to you. You may find producing a heatmap and
    labelling the axes with feature names useful (plt.imshow).

    You may sort the features by their correlation with life_expectancy and check out the highest or lowest correlation.

    - Inputs:
        - X: Features (np.ndarray)
        - feature_names: Names of the features (list of strings)
    """
    # TODO: Calculate the correlation
    

    correlation = np.corrcoef(X.T)
    plt.imshow(correlation)
    plt.colorbar()
    plt.show()

    return correlation

def single_feature_regression(X: np.ndarray, y: np.ndarray, feature_name: str):
    """
    Train a linear regression model for a single feature.
    Plot the model and the data, print train and test MSE.

    - Inputs:
        - X: Features (np.ndarray)
        - y: Target variable (np.ndarray)
        - feature_name: Name of the feature (string)
    """
    # TODO: Train a linear regression model for a single feature.
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    model = LinearRegression()
    model.fit(X_train,y_train)
    model.plot_model(X,y, title= feature_name + " vs Life Expectancy", xlabel=feature_name, ylabel="Life Expectancy")
    print("Testing MSE for " + feature_name + " is " + str(model.mse(X_test,y_test)))
    print("Training MSE for " + feature_name + " is " + str(model.mse(X_train,y_train)))
    

def extract_features(df: pd.DataFrame, feature: str):
    """
    Extract specific feature from dataset. (i.e., BMI or Schooling)
    
    - Inputs:
        - df: DataFrame containing the data
        - feature: Name of the feature to extract
    - Outputs:
        - X: Features (np.ndarray)
        - y: Target variable (np.ndarray)
        - feature_names: Names of the features
    """
    df = df.drop_duplicates(subset=['Country'])
    df = df.select_dtypes(include='number')
    if feature:
        if feature not in df.columns:
            print("Error...")
            print(f"Feature {feature} not found in the dataset.")
            print("Available features:", df.columns)
            sys.exit()
        data = df[[feature, 'life_expectancy']].dropna()
    else:
        # If no specific feature is requested, return all features
        data = df
    feature_names = data.drop('life_expectancy', axis=1).columns
    return data.drop('life_expectancy', axis=1).to_numpy(dtype=float), data['life_expectancy'].to_numpy(dtype=float), feature_names

def preprocess_data(df):
    """
    Preprocess the data by standardizing the numeric columns and dropping missing values.
    Does not standardize life_expectancy.

    - Inputs:
        - df: DataFrame containing the data
    - Outputs:
        - data: DataFrame with standardized numeric columns and dropped missing values
    """
    data = df.drop('life_expectancy', axis=1)
    numeric_columns = data.select_dtypes(include='number').columns
    data[numeric_columns] = (data[numeric_columns] - data[numeric_columns].min()) / (data[numeric_columns].max() - data[numeric_columns].min())

    data = data.dropna()
    # Add Life_expectancy back
    data['life_expectancy'] = df['life_expectancy']
    return data

def compare_models(X: np.ndarray, y: np.ndarray, args, feature_names=[]):
    """
    Compare different models and print their MSEs and coefficients.
    Split data into 80/20 train/test set split.
    Train a Linear Regression, Ridge Regression, and Lasso Regression.
    Report MSEs on train and test sets.

    - Inputs:
        - X: Features
        - y: Target variable
        - args: Command line arguments
        - feature_names: Names of the features (for printing)
    """
    lam = args.lam
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = []
    if args.linear:
        model = LinearRegression()
        model.fit(X_train, y_train)
        models.append(model)
    if args.ridge:
        model = RidgeRegression(lam=lam)
        model.fit(X_train, y_train)
        models.append(model)
    if args.lasso:
        model = Lasso(alpha=lam)
        model.fit(X_train, y_train)
        models.append(model)
    
    if args.all_models:
        models = [LinearRegression(), RidgeRegression(lam=lam), Lasso(alpha=lam)]
        for model in models:
            model.fit(X_train, y_train)
    for model in models:
        if hasattr(model, 'mse'):
            test_mse = model.mse(X_test, y_test)
            train_mse = model.mse(X_train, y_train)
        else:
            # Lasso Regression does not have an mse function :(, .score returns R^2
            test_mse = np.mean((model.predict(X_test) - y_test) ** 2)
            train_mse = np.mean((model.predict(X_train) - y_train) ** 2)
        
        print(f"Model: {model.__class__.__name__}, Train MSE: {train_mse}")
        print(f"Model: {model.__class__.__name__}, Test MSE: {test_mse}")

        # Lasso Regression does not have .coefficients, only .coef_
        if (hasattr(model, 'get_coefficients')):
            for f, c in zip(feature_names, model.get_coefficients()):
                print(f"{f}: {c:0.2f}")
            print()
        else:
            for f, c in zip(feature_names, model.coef_):
                print(f"{f}: {c:0.2f}")
            print()

def parse_args():
    parser = argparse.ArgumentParser(description="Linear Regression")
    parser.add_argument("--correlation", action='store_true', help="Generate correlation matrix")
    parser.add_argument("--lam", type=float, default=0.1, help="Lambda value for Ridge Regression")
    parser.add_argument("--feature", type=str, default=[], help="Features to use for regression")
    parser.add_argument("--all-features", action='store_true', help="Fit all features")
    parser.add_argument("--linear", action='store_true', help="Fit Linear Regression")
    parser.add_argument("--lasso", action='store_true', help="Fit Lasso Regression")
    parser.add_argument("--ridge", action='store_true', help="Fit Ridge Regression")
    parser.add_argument("--all-models", action='store_true', help="Fit All models (Linear, Ridge, Lasso)")
    return parser.parse_args()

def main():
    args = parse_args()

    # Load the dataset
    df = pd.read_csv('data/life_expectancy.csv')
    df = preprocess_data(df)

    if args.correlation:
        # Get correlation matrix
        X, y, feature_names = extract_features(df, None)
        corr(X, feature_names)

    
    if args.feature:
        # Plot model for single feature
        feature = args.feature
        X, y, feature_names = extract_features(df, feature)
        single_feature_regression(X, y, feature)
    
    if args.all_features:
        # Train model(s) for all features
        X, y, feature_names = extract_features(df, None)
        compare_models(X, y, args, feature_names=feature_names)
    



if __name__ == "__main__":
    main()