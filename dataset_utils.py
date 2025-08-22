from sklearn.datasets import make_classification, make_regression
import pandas as pd

def get_classification_data():
    X, y = make_classification(n_samples=500, n_features=2, n_informative=2, n_redundant=0, random_state=42)
    return pd.DataFrame(X, columns=["feature1", "feature2"]), pd.Series(y)

def get_regression_data():
    X, y = make_regression(n_samples=500, n_features=2, noise=10, random_state=42)
    return pd.DataFrame(X, columns=["feature1", "feature2"]), pd.Series(y)
