from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)


# --------------------------------------------
# Linear Regression
# --------------------------------------------

def create_linear_regression():
    

    return LinearRegression()


# --------------------------------------------
# Ridge Regression
# --------------------------------------------

def create_ridge_regression(alpha=1.0):
    

    return Ridge(alpha=alpha)


# --------------------------------------------
# Decision Tree
# --------------------------------------------

def create_decision_tree():
    

    return DecisionTreeRegressor(
        max_depth=5,
        random_state=42
    )


# --------------------------------------------
# Random Forest
# --------------------------------------------

def create_random_forest():
    

    return RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        max_depth=10
    )


# --------------------------------------------
# Gradient Boosting
# --------------------------------------------

def create_gradient_boosting():
    

    return GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )