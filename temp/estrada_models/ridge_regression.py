import os
import pandas as pd
from sklearn.linear_model import Ridge
from harness.th_model_classes.class_sklearn_regression import SklearnRegression


def ridge_regression():
    # Creating a Ridge regression model
    model = Ridge()

    # Creating an instance of the SklearnRegression TestHarnessModel subclass
    th_model = Ridge(model=model, model_author="Jan",
                                 model_description="Ridge Regression, sklearn default params"
    return th_model
