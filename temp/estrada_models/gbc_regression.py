import os
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from harness.th_model_classes.class_sklearn_regression import SklearnRegression


def ridge_regression():
    # Creating  model:
    model = GradientBoostingRegressor()

    # Creating an instance of the SklearnRegression TestHarnessModel subclass
    th_model = Ridge(model=model, model_author="Jan",
                                 model_description="Gradient Boosted Regressor, sklearn default params"
    return th_model
