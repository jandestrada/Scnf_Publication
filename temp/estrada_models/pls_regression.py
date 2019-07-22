import os
import pandas as pd
from sklearn.cross_decomposition import PLSRegression()
from harness.th_model_classes.class_sklearn_regression import SklearnRegression


def pls_regression():
    # Creating  model:
    model = PLSRegression()

    # Creating an instance of the SklearnRegression TestHarnessModel subclass
    th_model = Ridge(model=model, model_author="Jan",
                                 model_description="Partial least squares regression, sklearn default params"
    return th_model
