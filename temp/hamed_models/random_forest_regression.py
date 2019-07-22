import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from harness.th_model_classes.class_sklearn_regression import SklearnRegression


def random_forest_regression(bootstrap=False, min_samples_leaf=1, n_estimators=689, min_samples_split=2,
                             max_features=0.2, max_depth=86, n_jobs=-1):
    # Creating an sklearn random forest regression model:
    rfr = RandomForestRegressor(bootstrap=bootstrap, min_samples_leaf=min_samples_leaf, n_estimators=n_estimators,
                                min_samples_split=min_samples_split, max_features=max_features, max_depth=max_depth, n_jobs=n_jobs)

    # Creating an instance of the SklearnRegression TestHarnessModel subclass
    th_model = SklearnRegression(model=rfr, model_author="Hamed",
                                 model_description="Random Forest: bootstrap={}, min_samples_leaf={}, n_estimators={}, min_samples_split={}, max_features={}, max_depth={}, n_jobs={}".format(
                                     bootstrap, min_samples_leaf, n_estimators, min_samples_split, max_features, max_depth, n_jobs))
    return th_model
