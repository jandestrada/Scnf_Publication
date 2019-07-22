from sklearn.ensemble import RandomForestClassifier
from harness.th_model_classes.class_sklearn_classification import SklearnClassification


def random_forest_classification(n_estimators=361, max_features='auto', criterion='entropy', min_samples_leaf=13,
                                 n_jobs=-1, class_weight="balanced"):
    # Creating an sklearn random forest classification model:
    rfc = RandomForestClassifier(n_estimators=n_estimators, max_features=max_features, criterion=criterion,
                                 min_samples_leaf=min_samples_leaf, n_jobs=n_jobs, class_weight=class_weight)

    # Creating an instance of the SklearnClassification TestHarnessModel subclass
    th_model = SklearnClassification(model=rfc, model_author='Hamed',
                                     model_description="Random Forest: n_estimators={0}, max_features={1}, criterion={2}, min_samples_leaf={3}, n_jobs={4}".format(
                                         n_estimators, max_features, criterion, min_samples_leaf, n_jobs))
    return th_model
