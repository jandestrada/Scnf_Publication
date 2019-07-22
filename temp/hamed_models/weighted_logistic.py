from sklearn.linear_model import LogisticRegression
from harness.th_model_classes.class_sklearn_classification import SklearnClassification


def weighted_logistic_classifier():
    weighted_logistic = LogisticRegression(class_weight="balanced", n_jobs=-1)
    th_model = SklearnClassification(model=weighted_logistic, model_author='Hamed',
                                     model_description="Logistic: class_weight='balanced'")
    return th_model
