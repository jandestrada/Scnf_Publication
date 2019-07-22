from sklearn.ensemble import GradientBoostingClassifier
from harness.th_model_classes.class_sklearn_classification import SklearnClassification


def gbc_classification():
    # Creating  model:
    model = GradientBoostingClassifier()

    # Creating an instance of the SklearnClassification TestHarnessModel subclass
    th_model = SklearnClassification(model=model, model_author='Jan',
                                     model_description="Gradient Boosting Classifier, sklearn default params")
    return th_model
