from sklearn.naive_bayes import GaussianNB
from harness.th_model_classes.class_sklearn_classification import SklearnClassification


def naive_bayes_classification():
    # Creating  model:
    nbc = GaussianNB()

    # Creating an instance of the SklearnClassification TestHarnessModel subclass
    th_model = SklearnClassification(model=nbc, model_author='Jan',
                                     model_description="Naive Bayes Classifier, sklearn default params")
    return th_model
