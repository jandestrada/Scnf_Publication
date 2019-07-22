from sklearn.svm import SVC
from harness.th_model_classes.class_sklearn_classification import SklearnClassification


def svm_classification():
    # Creating  model:
    model = SVC(probability=True)

    # Creating an instance of the SklearnClassification TestHarnessModel subclass
    th_model = SklearnClassification(model=model, model_author='Jan',
                                     model_description="Support Vector Classifier, sklearn default params")
    return th_model
