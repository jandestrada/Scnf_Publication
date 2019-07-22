from sklearn.neighbors import KNeighborsClassifier
from harness.th_model_classes.class_sklearn_classification import SklearnClassification


def knn_classifier():
    # Creating  model:
    model = KNeighborsClassifier()

    # Creating an instance of the SklearnClassification TestHarnessModel subclass
    th_model = SklearnClassification(model=model, model_author='Jan',
                                     model_description="K-NN classifier, sklearn default params")
    return th_model
