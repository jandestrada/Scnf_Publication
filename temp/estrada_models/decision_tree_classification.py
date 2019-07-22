from sklearn.tree import DecisionTreeClassifier
from harness.th_model_classes.class_sklearn_classification import SklearnClassification


def decision_tree_classification():
    # Creating  model:
    model = DecisionTreeClassifier()

    # Creating an instance of the SklearnClassification TestHarnessModel subclass
    th_model = SklearnClassification(model=model, model_author='Jan',
                                     model_description="Decision Tree Classifier, sklearn default params")
    return th_model
