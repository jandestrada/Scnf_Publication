from sklearn.mixture import GaussianMixture
from harness.th_model_classes.class_sklearn_classification import SklearnClassification


def gmm_classification():
    # Creating  model:
    model = GaussianMixture(n_components=2)

    # Creating an instance of the SklearnClassification TestHarnessModel subclass
    th_model = SklearnClassification(model=model, model_author='Jan',
                                     model_description="Gaussian Mixture Classifier, n_components=2")
    return th_model
