import pytest
import numpy as np
from sklearn.datasets import load_iris, load_wine
from sklearn.tree import DecisionTreeClassifier
from model_evaluators import KFoldEvaluator, BootstrapEvaluator  


@pytest.fixture
def datasets():
    return {
        "Iris": load_iris(),
        "Wine": load_wine(),

    }

@pytest.fixture
def model():
    return DecisionTreeClassifier(random_state=42)


def test_kfold_evaluator(datasets, model):
    for name, data in datasets.items():
        X, y = data.data, data.target
        evaluator = KFoldEvaluator(model=model, k=5, random_state=42)
        score = evaluator.evaluate(X, y)
        
        assert 0 <= score <= 1, f"K-Fold score for {name} dataset is out of range: {score}"

       
        evaluator2 = KFoldEvaluator(model=model, k=5, random_state=42)
        score2 = evaluator2.evaluate(X, y)
        assert score == pytest.approx(score2), f"K-Fold results for {name} dataset are not reproducible."


def test_bootstrap_evaluator(datasets, model):
    for name, data in datasets.items():
        X, y = data.data, data.target
        evaluator = BootstrapEvaluator(model=model, n_iterations=100, random_state=42)
        score = evaluator.evaluate(X, y)
        

        assert 0 <= score <= 1 or np.isnan(score), f"Bootstrap score for {name} dataset is invalid: {score}"


        evaluator2 = BootstrapEvaluator(model=model, n_iterations=100, random_state=42)
        score2 = evaluator2.evaluate(X, y)
        assert score == pytest.approx(score2), f"Bootstrap results for {name} dataset are not reproducible."