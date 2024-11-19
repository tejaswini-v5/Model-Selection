import numpy as np
from sklearn.model_selection import KFold
from sklearn.utils import resample

class BaseModelEvaluator:
    def __init__(self, model):
    
        self.model = model

    def fit(self, X, y):
        self.model.fit(X, y)

        return self

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X, y)


class KFoldEvaluator(BaseModelEvaluator):
    def __init__(self, model, k=5, random_state=None):
        super().__init__(model)
        self.k = k

        self.random_state = random_state

    def evaluate(self, X, y):
        kf = KFold(n_splits=self.k, shuffle=True, random_state=self.random_state)
        scores = []

        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]

            y_train, y_test = y[train_index], y[test_index]

            self.fit(X_train, y_train)

            score = self.score(X_test, y_test)
            scores.append(score)

        return np.mean(scores)


class BootstrapEvaluator(BaseModelEvaluator):
    def __init__(self, model, n_iterations=100, random_state=None):
        super().__init__(model)
        self.n_iterations = n_iterations
        self.random_state = random_state

        self.rng = np.random.default_rng(random_state)

    def evaluate(self, X, y):
        
        scores = []

        for _ in range(self.n_iterations):
          
            X_resampled, y_resampled = resample(X, y, replace=True, n_samples=len(X), random_state=self.rng.integers(0, 1e6))
            
            oob_mask = ~np.isin(np.arange(len(X)), np.unique(np.where(X == X_resampled)[0]))
            X_oob = X[oob_mask]
            y_oob = y[oob_mask]

            if len(y_oob) > 0:  
                self.fit(X_resampled, y_resampled)
                score = self.score(X_oob, y_oob)
                scores.append(score)

        return np.mean(scores)
