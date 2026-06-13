from sklearn.linear_model import LogisticRegression


class RetinaClassifier:
    """Simple logistic regression wrapper for final prediction.

    This placeholder shows where the CNN features will feed into the classifier.
    """

    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
