from base import BaseExpert

class TrueExpert(BaseExpert):
    """docstring for TrueExpert"""
    def __init__(self, classifier):
        super(TrueExpert, self).__init__(classifier)
        

    def label(self, x, y=None):
        if y is None:
            raise Exception("True labels are missing")
        else:
            return y

    def fit(self, x, y=None):
        return self


class PredictingExpert(BaseExpert):
    """docstring for PredictingExpert"""
    def __init__(self, classifier):
        super(PredictingExpert, self).__init__()
        self.classifier = classifier
    
    def label(self, x, y=None):
        return self.classifier.predict(x)

    def fit(self, x, y=None):
        self.classifier.fit(x, y)
        return self