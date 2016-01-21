import numpy as np

class Learner(object):
    """docstring for Learner"""
    def __init__(self, classifier):
        super(Learner, self).__init__()
        self.classifier = classifier
        self.training_size = 0
        
    def query(self, pool, step):
        raise Exception("Learner has not model")        

    def fit(self, x, y):
        raise Exception("Learner has not model")        

    def predict(self, x):
        raise Exception("Learner has not model")        

    def predict_proba(self, x):
        raise Exception("Learner has not model")        


class ActiveLearner(Learner):
    """docstring for ActiveLearner"""
    def __init__(self, classifier, rnd=122):
        super(ActiveLearner, self).__init__(classifier)
        
        self.subsample = 250

    def objective_fn(self, pool):
        """ Compute the objective function score for every instance in pool"""

        subsample = pool.shape[0]
        if self.subsample is not None:
            subsample = self.subsample
        pred = self.predict_proba(pool[:subsample])

        unc = 1 - pred.max(axis=0)

        return unc
        
    def query(self, pool, step):
        score= self.objective_fn(pool)

        # Maximize score
        top = np.argsort(score)[::-1] 
        
        return top[:step]

    def fit(self, x, y):
        
        self.classifier.fit(x, y)
        self.training_size = x.shape[0]
        return self

    def predict(self, x):
        return self.classifier.predict(x)

    def predict_proba(self, x):
        return self.classifier.predict_proba(x)
