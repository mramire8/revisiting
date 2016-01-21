
class BaseExpert(object):
    """docstring for Expert"""
    def __init__(self, classifier):
        super(BaseExpert, self).__init__()
        self.classifier = classifier
    
    def label(self, x, y=None):
        raise Exception("Expert has not model")        

    def fit(self, x, y=None):
        raise Exception("Expert has not model")        
