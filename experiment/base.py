import numpy as np

class BaseActiveLearning(object):
    """docstring for BaseActiveLearning"""
    def __init__(self, student, expert, data, bootstrap, step, budget, seed=123):
        super(BaseActiveLearning, self).__init__()
        self.student = student
        self.data = data
        self.bootstrap = bootstrap
        self.step = step
        self.budget = budget
        self.expert = expert
        self.rnd = np.random.RandomState(seed)

    def preprocessing_data(self, data):
        pass

    def initialize_experiment(self):
        pass

    def run_experiment(self):
        pass

    def record_results(self, results):
        pass

    def train_fn(self, x, y):
        raise Exception("Experiment does not have a training function")        

    def test_fn(self):
        raise Exception("Experiment does not have a testing function")        

    def query_fn(self, x):
        raise Exception("Experiment does not have a querying function")

    def cost_fn(self, x):
        if isinstance(x, list):
            return len(x)
        elif isinstance(array, x):
            pass
        
        return x.shape[0]

    def _size(self, x):
        if isinstance(x, list):
            return len(x)
        else:
            return x.shape[0]            
        

class ActiveExperiment(BaseActiveLearning):
    """docstring for ActiveExperiment"""
    def __init__(self, student, expert, data, bootstrap, step, budget, test=None, seed=1243):
        super(ActiveExperiment, self).__init__(student, expert, data, bootstrap, step, budget, seed=seed)
        self.test = None

    def preprocessing_data(self, data):
        return data

    def train_fn(self, x, y):
        yy = np.array(y)
        self.student.fit(x,yy)

    def test_fn(self, testx, testy):
        test = {}
        pred = self.student.predict(testx)
        test['accuracy'] = accuracy_score(y, pred)

        return test

    def query_fn(self, pool):
        return self.student.query(pool, self.step)

    def _update_index(self, list_a, list_b):
        return [la for la in list_a if la not in list_b]

    def run_experiment(self):

        training_index = []
        training_labels = []
        current_budget = 0
        iteration = 0
        x = self.data.data_x
        y = self.data.target

        pool_index = self.rnd.permutation(self._size(x))


        self.train_fn(x[pool_index[:self.bootstrap]], y[pool_index[:self.bootstrap]])

        training_labels.extend(y[pool_index[:self.bootstrap]])
        training_index.extend(pool_index[:self.bootstrap])

        pool_index = self._update_index(pool_index, pool_index[:self.bootstrap])
        results = {}

        while current_budget <= self.budget:

            query = self.query_fn(x[pool_index])
            
            new_lbl = self.expert.label(x[query], y=y[query])
            
            training_labels.extend(new_lbl)
            training_index.extend(query)
            pool_index = self._update_index(pool_index, query)
            
            self.train_fn(x[training_index], training_labels)

            test_results = self.test_fn(self.test.data_x, self.test.target)
            results.update({iteration:test_results})
            record_results(results)

            iteration += 1

        return results