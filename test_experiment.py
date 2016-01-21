
from expert.document_expert import TrueExpert
from learner.base import ActiveLearner
from sklearn.linear_model import LogisticRegression
from experiment.base import ActiveExperiment
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer




clf = LogisticRegression(penalty='l1', C=1)
documents = datasets.fetch_20newsgroups(shuffle=True, random_state=1,
	remove=('headers', 'footers', 'quotes'))
vectorizer =  CountVectorizer(max_df=1., min_df=5)

documents.data_x = vectorizer.fit_transform(documents.data)

student = ActiveLearner(clf)
expert = TrueExpert(None)

experiment = ActiveExperiment(student, expert, documents, 10, 1, 50, seed=1)
# print experiment.student
res = experiment.run_experiment()
print res