import os
import csv
import numpy
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix, f1_score


def build_data_frame(path):
    rows = []
    index = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in list(reader)[:314]:
            rows.append({'text': row.get('purpose'), 'class': row.get('classified')})

    data_frame = DataFrame(rows)
    return data_frame


data = build_data_frame("training.csv")

data = data.reindex(numpy.random.permutation(data.index))

pipeline = Pipeline([
    ('count_vectorizer',    CountVectorizer()),
    ('tfidf',               TfidfTransformer()),
    ('classifier',          SGDClassifier()),
    ])

k_fold = KFold(n=len(data), n_folds=6)
scores = []
confusion = numpy.array([[0, 0], [0, 0]])
for train_indices, test_indices in k_fold:
    train_text = data.iloc[train_indices]['text'].values
    train_y = data.iloc[train_indices]['class'].values.astype(str)

    test_text = data.iloc[test_indices]['text'].values
    test_y = data.iloc[test_indices]['class'].values.astype(str)

    pipeline.fit(train_text, train_y)
    predictions = pipeline.predict(test_text)

    #confusion += confusion_matrix(test_y, predictions)
    score = f1_score(test_y, predictions)
    scores.append(score)

print('Total classified:', len(data))
print('Score:', sum(scores)/len(scores))
#print('Confusion matrix:')
#print(confusion)