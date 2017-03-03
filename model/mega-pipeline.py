#Load required packages
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import SGDRegressor, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline, FeatureUnion

#Create custom classes
class ItemSelector(BaseEstimator, TransformerMixin):
    
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]

class NoFitMixin:
    def fit(self, X, y=None):
        return self

class SelectTransform(TransformerMixin, NoFitMixin, BaseEstimator):
    def __init__(self, func, copy=True):
        self.func = func
        self.copy = copy

    def transform(self, X):
        X_ = X if not self.copy else X.copy()
        return self.func(X_)
        
#Read in csv
reviews = pd.read_csv('Reviews.csv', index_col = 'Id', usecols = ['Id', 'Summary', 'Text', 'Score'])

#Drop duplicate score-text values
reviews = reviews.drop_duplicates(subset = ['Summary', 'Text', 'Score'])

#Create text - remove line breaks
reviews['Summary'] = reviews['Summary'].str.replace('<br />', ' ')
reviews['Text'] = reviews['Text'].str.replace('<br />', ' ')

#Replace nan with ""
reviews['Summary'] = reviews['Summary'].fillna(value = "")
reviews['Text'] = reviews['Text'].fillna(value = "")

#Extract target variable
score = reviews['Score']

#Remove score from reviews
reviews = reviews.drop('Score', axis = 'columns')

#Count the number of words
reviews['summary_count'] = reviews['Summary'].str.split().apply(len)
reviews['text_count'] = reviews['Text'].str.split().apply(len)
reviews = reviews.assign(all_words_count = reviews['summary_count'] + reviews['text_count'])

#Create train and test stratified w.r.t score
train, test, train_score, test_score = train_test_split(reviews, score, train_size = 0.5,
                                                        stratify = score, random_state = 44)

#Order train_score and test_score by index
train = train.sort_index()
train_score = train_score.sort_index()
test = test.sort_index()
test_score = test_score.sort_index()

pipeline = Pipeline([
    ('union', FeatureUnion([
        
        ('summary', Pipeline([
            ('selector', ItemSelector(key = 'Summary')),
            ('count_vectorizer', CountVectorizer(min_df = 0.0005, max_df = 1.0)),
        ])),
        
        ('text', Pipeline([
            ('selector', ItemSelector(key = 'Text')),
            ('count_vectorizer', CountVectorizer(min_df = 0.001, max_df = 1.0)),
        ])),
        
        #Select all numerical features
        ('numerical', Pipeline([
            ('select', SelectTransform(lambda X: X.select_dtypes(exclude=['object']))),
            #Code to select all categorial features:
            #('select', DFTransform(lambda X: X.select_dtypes(include=['object']))),
        ])),
    ])),
    
    ('scaler',  MaxAbsScaler()),

    #Only instatiating a model
    ('model', SGDRegressor()),
])

#Parameters to be searched
param_grid = [
    {'model' : [SGDRegressor(loss = 'squared_loss', n_iter = np.ceil(10**6 /train.shape[0]), random_state = 44,
                             verbose = 3)],
        'model__l1_ratio' : [0, 0.5, 1],
        'model__alpha' : [0.0001, 0.1, 1, 10],
        'union__summary__count_vectorizer__ngram_range': [(1, 1), (1, 2)],
        'union__text__count_vectorizer__ngram_range': [(1, 1), (1, 2)]},
    {'model' : [RandomForestRegressor(n_estimators = 2, random_state = 44, verbose = 3, n_jobs = -1)],
        'model__max_depth' : [3],
        'scaler' : [None],
        'union__summary__count_vectorizer__ngram_range': [(1, 1), (1, 2)],
        'union__text__count_vectorizer__ngram_range': [(1, 1), (1, 2)]},
    {'model' : [LinearRegression(n_jobs = -1)],
        'union__summary__count_vectorizer__ngram_range': [(1, 1), (1, 2)],
        'union__text__count_vectorizer__ngram_range': [(1, 1), (1, 2)]},
    {'model' : [Ridge(random_state = 44)],
        'model__alpha' : [0.1, 1, 10],
        'union__summary__count_vectorizer__ngram_range': [(1, 1), (1, 2)],
        'union__text__count_vectorizer__ngram_range': [(1, 1), (1, 2)]},
    {'model' : [Lasso(random_state = 44)],
        'model__alpha' : [0.1, 1, 10],
        'union__summary__count_vectorizer__ngram_range': [(1, 1), (1, 2)],
        'union__text__count_vectorizer__ngram_range': [(1, 1), (1, 2)]},
    {'model' : [ElasticNet(random_state = 44)],
        'model__l1_ratio' : [0.1, 0.5, 0.7, 0.9],          
        'model__alpha' : [0.1, 1, 10],
        'union__summary__count_vectorizer__ngram_range': [(1, 1), (1, 2)],
        'union__text__count_vectorizer__ngram_range': [(1, 1), (1, 2)]}
]

grid = GridSearchCV(pipeline, param_grid, cv = 5, scoring = 'neg_mean_squared_error', n_jobs = -1)
                                                        
grid.fit(train, train_score)

#Check the scores
scores = grid.grid_scores_

#Print the best score
print("The best score is %s" % grid.best_score_)
print("The best model parameters are: %s" % grid.best_params_)

#Make test set predictions
test_preds = grid.predict(test)

#Test score
print(mean_squared_error(test_score, test_preds))

#Save the test predictions as csv
test_preds_df = pd.DataFrame({
        "id": test_score.index,
        "score": test_preds
})
    
test_preds_df.to_csv('model/mega-pipeline.csv', index=False)
