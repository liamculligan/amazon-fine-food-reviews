#Predict the product review score

#Load required packages
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import SGDRegressor

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

train = reviews
train_score = score

#Order train_score and test_score by index
train = train.sort_index()
train_score = train_score.sort_index()

pipeline = Pipeline([
    ('union', FeatureUnion([
        
        ('summary', Pipeline([
            ('selector', ItemSelector(key = 'Summary')),
            ('hashing_vectorizer', HashingVectorizer(ngram_range = (1, 3), n_features = 2**20)),
        ])),
        
        ('text', Pipeline([
            ('selector', ItemSelector(key = 'Text')),
            ('hashing_vectorizer', HashingVectorizer(ngram_range = (1, 3), n_features = 2**20)),
        ])),
        
        #Select all numerical features
        ('numerical', Pipeline([
            ('select', SelectTransform(lambda X: X.select_dtypes(exclude=['object']))),
        ])),
    ])),
    
    ('scaler',  MaxAbsScaler()),

    #Only instatiating a model
    ('model', SGDRegressor()),
])

#Parameters to be searched
param_grid = [{'model': [SGDRegressor(loss = 'squared_loss', n_iter = np.ceil(10**6 /train.shape[0]), \
                                       random_state = 44, verbose = 3)],
               'model__l1_ratio': [0],
               'model__alpha': [10**-10]}
              ]

#Instantiate the grid
grid = GridSearchCV(pipeline, param_grid, cv = 5, scoring = 'neg_mean_squared_error')

#Fit the models
grid.fit(train, train_score)

#Check the scores
scores = grid.cv_results_

#Print the best score
print("The best score is %s" % grid.best_score_)
print("The best model parameters are: %s" % grid.best_params_)
