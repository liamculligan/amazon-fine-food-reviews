#Predict the product review score

#Load required packages
import pandas as pd
import numpy as np
import spacy
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_predict
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import NMF
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline

#Read in csv
reviews = pd.read_csv('Reviews.csv', index_col = 'Id', usecols = ['Id', 'Summary', 'Score'])

#Drop duplicate score-text values
reviews = reviews.drop_duplicates(subset = ['Summary', 'Score'])

#Create text - remove line breaks
reviews['Summary'] = reviews['Summary'].str.replace('<br />', ' ')

#Replace nan with ""
reviews['Summary'] = reviews['Summary'].fillna(value = "")

#Extract target variable
score = reviews['Score']

#Remove score from reviews
reviews = reviews.drop('Score', axis = 'columns')


#Create train and test stratified w.r.t score
train, test, train_score, test_score = train_test_split(reviews, score, train_size = 0.5, stratify = score, \
                                                        random_state = 44)

#Use spacy's lemmatizer

# load spacy language model and save old tokenizer
en_nlp = spacy.load('en')
old_tokenizer = en_nlp.tokenizer

# create a custom tokenizer using the spacy document processing pipeline
# (now using our own tokenizer)
def custom_tokenizer(document):
    doc_spacy = en_nlp(document, entity=False, parse=False)
    return [token.lemma_ for token in doc_spacy]

#Create custom classes
class ItemSelector(BaseEstimator, TransformerMixin):
    
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]

pipeline = make_pipeline(ItemSelector(key = 'Summary'),
                         CountVectorizer(tokenizer = custom_tokenizer, min_df = 0.0005, max_df = 1.0, ngram_range = (1, 1)),
                         MaxAbsScaler(),
                         SGDRegressor(loss = 'squared_loss', n_iter = np.ceil(10**6 /train.shape[0]), random_state = 44, verbose = 3))

#Define the parameter values that should be searched
l1_ratio_range = [0, 1]
alpha_range = [0.00001, 0.01]

#Create a parameter grid: map the parameter names to the values that should be searched
param_grid = dict(sgdregressor__l1_ratio = l1_ratio_range, sgdregressor__alpha = alpha_range)
print(param_grid)

grid = GridSearchCV(pipeline, param_grid, cv = 5, scoring = 'neg_mean_squared_error', verbose = 3)

#Fit the models
grid.fit(train, train_score)

#Check the scores
scores = grid.cv_results_

#Print the best score - 07/03/2017 - 0.791 (rmse - 0.889)
print("The best score is %s" % grid.best_score_)
print("The best model parameters are: %s" % grid.best_params_)

#Out of fold predictions (for classification , method = 'predict_proba')
train_preds = cross_val_predict(grid.best_estimator_, train, train_score)

#Save the test predictions as csv
train_preds_df = pd.DataFrame({
        "Id": train_score.index,
        "score": train_preds
})

train_preds_df.to_csv('lm_text_oos_preds.csv', index = False)

#Make test set predictions
test_preds = grid.predict(test)

#Test score
print(np.sqrt(mean_squared_error(test_score, test_preds)))

#Save the test predictions as csv
test_preds_df = pd.DataFrame({
        "Id": test_score.index,
        "score": test_preds
})

test_preds_df.to_csv('lm_text_test_preds.csv', index=False)