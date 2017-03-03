#Predict the product review score

#Load required packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import csr_matrix, hstack
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import NMF
from sklearn.pipeline import make_pipeline

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

#Create train and test stratified w.r.t score
train, test, train_score, test_score = train_test_split(reviews, score, train_size = 0.5, stratify = score)

#dtm
vectorizer = CountVectorizer(min_df = 0.0005, max_df = 1.0, ngram_range = (1, 3))
vectorizer.fit(train['Summary'])
train_summary_dtm = vectorizer.transform(train['Summary'])
test_summary_dtm = vectorizer.transform(test['Summary'])

vectorizer = CountVectorizer(min_df = 0.001, max_df = 1.0, ngram_range = (1, 3))
vectorizer.fit(train['Text'])
train_text_dtm = vectorizer.transform(train['Text'])
test_text_dtm = vectorizer.transform(test['Text'])

#Non-negative matrix factorisation to identify topics in Text
#Remove punctuation
train_text = train['Text'].str.replace('[^\w\s]','')
test_text = test['Text'].str.replace('[^\w\s]','')

#Convert to arrays
train_text_arr = train_text.values
test_text_arr = test_text.values

#Instantiate Tfidf
vectorizer = TfidfVectorizer(min_df = 5, ngram_range = (1, 1), stop_words = 'english')

#Instantiate NMF
nmf = NMF(n_components = 6, random_state = 44)

#Pipeline
pipeline = make_pipeline(vectorizer, nmf)
train_text_arr = pipeline.fit_transform(train_text_arr)
test_text_arr = pipeline.transform(test_text_arr)

#Determine top words for each topic
def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()

#Call function
print_top_words(nmf, vectorizer.get_feature_names(), 5)

#Find most likely topic for each observation
train_text_topics = train_text_arr.argmax(axis=1)
test_text_topics = test_text_arr.argmax(axis=1)

#One hot encode topics
ohe_enc = OneHotEncoder()
train_text_topics = ohe_enc.fit_transform(train_text_topics.reshape(-1, 1))
test_text_topics = ohe_enc.transform(test_text_topics.reshape(-1, 1))

#Convert to sparse arrays
train_text_arr = csr_matrix(train_text_arr)
test_text_arr = csr_matrix(test_text_arr)

#Combine sparse arrays
train = hstack([train_summary_dtm, train_text_dtm, train_text_arr, train_text_topics])
test = hstack([test_summary_dtm, test_text_dtm, test_text_arr, test_text_topics])

#Create a parameter grid: map the parameter names to the values that should be searched
param_grid = dict(reg_alpha = [0], colsample_bytree = [0.3], learning_rate = [0.7], gamma = [0], max_depth = [3], \
                  min_child_weight = [1], n_estimators = [175], subsample = [0.5])

#Instantiate the grid
xgb_mod = xgb.XGBRegressor(objective = 'reg:linear', seed = 44, silent = False)
print(xgb_mod)
grid = GridSearchCV(xgb_mod, param_grid, cv = 5, scoring = 'neg_mean_squared_error', verbose = 3)

#Fit the models
grid.fit(train, train_score)

#Check the scores
scores = grid.cv_results_

#Print the best score - 19/02/2017 - 1.173 (rmse - 1.083)
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
    
test_preds_df.to_csv('model/xgb-sk.csv', index=False)
