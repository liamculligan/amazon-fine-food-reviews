#Predict the product review score

#Load required packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.linear_model import SGDRegressor
import numpy as np
from scipy.sparse import coo_matrix, hstack
from sklearn.metrics import mean_squared_error

#Read in Id and Score for stratifying
reviews_score = pd.read_csv('Reviews.csv', index_col = 'Id', usecols = ['Id', 'Score'])

#Split train and test - only for target variable
train_score_indices, test_score_indices = train_test_split(reviews_score, train_size = 0.8,
                                                           stratify = reviews_score['Score'], random_state = 44)

#Instantiate hashing vectorizers
text_vectorizer = HashingVectorizer(ngram_range = (1, 3), n_features = 2**20)
summary_vectorizer = HashingVectorizer(ngram_range = (1, 3), n_features = 2**20)

#Instantiate scaler
scaler = MaxAbsScaler()

#Instantiate model
lm = SGDRegressor(loss = 'squared_loss', n_iter = np.ceil(10**6 /train_score_indices.shape[0]), alpha = 1,
                      l1_ratio = 0, random_state = 44, verbose = 3)

#Set number of rows to be read in at a time
chunksize = 50000

#Loop through chunks for fitting
for reviews in pd.read_csv('Reviews.csv', index_col = 'Id', usecols = ['Id', 'Summary', 'Text', 'Score'],
                      chunksize = chunksize):
   
    #Only need training data
    train = reviews.iloc[reviews.index.isin(train_score_indices.index)]
    
    #Drop duplicate score-text values
    train = train.drop_duplicates(subset = ['Summary', 'Text', 'Score'])
    
    #Create text - remove line breaks
    train['Summary'] = train['Summary'].str.replace('<br />', ' ')
    train['Text'] = train['Text'].str.replace('<br />', ' ')
    
    #Replace nan with ""
    train['Summary'] = train['Summary'].fillna(value = "")
    train['Text'] = train['Text'].fillna(value = "")
    
    #Remove target variable
    train = train.drop('Score', axis = 'columns')
    
    #Count the number of words
    train['summary_count'] = train['Summary'].str.split().apply(len)
    train['text_count'] = train['Text'].str.split().apply(len)
    train = train.assign(all_words_count = train['summary_count'] + train['text_count'])
    
    #Summary dtm - could do fit rather than partial_fit as the transformer is stateless
    summary_vectorizer.partial_fit(train['Summary'])
    train_summary_dtm = summary_vectorizer.transform(train['Summary'])
    train_summary_dtm = coo_matrix(train_summary_dtm)
    
    #Text dtm
    text_vectorizer.partial_fit(train['Text'])
    train_text_dtm = text_vectorizer.transform(train['Text'])
    train_text_dtm = coo_matrix(train_text_dtm)
    
    #Non-negative matrix factorisation to identify topics in Text
    #TO DO
    
    #Remove text columns that have already been converted into numeric features
    train_features = train.drop(['Text', 'Summary'], axis = 'columns')
    
    #Convert features to sparse matrices
    train_features = coo_matrix(train_features.values)
    
    #Combine sparse arrays
    train = hstack([train_summary_dtm, train_text_dtm, train_features])
    
    #Scale
    scaler.partial_fit(train)

#Loop through chunks for training
for reviews in pd.read_csv('Reviews.csv', index_col = 'Id', usecols = ['Id', 'Summary', 'Text', 'Score'],
                      chunksize = chunksize):
   
    #Only need training data
    train = reviews.iloc[reviews.index.isin(train_score_indices.index)]
    
    #Drop duplicate score-text values
    train = train.drop_duplicates(subset = ['Summary', 'Text', 'Score'])
    
    #Create text - remove line breaks
    train['Summary'] = train['Summary'].str.replace('<br />', ' ')
    train['Text'] = train['Text'].str.replace('<br />', ' ')
    
    #Replace nan with ""
    train['Summary'] = train['Summary'].fillna(value = "")
    train['Text'] = train['Text'].fillna(value = "")
    
    #Extract target variable
    train_score = train['Score']
    
    #Remove target variable
    train = train.drop('Score', axis = 'columns')
    
    #Count the number of words
    train['summary_count'] = train['Summary'].str.split().apply(len)
    train['text_count'] = train['Text'].str.split().apply(len)
    train = train.assign(all_words_count = train['summary_count'] + train['text_count'])
    
    #Summary dtm - could do fit rather than partial_fit as the transformer is stateless
    train_summary_dtm = summary_vectorizer.transform(train['Summary'])
    train_summary_dtm = coo_matrix(train_summary_dtm)
    
    #Text dtm
    train_text_dtm = text_vectorizer.transform(train['Text'])
    train_text_dtm = coo_matrix(train_text_dtm)
    
    #Non-negative matrix factorisation to identify topics in Text
    #TO DO
    
    #Remove text columns that have already been converted into numeric features
    train_features = train.drop(['Text', 'Summary'], axis = 'columns')
    
    #Convert features to sparse matrices
    train_features = coo_matrix(train_features.values)
    
    #Combine sparse arrays
    train = hstack([train_summary_dtm, train_text_dtm, train_features])
    
    train = scaler.transform(train)
    
    #Compute partial fit
    lm.partial_fit(train, train_score)

#Create empty lists
test_pred = []
test_score = []

#Loop through chunks for testing
for reviews in pd.read_csv('Reviews.csv', index_col = 'Id', usecols = ['Id', 'Summary', 'Text', 'Score'],
                      chunksize = chunksize):

    #Only need testing data
    test = reviews.iloc[reviews.index.isin(train_score_indices.index) == False]
    
    #Create text - remove line breaks
    test['Summary'] = test['Summary'].str.replace('<br />', ' ')
    test['Text'] = test['Text'].str.replace('<br />', ' ')
    
    #Replace nan with ""
    test['Summary'] = test['Summary'].fillna(value = "")
    test['Text'] = test['Text'].fillna(value = "")
    
    #Extract target variable
    test_score_chunk = test['Score']
    
    #Remove score
    test = test.drop('Score', axis = 'columns')
    
    #Count the number of words
    test['summary_count'] = test['Summary'].str.split().apply(len)
    test['text_count'] = test['Text'].str.split().apply(len)
    test = test.assign(all_words_count = test['summary_count'] + test['text_count'])
    
    #Summary dtm - could do fit rather than partial_fit as the transformer is stateless
    test_summary_dtm = summary_vectorizer.transform(test['Summary'])
    test_summary_dtm = coo_matrix(test_summary_dtm)
    
    #Text dtm
    test_text_dtm = text_vectorizer.transform(test['Text'])
    test_text_dtm = coo_matrix(test_text_dtm)
    
    #Remove text columns that have already been converted into numeric features
    test_features = test.drop(['Text', 'Summary'], axis = 'columns')
    
    #Convert features to sparse matrices
    test_features = coo_matrix(test_features.values)
    
    #Combine sparse arrays
    test = hstack([test_summary_dtm, test_text_dtm, test_features])
    
    #Scale
    test = scaler.transform(test)
    
    #Predict
    test_pred_chunk = lm.predict(test)
    
    test_pred.extend(test_pred_chunk)
    test_score.extend(list(test_score_chunk))


#Test score
print(mean_squared_error(test_score, test_pred))
