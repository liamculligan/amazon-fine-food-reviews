#Predict the product review score

#Load required packages
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.linear_model import SGDRegressor
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import mean_squared_error

#Read in Id and Score for stratifying
reviews = pd.read_csv('Reviews.csv', index_col = 'Id', usecols = ['Id', 'Score'])
reviews_score = reviews['Score']

#Set number of rows to be read in at a time
chunksize = 500000

n_fold = 5

kf = StratifiedKFold(n_splits = n_fold, shuffle = True, random_state = 44)

train_indices = []
validation_indices = []

#Set training/validation indices
for train_index, validation_index in kf.split(reviews, reviews_score):
    train_indices.append(train_index)
    validation_indices.append(validation_index)

#Initialise empty fold_scores list
fold_scores = []

#Loop through alpha values
for alpha in [10**-10, 1]:
    
    print(alpha)
    
    #Loop through l1_ratio values
    for l1_ratio in [0, 0.2]:
        
        print(l1_ratio)
    
        #Loop through folds
        for fold in range(n_fold):
            
            print(fold)
            
            #Instantiate hashing vectorizers
            text_vectorizer = HashingVectorizer(ngram_range = (1, 3), n_features = 2**20)
            summary_vectorizer = HashingVectorizer(ngram_range = (1, 3), n_features = 2**20)
            
            #Instantiate scaler
            scaler = MaxAbsScaler()
            
            #Instantiate model
            lm = SGDRegressor(loss = 'squared_loss', n_iter = np.ceil(10**6 /train_indices[fold].shape[0]), alpha = alpha,
                                  l1_ratio = l1_ratio, random_state = 44, verbose = 3)
            
            #Loop through chunks for fitting
            for reviews in pd.read_csv('Reviews.csv', index_col = 'Id', usecols = ['Id', 'Summary', 'Text', 'Score'],
                                  chunksize = chunksize):
        
                #Only need training data
                train = reviews.iloc[reviews.index.isin(train_indices[fold])]
                
                #Drop duplicate score-text values
                train = train.drop_duplicates(subset = ['Summary', 'Text', 'Score'])
                
                #Remove line breaks
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
                
                #Summary dtm
                train_summary_dtm = summary_vectorizer.transform(train['Summary'])
                
                #Text dtm
                train_text_dtm = text_vectorizer.transform(train['Text'])
                
                #Non-negative matrix factorisation to identify topics in Text
                #TO DO
                
                #Remove text columns that have already been converted into numeric features
                train_features = train.drop(['Text', 'Summary'], axis = 'columns')
                
                #Convert features to sparse matrix
                train_features = csr_matrix(train_features.values)
                
                #Combine sparse matrices
                train = hstack([train_summary_dtm, train_text_dtm, train_features])
                
                #Scale
                scaler.partial_fit(train)
            
            #Loop through chunks for training
            for reviews in pd.read_csv('Reviews.csv', index_col = 'Id', usecols = ['Id', 'Summary', 'Text', 'Score'],
                                  chunksize = chunksize):
                
                #Only need training data
                train = reviews.iloc[reviews.index.isin(train_indices[fold])]
                
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
                
                #Summary dtm
                train_summary_dtm = summary_vectorizer.transform(train['Summary'])
                
                #Text dtm
                train_text_dtm = text_vectorizer.transform(train['Text'])
                
                #Non-negative matrix factorisation to identify topics in Text
                #TO DO
                
                #Remove text columns that have already been converted into numeric features
                train_features = train.drop(['Text', 'Summary'], axis = 'columns')
                
                #Convert features to sparse matrix
                train_features = csr_matrix(train_features.values)
                
                #Combine sparse matrices
                train = hstack([train_summary_dtm, train_text_dtm, train_features])
                
                #Scale
                train = scaler.transform(train)
                
                #Compute partial fit
                lm.partial_fit(train, train_score)
            
            #Create empty lists
            validation_pred = []
            validation_score = []
            
            #Loop through chunks for validation
            for reviews in pd.read_csv('Reviews.csv', index_col = 'Id', usecols = ['Id', 'Summary', 'Text', 'Score'],
                                  chunksize = chunksize):
            
                #Only need validation data
                validation = reviews.iloc[reviews.index.isin(validation_indices[fold])]
                
                #Create text - remove line breaks
                validation['Summary'] = validation['Summary'].str.replace('<br />', ' ')
                validation['Text'] = validation['Text'].str.replace('<br />', ' ')
                
                #Replace nan with ""
                validation['Summary'] = validation['Summary'].fillna(value = "")
                validation['Text'] = validation['Text'].fillna(value = "")
                
                #Extract target variable
                validation_score_chunk = validation['Score']
                
                #Remove score
                validation = validation.drop('Score', axis = 'columns')
                
                #Count the number of words
                validation['summary_count'] = validation['Summary'].str.split().apply(len)
                validation['text_count'] = validation['Text'].str.split().apply(len)
                validation = validation.assign(all_words_count = validation['summary_count'] + validation['text_count'])
                
                #Summary dtm - could do fit rather than partial_fit as the transformer is stateless
                validation_summary_dtm = summary_vectorizer.transform(validation['Summary'])
                
                #Text dtm
                validation_text_dtm = text_vectorizer.transform(validation['Text'])
                
                #Remove text columns that have already been converted into numeric features
                validation_features = validation.drop(['Text', 'Summary'], axis = 'columns')
                
                #Convert features to sparse matrices
                validation_features = csr_matrix(validation_features.values)
                
                #Combine sparse arrays
                validation = hstack([validation_summary_dtm, validation_text_dtm, validation_features])
                
                #Scale
                validation = scaler.transform(validation)
                
                #Predict
                validation_pred_chunk = lm.predict(validation)
                
                validation_pred.extend(validation_pred_chunk)
                validation_score.extend(list(validation_score_chunk))
            
            
            #Add validation score
            fold_scores.append(mean_squared_error(validation_score, validation_pred))
