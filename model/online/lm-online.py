#Predict the product review score

#Load required packages
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.linear_model import PassiveAggressiveRegressor
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import mean_squared_error

#Read in Id and Score for stratifying
reviews = pd.read_csv('Reviews.csv', index_col = 'Id', usecols = ['Id', 'Score'])
reviews_score = reviews['Score']

#Create evaluation and test stratified w.r.t score
evaluation, test = train_test_split(reviews, train_size = 0.8, stratify = reviews_score, random_state = 44)

evaluation_score = evaluation["Score"]
test_score = test["Score"]

evaluation_indices = evaluation.index
test_indices = test.index

#Define number of folds for cross-validation
n_fold = 5

kf = StratifiedKFold(n_splits = n_fold, shuffle = True, random_state = 44)

train_indices = []
validation_indices = []

#Set training/validation indices
for train_index, validation_index in kf.split(evaluation, evaluation_score):
    train_indices.append(train_index)
    validation_indices.append(validation_index)

#Set number of rows to be read in at a time
chunksize = 100000

#Initialise empty fold_scores list
fold_scores = []

#Initialise empty test_scores list
test_scores = []

#Loop through alpha values
for C in [1]:
    
    print(C)
    
    #Loop through folds
    for fold in range(n_fold):
        
        print(fold)
        
        #Instantiate hashing vectorizers
        text_vectorizer = HashingVectorizer(ngram_range = (1, 3), n_features = 2**20)
        summary_vectorizer = HashingVectorizer(ngram_range = (1, 3), n_features = 2**20)
        
        #Instantiate scaler
        scaler = MaxAbsScaler()
        
        #Instantiate model
        lm = PassiveAggressiveRegressor(C = C, verbose = 3, random_state = 44)
        
        #Loop through chunks for fitting
        for reviews in pd.read_csv('Reviews.csv', index_col = 'Id', usecols = ['Id', 'Summary', 'Text', 'Score'],
                              chunksize = chunksize):
    
            #Only need training data
            train = reviews.iloc[reviews.index.isin(train_indices[fold])]
            
            #Continue to next iteration if there is no data in this chunk
            if train.shape[0] == 0:
                continue
                
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
            
            #Continue to next iteration if there is no data in this chunk
            if train.shape[0] == 0:
                continue
                
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
            
            #Continue to next iteration if there is no data in this chunk
            if validation.shape[0] == 0:
                continue
                
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
        
        
        #Evaluate and save validation score
        fold_scores.append(mean_squared_error(validation_score, validation_pred))

    
    ###Train on entire evaluation set
    
    #Instantiate hashing vectorizers
    text_vectorizer = HashingVectorizer(ngram_range = (1, 3), n_features = 2**20)
    summary_vectorizer = HashingVectorizer(ngram_range = (1, 3), n_features = 2**20)
    
    #Instantiate scaler
    scaler = MaxAbsScaler()
    
    #Instantiate model
    lm = PassiveAggressiveRegressor(C = C, verbose = 3, random_state = 44)
    
    #Loop through chunks for fitting
    for reviews in pd.read_csv('Reviews.csv', index_col = 'Id', usecols = ['Id', 'Summary', 'Text', 'Score'],
                          chunksize = chunksize):
        
        #Only need training data - FIX
        evaluation = reviews.iloc[reviews.index.isin(evaluation_indices)]
        
        #Continue to next iteration if there is no data in this chunk
        if evaluation.shape[0] == 0:
            continue
        
        #Drop duplicate score-text values
        evaluation = evaluation.drop_duplicates(subset = ['Summary', 'Text', 'Score'])
        
        #Remove line breaks
        evaluation['Summary'] = evaluation['Summary'].str.replace('<br />', ' ')
        evaluation['Text'] = evaluation['Text'].str.replace('<br />', ' ')
        
        #Replace nan with ""
        evaluation['Summary'] = evaluation['Summary'].fillna(value = "")
        evaluation['Text'] = evaluation['Text'].fillna(value = "")
        
        #Remove target variable
        evaluation = evaluation.drop('Score', axis = 'columns')
        
        #Count the number of words
        evaluation['summary_count'] = evaluation['Summary'].str.split().apply(len)
        evaluation['text_count'] = evaluation['Text'].str.split().apply(len)
        evaluation = evaluation.assign(all_words_count = evaluation['summary_count'] + evaluation['text_count'])
        
        #Summary dtm
        evaluation_summary_dtm = summary_vectorizer.transform(evaluation['Summary'])
        
        #Text dtm
        evaluation_text_dtm = text_vectorizer.transform(evaluation['Text'])
        
        #Remove text columns that have already been converted into numeric features
        evaluation_features = evaluation.drop(['Text', 'Summary'], axis = 'columns')
        
        #Convert features to sparse matrix
        evaluation_features = csr_matrix(evaluation_features.values)
        
        #Combine sparse matrices
        evaluation = hstack([evaluation_summary_dtm, evaluation_text_dtm, evaluation_features])
        
        #Scale
        scaler.partial_fit(evaluation)
    
    #Loop through chunks for training
    for reviews in pd.read_csv('Reviews.csv', index_col = 'Id', usecols = ['Id', 'Summary', 'Text', 'Score'],
                          chunksize = chunksize):
        
        #Only need training data
        evaluation = reviews.iloc[reviews.index.isin(evaluation_indices)]
        
        #Continue to next iteration if there is no data in this chunk
        if evaluation.shape[0] == 0:
            continue
            
        #Drop duplicate score-text values
        evaluation = evaluation.drop_duplicates(subset = ['Summary', 'Text', 'Score'])
        
        #Create text - remove line breaks
        evaluation['Summary'] = evaluation['Summary'].str.replace('<br />', ' ')
        evaluation['Text'] = evaluation['Text'].str.replace('<br />', ' ')
        
        #Replace nan with ""
        evaluation['Summary'] = evaluation['Summary'].fillna(value = "")
        evaluation['Text'] = evaluation['Text'].fillna(value = "")
        
        #Extract target variable
        evaluation_score = evaluation['Score']
        
        #Remove target variable
        evaluation = evaluation.drop('Score', axis = 'columns')
        
        #Count the number of words
        evaluation['summary_count'] = evaluation['Summary'].str.split().apply(len)
        evaluation['text_count'] = evaluation['Text'].str.split().apply(len)
        evaluation = evaluation.assign(all_words_count = evaluation['summary_count'] + evaluation['text_count'])
        
        #Summary dtm
        evaluation_summary_dtm = summary_vectorizer.transform(evaluation['Summary'])
        
        #Text dtm
        evaluation_text_dtm = text_vectorizer.transform(evaluation['Text'])
        
        #Non-negative matrix factorisation to identify topics in Text
        #TO DO
        
        #Remove text columns that have already been converted into numeric features
        evaluation_features = evaluation.drop(['Text', 'Summary'], axis = 'columns')
        
        #Convert features to sparse matrix
        evaluation_features = csr_matrix(evaluation_features.values)
        
        #Combine sparse matrices
        evaluation = hstack([evaluation_summary_dtm, evaluation_text_dtm, evaluation_features])
        
        #Scale
        evaluation = scaler.transform(evaluation)
        
        #Compute partial fit
        lm.partial_fit(evaluation, evaluation_score)
    
    #Create empty lists
    test_pred = []
    test_score = []
    
    #Loop through chunks for validation
    for reviews in pd.read_csv('Reviews.csv', index_col = 'Id', usecols = ['Id', 'Summary', 'Text', 'Score'],
                          chunksize = chunksize):
    
        #Only need validation data
        test = reviews.iloc[reviews.index.isin(test_indices)]
        
        #Continue to next iteration if there is no data in this chunk
        if test.shape[0] == 0:
            continue
            
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
        
        #Text dtm
        test_text_dtm = text_vectorizer.transform(test['Text'])
        
        #Remove text columns that have already been converted into numeric features
        test_features = test.drop(['Text', 'Summary'], axis = 'columns')
        
        #Convert features to sparse matrices
        test_features = csr_matrix(test_features.values)
        
        #Combine sparse arrays
        test = hstack([test_summary_dtm, test_text_dtm, test_features])
        
        #Scale
        test = scaler.transform(test)
        
        #Predict
        test_pred_chunk = lm.predict(test)
        
        test_pred.extend(test_pred_chunk)
        test_score.extend(list(test_score_chunk))
    
    #Evaluate and save test score
    test_scores.append(mean_squared_error(test_score, test_pred))
