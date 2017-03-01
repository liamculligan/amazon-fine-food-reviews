#Predict the product review score

#Load required packages
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MaxAbsScaler
from scipy.sparse import coo_matrix, hstack, vstack
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib

#Read in train indeces
train_score_indices = joblib.load('train_score_indices.pkl')

#Determine number of rows
nrows = pd.read_csv('Reviews.csv', usecols = ['Id']).shape[0]

#Initiliase chunksize
chunksize = 50000

#Initialise number of rows read in
chunks_read = 0

#Loop through batches
for reviews in pd.read_csv('Reviews.csv', index_col = 'Id', usecols = ['Id', 'Summary', 'Text', 'Score'],
                      chunksize = chunksize):
    
    #Create text - remove line breaks
    reviews['Summary'] = reviews['Summary'].str.replace('<br />', ' ')
    reviews['Text'] = reviews['Text'].str.replace('<br />', ' ')
    
    #Replace nan with ""
    reviews['Summary'] = reviews['Summary'].fillna(value = "")
    reviews['Text'] = reviews['Text'].fillna(value = "")
    
    #Count the number of words
    reviews['summary_count'] = reviews['Summary'].str.split().apply(len)
    reviews['text_count'] = reviews['Text'].str.split().apply(len)
    reviews = reviews.assign(all_words_count = reviews['summary_count'] + reviews['text_count'])
    
    #Split the chunks into train and test
    train_chunk = reviews.iloc[reviews.index.isin(train_score_indices.index)]
    test_chunk = reviews.iloc[reviews.index.isin(train_score_indices.index) == False]
    
    #Summary dtm
    summary_vectorizer = joblib.load('summary_vectorizer.pkl')
    train_summary_dtm = summary_vectorizer.transform(train_chunk['Summary'])
    test_summary_dtm = summary_vectorizer.transform(test_chunk['Summary'])

    #Text dtm
    text_vectorizer = joblib.load('text_vectorizer.pkl')
    train_text_dtm = text_vectorizer.transform(train_chunk['Text'])
    test_text_dtm = text_vectorizer.transform(test_chunk['Text'])
    
    #Non-negative matrix factorisation to identify topics in Text
    #Remove punctuation
    train_chunk_text = train_chunk['Text'].str.replace('[^\w\s]','')
    test_chunk_text = test_chunk['Text'].str.replace('[^\w\s]','')
    
    #Read in pickled pipeline
    nmf_pipeline = joblib.load('nmf_pipeline.pkl')
    
    #Transform using pipeline
    train_chunk_text_nmf = nmf_pipeline.transform(train_chunk_text)
    test_chunk_text_nmf = nmf_pipeline.transform(test_chunk_text)
     
    #Find most likely topic for each observation
    train_chunk_text_topics = train_chunk_text_nmf.argmax(axis=1)
    test_chunk_text_topics = test_chunk_text_nmf.argmax(axis=1)
    
    #Convert to sparse matrices
    train_chunk_text_nmf = coo_matrix(train_chunk_text_nmf)
    test_chunk_text_nmf = coo_matrix(test_chunk_text_nmf)
    
    #Read in pickled one hot encoder
    topic_ohe = joblib.load('topic_ohe.pkl')
    
    #Transform using OHE
    train_chunk_text_topics = topic_ohe.transform(train_chunk_text_topics.reshape(-1, 1))
    test_chunk_text_topics = topic_ohe.transform(test_chunk_text_topics.reshape(-1, 1))

    #Extract target variable
    train_chunk_score = train_chunk['Score']
    test_chunk_score = test_chunk['Score']

    #Remove text columns that have already been converted into numeric features
    train_features = train_chunk.drop(['Text', 'Summary', 'Score'], axis = 'columns')
    test_features = test_chunk.drop(['Text', 'Summary', 'Score'], axis = 'columns')
    
    #Convert train_features and test_features to sparse matrices
    train_features = coo_matrix(train_features.values)
    test_features = coo_matrix(test_features.values)

    #Combine sparse matrices
    train_chunk = hstack([train_summary_dtm, train_text_dtm, train_chunk_text_nmf, train_chunk_text_topics, \
                          train_features])
    test_chunk = hstack([test_summary_dtm, test_text_dtm, test_chunk_text_nmf, test_chunk_text_topics, \
                         test_features])
     
    if chunks_read == 0:
        #Create universal train and test
        train = train_chunk
        test = test_chunk
        
        train_score = train_chunk_score
        test_score = test_chunk_score
        
    else:
        #Combine with universal train and test
        train = vstack([train, train_chunk])
        test = vstack([test, test_chunk])
        
        train_score = train_score.append(train_chunk_score)
        test_score = test_score.append(test_chunk_score)
    
    chunks_read += 1
    
    print(chunks_read)

#Standardise Data
scaler = MaxAbsScaler()
train = scaler.fit_transform(train)
test = scaler.transform(test)

#Define the parameter values that should be searched
l1_ratio_range = np.arange(0, 1.2, 0.2)
alpha_range = 10.0**-np.arange(0,7)

#Create a parameter grid: map the parameter names to the values that should be searched
param_grid = dict(l1_ratio = l1_ratio_range, alpha = alpha_range)
print(param_grid)

#Instantiate the grid
lm = SGDRegressor(loss = 'squared_loss', n_iter = np.ceil(10**6 /train.shape[0]), random_state = 44, verbose = 3)
print(lm)

grid = GridSearchCV(lm, param_grid, cv = 5, scoring = 'neg_mean_squared_error')

#Fit the models
grid.fit(train, train_score)

#Check the scores
scores = grid.cv_results_

#Print the best score - 22/02/2017 - 0.789 (rmse - 0.888)
print("The best score is %s" % grid.best_score_)
print("The best model parameters are: %s" % grid.best_params_)

#Make test set predictions
test_preds = grid.predict(test)

#Test score
print(np.sqrt(mean_squared_error(test_score, test_preds)))

#Save the test predictions as csv
test_preds_df = pd.DataFrame({
        "id": test_score.index,
        "score": test_preds
})
    
test_preds_df.to_csv('model/chunk-model.csv', index=False)