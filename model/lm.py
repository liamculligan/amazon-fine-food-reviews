#Predict the product review score

#Load required packages
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk import pos_tag
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import csr_matrix, hstack
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
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

#Count the number of words
reviews['summary_count'] = reviews['Summary'].str.split().apply(len)
reviews['text_count'] = reviews['Text'].str.split().apply(len)
reviews = reviews.assign(all_words_count = reviews['summary_count'] + reviews['text_count'])

#Create train and test stratified w.r.t score
train, test, train_score, test_score = train_test_split(reviews, score, train_size = 0.5, stratify = score, \
                                                        random_state = 44)

#Order train_score and test_score by index
train = train.sort_index()
train_score = train_score.sort_index()
test = test.sort_index()
test_score = test_score.sort_index()

#Parts of Speech Tagging
def pos_count(series):
    
    """Input a pandas series to convert it into a count for each part of speech."""
    
    rows = list(series)
    
    stop_words = set(stopwords.words("english"))
    
    #Tokenizer does not include punctuation
    tokenizer = RegexpTokenizer(r'\w+')
    
    row_pos = []
    
    for row in rows:
        row_tokenized = tokenizer.tokenize(row)
        row_words = []
        for word in row_tokenized:
            word = word.lower()
            if word not in stop_words:
                #Specifically converting to lower - all words are capitalised in headline
                row_words.append(word.lower())
        word_pos = pos_tag(row_words)
        pos = []
        for tag in word_pos:
            pos.append(tag[1])
        row_pos.append(pos)
        
    #Convert list to a pandas DataFrame - each word in a separate column
    row_pos = pd.DataFrame(row_pos)
    row_pos.index = series.index
    
    #Stack to create multiindexed Series by row and column (long df)
    #Dummy this series but still multiindexed by row and column
    #Sum this by row to get the count of each pos for each row
    pos_count = pd.get_dummies(row_pos.stack()).sum(level = 0)
    
    #Add series to to each column name
    pos_count = pos_count.add_prefix(series.name + '_')
    
    return(pos_count)

def pos_concat(train, test, col_name):
    
    """Add pos_counts to train and test ensuring that same columns exist in each."""
    
    train_pos = pos_count(train[col_name])
    test_pos = pos_count(test[col_name])
    
    train_pos_cols = list(train_pos.columns)
    test_pos_cols = list(test_pos.columns)
    
    #Remove any columns in test not in train
    test_cols_drop = set(test_pos_cols) - set(train_pos_cols)
    test_pos = test_pos.drop(test_cols_drop, axis = 'columns')
        
    #Add any columns in train not in test
    test_cols_add = set(train_pos_cols) - set(test_pos_cols)
    for col in test_cols_add:
        test_pos[col] = 0
    
    #Add counts of pos to train and test
    train = pd.concat([train, train_pos], axis = 'columns')
    test = pd.concat([test, test_pos], axis = 'columns')
    
    #Some rows have no pos - these become nan - replace with 0
    train[train_pos_cols] = train[train_pos_cols].fillna(value = 0)
    test[train_pos_cols] = test[train_pos_cols].fillna(value = 0)
     
    return(train, test)

#Call function
train, test = pos_concat(train, test, 'Summary')

#Reorder DataFrames and Series by index
train = train.sort_index()
train_score = train_score.sort_index()
test = test.sort_index()
test_score = test_score.sort_index()

#Summary dtm
vectorizer = CountVectorizer(min_df = 0.0005, max_df = 1.0, ngram_range = (1, 3))
vectorizer.fit(train['Summary'])
train_summary_dtm = vectorizer.transform(train['Summary'])
test_summary_dtm = vectorizer.transform(test['Summary'])

#Summary dtm column names
names_summary_dtm = vectorizer.get_feature_names()

#Text dtm
vectorizer = CountVectorizer(min_df = 0.001, max_df = 1.0, ngram_range = (1, 3))
vectorizer.fit(train['Text'])
train_text_dtm = vectorizer.transform(train['Text'])
test_text_dtm = vectorizer.transform(test['Text'])

#Text dtm column names
names_text_dtm = vectorizer.get_feature_names()

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

#NMF pipeline
pipeline = make_pipeline(vectorizer, nmf)
train_text_arr = pipeline.fit_transform(train_text_arr)
test_text_arr = pipeline.transform(test_text_arr)

#NMF pipeline column names
names_text_arr = ['nmf_0', 'nmf_1', 'nmf_2', 'nmf_3', 'nmf_4', 'nmf_5']

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

#Add main topic column name
names_text_topics = ['ohe_' + str(x) for x in list(ohe_enc.active_features_)]

#Convert to sparse arrays
train_text_arr = csr_matrix(train_text_arr)
test_text_arr = csr_matrix(test_text_arr)

#Remove text columns that have already been converted into numeric features
train_features = train.drop(['Text', 'Summary'], axis = 'columns')
test_features = test.drop(['Text', 'Summary'], axis = 'columns')

#Feature names
names_features = list(train_features.columns)

#Convert train_features and test_features to sparse matrices
train_features = csr_matrix(train_features.values)
test_features = csr_matrix(test_features.values)

#Combine sparse arrays
train = hstack([train_summary_dtm, train_text_dtm, train_text_arr, train_text_topics, train_features])
test = hstack([test_summary_dtm, test_text_dtm, test_text_arr, test_text_topics, test_features])

col_names = names_summary_dtm + names_text_dtm + names_text_arr + names_text_topics + names_features

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

#Print the best score - 22/02/2017 - 0.809 (rmse - 0.890)
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
    
test_preds_df.to_csv('model/lm3.csv', index=False)