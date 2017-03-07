#Predict the product review score

#Load required packages
import pandas as pd
import numpy as np
import re
import spacy
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import csr_matrix, hstack
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
reviews['summary_word_count'] = reviews['Summary'].str.split().apply(len)
reviews['text_word_count'] = reviews['Text'].str.split().apply(len)
reviews = reviews.assign(all_word_count = reviews['summary_word_count'] + reviews['text_word_count'])

#Count the number of letters
reviews['summary_letter_count'] = reviews['Summary'].apply(lambda x: len(re.findall("[aA-zZ]", x)))
reviews['text_letter_count'] = reviews['Text'].apply(lambda x: len(re.findall("[aA-zZ]", x)))
reviews = reviews.assign(all_capital_char_count = reviews['summary_letter_count'] + 
                         reviews['text_letter_count'])

#Count the number of capital letters
reviews['summary_capital_letter_count'] = reviews['Summary'].apply(lambda x: len(re.findall("[A-Z]", x)))
reviews['text_capital_letter_count'] = reviews['Text'].apply(lambda x: len(re.findall("[A-Z]", x)))
reviews = reviews.assign(all_capital_char_count = reviews['summary_capital_letter_count'] + 
                         reviews['text_capital_letter_count'])

#Count the ratio of capital letters to all letters
reviews = reviews.assign(summary_capital_char_ratio = reviews['summary_capital_letter_count'] / 
                         reviews['summary_letter_count'])
reviews = reviews.assign(text_capital_char_ratio = reviews['text_capital_letter_count'] / 
                         reviews['text_letter_count'])

#Set nan to 0
reviews['summary_capital_char_ratio'] = reviews['summary_capital_char_ratio'].fillna(value = 0)
reviews['text_capital_char_ratio'] = reviews['text_capital_char_ratio'].fillna(value = 0)

#Count the number of white spaces
reviews['summary_whitespace_count'] = reviews['Summary'].apply(lambda x: x.count(" "))
reviews['text_whitespace_count'] = reviews['Text'].apply(lambda x: x.count(" "))
reviews = reviews.assign(all_whitespace_count = reviews['summary_whitespace_count'] +
                         reviews['text_whitespace_count'])

#Count the average of characters per word
reviews['summary_chars_per_word'] = reviews['Summary'].apply(lambda x: len(x) / (x.count(" ") + 1))
reviews['text_chars_per_word'] = reviews['Text'].apply(lambda x: len(x) / (x.count(" ") + 1))

#Count the average number of words per sentence (rough approximation)
reviews['summary_words_per_sentence'] = reviews['Summary'].apply(lambda x: x.count(" ") / (x.count(".") + 1))
reviews['text_words_per_sentence'] = reviews['Text'].apply(lambda x: x.count(" ") / (x.count(".") + 1))

reviews['summary_words_per_sentence'] = reviews['summary_words_per_sentence'].fillna(value = 0)
reviews['text_words_per_sentence'] = reviews['text_words_per_sentence'].fillna(value = 0)

#Count the number of digits
reviews['summary_digit_count'] = reviews['Summary'].apply(lambda x: len(re.findall("\d", x)))
reviews['text_digit_count'] = reviews['Text'].apply(lambda x: len(re.findall("\d", x)))
reviews = reviews.assign(all_digit_count = reviews['summary_digit_count'] + reviews['text_digit_count'])

#Create train and test stratified w.r.t score
train, test, train_score, test_score = train_test_split(reviews, score, train_size = 0.5, stratify = score, \
                                                        random_state = 44)

#Order train_score and test_score by index
train = train.sort_index()
train_score = train_score.sort_index()
test = test.sort_index()
test_score = test_score.sort_index()

#Load spacy language model
nlp = spacy.load('en')

#Parts of Speech Tagging
def pos_count(series, to_lower = False):
    
    """Input a pandas series to convert it into a count for each part of speech."""
    
    rows = list(series)
    
    #Create an empty list
    row_pos = []
    
    #For each row in the pandas series
    for row in rows:
        
        #Convert to lowercase if the to_lower argument is set to True
        if to_lower == True:
            row = row.lower()
        
        #Apply the english language model to the current row
        row_parsed = nlp(row)
        
        #For each word that isn't a stop word in the parsed document, add its part of speech to a list
        pos = [word.pos_ for word in row_parsed if not word in spacy.en.language_data.STOP_WORDS]
        
        #Add this row's list to the overall list
        row_pos.append(pos)
        
    #Convert list to a pandas DataFrame - each part of speech in a separate column
    row_pos = pd.DataFrame(row_pos)
    row_pos.index = series.index
    
    #Stack to create multiindexed Series by row and column (long df)
    #Dummy this series but still multiindexed by row and column
    #Sum this by row to get the count of each pos for each row
    pos_count = pd.get_dummies(row_pos.stack()).sum(level = 0)
    
    #Add series name to each new part of speech column name
    pos_count = pos_count.add_prefix(series.name + '_')
    
    return(pos_count)

#Entity Recognition
def entity_count(series, to_lower = False):
    
    """Input a pandas series to convert it into a count for each part of speech."""
    
    rows = list(series)
    
    #Create an empty list
    row_entities = []
    
    #For each row in the pandas series
    for row in rows:
        
        #Convert to lowercase if the to_lower argument is set to True
        if to_lower == True:
            row = row.lower()
        
        #Apply the english language model to the current row
        row_parsed = nlp(row)
        
        #For each entity, add its type to a list
        entities = [ent.label_ for ent in row_parsed.ents]
        
        #Add this row's list to the overall list
        row_entities.append(entities)
        
    #Convert list to a pandas DataFrame - each part of speech in a separate column
    row_entities = pd.DataFrame(row_entities)
    row_entities.index = series.index
    
    #Stack to create multiindexed Series by row and column (long df)
    #Dummy this series but still multiindexed by row and column
    #Sum this by row to get the count of each pos for each row
    entity_count = pd.get_dummies(row_entities.stack()).sum(level = 0)
    
    #Add series name to each new part of speech column name
    entity_count = entity_count.add_prefix(series.name + '_')
    
    return(entity_count)

def counts_concat(train, test, col_name, count_function):
    
    """
    Add results of count function (pos or entities) to train and test ensuring that same 
    columns exist in each.
    """
    
    train_counts = count_function(train[col_name])
    test_counts = count_function(test[col_name])
    
    train_count_cols = list(train_counts.columns)
    test_count_cols = list(test_counts.columns)
    
    #Remove any columns in test not in train
    test_cols_drop = set(test_count_cols) - set(train_count_cols)
    test_counts = test_counts.drop(test_cols_drop, axis = 'columns')
        
    #Add any columns in train not in test
    test_cols_add = set(train_count_cols) - set(test_count_cols)
    for col in test_cols_add:
        test_counts[col] = 0
    
    #Add counts of pos to train and test
    train = pd.concat([train, train_counts], axis = 'columns')
    test = pd.concat([test, test_counts], axis = 'columns')
    
    #Some rows have no pos - these become nan - replace with 0
    train[train_count_cols] = train[train_count_cols].fillna(value = 0)
    test[test_count_cols] = test[test_count_cols].fillna(value = 0)
     
    return(train, test)

#Call functions
train, test = counts_concat(train, test, 'Summary', pos_count)
train, test = counts_concat(train, test, 'Summary', entity_count)

#Reorder DataFrames and Series by index
train = train.sort_index()
train_score = train_score.sort_index()
test = test.sort_index()
test_score = test_score.sort_index()

#Use spacy's lemmatizer

# load spacy language model and save old tokenizer
en_nlp = spacy.load('en')
old_tokenizer = en_nlp.tokenizer

# create a custom tokenizer using the spacy document processing pipeline
# (now using our own tokenizer)
def custom_tokenizer(document):
    doc_spacy = en_nlp(document, entity=False, parse=False)
    return [token.lemma_ for token in doc_spacy]

# define a count vectorizer with the custom tokenizer
#Summary dtm
vectorizer = CountVectorizer(tokenizer = custom_tokenizer, min_df = 0.0005, max_df = 1.0, ngram_range = (1, 3))
vectorizer.fit(train['Summary'])
train_summary_dtm = vectorizer.transform(train['Summary'])
test_summary_dtm = vectorizer.transform(test['Summary'])

#Summary dtm column names
names_summary_dtm = vectorizer.get_feature_names()

#Text dtm
vectorizer = CountVectorizer(tokenizer = custom_tokenizer, min_df = 0.001, max_df = 1.0, ngram_range = (1, 3))
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

#Print the best score - 07/03/2017 - 0.791 (rmse - 0.889)
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
    
test_preds_df.to_csv('model/lm.csv', index=False)