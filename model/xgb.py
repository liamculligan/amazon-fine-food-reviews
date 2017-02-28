#Predict the product review score

#Stemming/lemmatisation

#Load required packages
import pandas as pd
import numpy as np
import nltk
import spacy
import re
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

#Word stem

#Create train and test stratified w.r.t score
train, test, train_score, test_score = train_test_split(reviews, score, train_size = 0.5, stratify = score)

# Technicality: we want to use the regexp-based tokenizer
# that is used by CountVectorizer and only use the lemmatization
# from spacy. To this end, we replace en_nlp.tokenizer (the spacy tokenizer)
# with the regexp-based tokenization.

# instantiate Porter stemmer
stemmer = nltk.stem.PorterStemmer()

# regexp used in CountVectorizer
regexp = re.compile('(?u)\\b\\w\\w+\\b')

# load spacy language model and save old tokenizer
en_nlp = spacy.load('en')
old_tokenizer = en_nlp.tokenizer

# replace the tokenizer with the preceding regexp tokenizer
en_nlp.tokenizer = lambda string: old_tokenizer.tokens_from_list(regexp.findall(string))

# create a custom tokenizer using the spacy document processing pipeline
# (now using our own tokenizer)
def porter_tokenizer(document):
    doc_spacy = en_nlp(document, entity=False, parse=False)
    return [stemmer.stem(token.norm_.lower()) for token in doc_spacy]
            
#dtm
vectorizer = CountVectorizer(tokenizer = porter_tokenizer, min_df = 0.0005, max_df = 1.0, ngram_range = (1, 3))
vectorizer.fit(train['Summary'])
train_summary_dtm = vectorizer.transform(train['Summary'])
test_summary_dtm = vectorizer.transform(test['Summary'])

vectorizer = CountVectorizer(tokenizer = porter_tokenizer, min_df = 0.001, max_df = 1.0, ngram_range = (1, 3))
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
vectorizer = TfidfVectorizer(tokenizer = porter_tokenizer, min_df = 5, ngram_range = (1, 1), stop_words = 'english')

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

#Convert to DMatrix
dtrain = xgb.DMatrix(train, label = train_score)
dtest = xgb.DMatrix(test, label = test_score)

results = pd.DataFrame({'eta' : np.nan, 'max_depth' : np.nan, 'subsample' : np.nan, 'colsample_bytree' : np.nan, \
                        'gamma' : np.nan, 'min_child_weight' : np.nan, 'alpha' : np.nan, \
                        'score' : np.nan, 'st_dev' : np.nan, 'n_rounds' : np.nan}, index = [0])

eta_values = [0.7] 
max_depth_values = [3, 6]
subsample_values = [0.5] 
colsample_bytree_values = [0.3]
gamma_values = [0]
min_child_weight_values = [1]
alpha_values = [0]

early_stopping_rounds = 5

for eta in eta_values:
    for max_depth in max_depth_values:
        for subsample in subsample_values:
            for colsample_bytree in colsample_bytree_values:
                for gamma in gamma_values:
                    for min_child_weight in min_child_weight_values:
                        for alpha in alpha_values:
                            
                            print(eta)
                            
                            param = {'objective':'reg:linear',
                            'booster':'gbtree',
                            'eval_metric':'rmse',
                            'eta': eta,
                            'max_depth': max_depth,
                            'subsample':subsample,
                            'colsample_bytree':colsample_bytree,
                            'gamma':gamma,
                            'min_child_weight':min_child_weight,
                            'alpha':alpha,
                            'seed' : 44}
                            
                            xgb_cv = xgb.cv(param,
                                   dtrain,
                                   num_boost_round = 10000,
                                   nfold = 5,
                                   stratified = True,
                                   early_stopping_rounds = early_stopping_rounds,
                                   verbose_eval = True,
                                   seed = 44)
                            
                            n_rounds =  xgb_cv.shape[0] - early_stopping_rounds

                            score = xgb_cv['test-rmse-mean'].iloc[n_rounds]
                            
                            #Save the standard deviation of the scoring metric for this set of parameters
                            st_dev = xgb_cv['test-rmse-std'].iloc[n_rounds]

                            results = results.append({'eta' : eta, 'max_depth' : max_depth, \
                            'subsample' : subsample, 'colsample_bytree' : colsample_bytree, \
                            'gamma' : gamma, 'min_child_weight' : min_child_weight, 'alpha' : alpha, \
                            'score' : score, 'st_dev' : st_dev, 'n_rounds' : n_rounds}, \
                            ignore_index=True)
                            
#Remove initial missing row
results = results.dropna(axis = 'rows', how = 'all')

#Correct columns types
results[['max_depth', 'n_rounds']] = results[['max_depth', 'n_rounds']].astype(int)

#Order from best to worst score - xgb7 best score - 20/02/2017 - rmse - 0.875

results = results.sort_values('score', ascending = True)

#Train model on the full training set
param = {'objective':'reg:linear',
         'booster':'gbtree',
         'eval_metric':'rmse',
         'eta': results['eta'].iloc[0],
         'max_depth': results['max_depth'].iloc[0],
         'colsample_bytree':results['colsample_bytree'].iloc[0],
         'gamma':results['gamma'].iloc[0],
         'min_child_weight':results['min_child_weight'].iloc[0],
         'alpha':results['alpha'].iloc[0],
         'seed' : 44
}

xgb_mod = xgb.train(param,
                    dtrain,
                    num_boost_round = n_rounds,
                    verbose_eval = True
                    )

#Make predictions on the test set
test_preds = xgb_mod.predict(dtest)

#Test score
print(np.sqrt(mean_squared_error(test_score, test_preds)))

#Save the test predictions as csv
test_preds_df = pd.DataFrame({
        "id": test_score.index,
        "score": test_preds
})
    
test_preds_df.to_csv('model/xgb5.csv', index=False)