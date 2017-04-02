#Predict the product review score

#Load required packages
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk import pos_tag
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import csc_matrix, hstack
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import NMF
from sklearn.pipeline import make_pipeline

#Read in csv
reviews = pd.read_csv('Reviews.csv', usecols = ['Id', 'Summary', 'Text', 'Score'])
summary_preds_train = pd.read_csv('lm_text_oos_preds.csv')
summary_preds_test = pd.read_csv('lm_text_test_preds.csv')


#Drop duplicate score-text values
reviews = reviews.drop_duplicates(subset = ['Summary', 'Score'])

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

#Merge
train = train.merge(summary_preds_train, how = 'left', on = 'Id')
test = test.merge(summary_preds_test, how = 'left', on = 'Id')

train = train.set_index('Id')
test = test.set_index('Id')

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

train_text_topics = csc_matrix(train_text_topics)
test_text_topics = csc_matrix(test_text_topics)

#Add main topic column name
names_text_topics = ['ohe_' + str(x) for x in list(ohe_enc.active_features_)]

#Convert to sparse arrays
train_text_arr = csc_matrix(train_text_arr)
test_text_arr = csc_matrix(test_text_arr)

#Remove text columns that have already been converted into numeric features
train_features = train.drop(['Text', 'Summary'], axis = 'columns')
test_features = test.drop(['Text', 'Summary'], axis = 'columns')

#Feature names
names_features = list(train_features.columns)

#Convert train_features and test_features to sparse matrices
train_features = csc_matrix(train_features.values)
test_features = csc_matrix(test_features.values)

#Combine sparse arrays
train = hstack([train_text_arr, train_text_topics, train_features])
test = hstack([test_text_arr, test_text_topics, test_features])

col_names = names_text_arr + names_text_topics + names_features

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
                            
                            n_rounds =  xgb_cv.shape[0]

                            score = xgb_cv['test-rmse-mean'].iloc[n_rounds - 1]
                            
                            #Save the standard deviation of the scoring metric for this set of parameters
                            st_dev = xgb_cv['test-rmse-std'].iloc[n_rounds - 1]

                            results = results.append({'eta' : eta, 'max_depth' : max_depth, \
                            'subsample' : subsample, 'colsample_bytree' : colsample_bytree, \
                            'gamma' : gamma, 'min_child_weight' : min_child_weight, 'alpha' : alpha, \
                            'score' : score, 'st_dev' : st_dev, 'n_rounds' : n_rounds}, \
                            ignore_index=True)
                            
#Remove initial missing row
results = results.dropna(axis = 'rows', how = 'all')

#Correct columns types
results[['max_depth', 'n_rounds']] = results[['max_depth', 'n_rounds']].astype(int)

#Order from best to worst score - xgb10 best score - 21/02/2017 - rmse - 0.882

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

#Feature importance

#Not working maybe with so many columns 

"""
import operator

def create_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    outfile.close()
    
create_feature_map(col_names)

importance = xgb_mod.get_fscore(fmap='xgb.fmap')

importance = sorted(importance.items(), key=operator.itemgetter(1))

df = pd.DataFrame(importance, columns=['feature', 'fscore'])
df['fscore'] = df['fscore'] / df['fscore'].sum()

featp = df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))
plt.title('XGBoost Feature Importance')
plt.xlabel('relative importance')
fig_featp = featp.get_figure()
fig_featp.savefig('feature_importance_xgb.png', bbox_inches='tight', pad_inches=1)

"""

#Make predictions on the test set
test_preds = xgb_mod.predict(dtest)

#Test score
print(np.sqrt(mean_squared_error(test_score, test_preds)))

#Save the test predictions as csv
test_preds_df = pd.DataFrame({
        "id": test_score.index,
        "score": test_preds
})
    
test_preds_df.to_csv('model/xgb10.csv', index=False)