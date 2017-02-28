#Predict the product review score
#XGB2

#Using td-idf rather than dtm

#Load required packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import NMF
from sklearn.pipeline import make_pipeline
import gc

#Read in csv
reviews = pd.read_csv('Reviews.csv', index_col = 'Id', usecols = ['Id', 'Summary', 'Text', 'Score'])

#Drop duplicate score-text values
reviews = reviews.drop_duplicates(subset = ['Summary', 'Text', 'Score'])

#Remove Text
reviews = reviews.drop('Text', axis = 'columns')

#Create text - drop duplicates, convert to lower case and remove line breaks
reviews['Summary'] = reviews['Summary'].str.lower().str.replace('<br />','')

#Replace nan with ""
reviews['Summary'] = reviews['Summary'].fillna(value = "")

#Extract target variable
score = reviews['Score']

#Remove score from reviews
reviews = reviews.drop('Score', axis = 'columns')

#Create train and test stratified w.r.t score
train, test, train_score, test_score = train_test_split(reviews, score, train_size = 0.5, stratify = score)

#dtm
vectorizer = TfidfVectorizer(min_df = 0.0005, max_df = 1.0, ngram_range = (1, 3), stop_words = 'english')
vectorizer.fit(train['Summary'])
train_text_dtm = vectorizer.transform(train['Summary'])
test_text_dtm = vectorizer.transform(test['Summary'])

train = train_text_dtm
test = test_text_dtm

"""

#Singular Value Decomposition
#max components == min(nrow -1, ncol - 1)
svd = TruncatedSVD(n_components = 850, random_state = 44)
svd.fit(train)

#The amount of variance that each PC explains
var = svd.explained_variance_ratio_
print(var)

#Cumulative Variance explains
var1 = np.cumsum(np.round(svd.explained_variance_ratio_, decimals=4)*100)
plt.plot(var1)

#From the plot we will select 850 principal components, which explain ... of the variance:
print(sum(var[0:850]))

#From the plot, select 
svd = TruncatedSVD(n_components = 850, random_state = 44)
svd.fit(train)
train = svd.transform(train)
test = svd.transform(test)

"""

#Convert to DMatrix
dtrain = xgb.DMatrix(train, label = train_score)
dtest = xgb.DMatrix(test, label = test_score)

del train
del test
del reviews
gc.collect()

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

#Order from best to worst score - xgb3 best score - 19/02/2017 - rmse - 1.0549

results = results.sort_values('score', ascending = False)

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
    
test_preds_df.to_csv('model/xgb3.csv', index=False)
