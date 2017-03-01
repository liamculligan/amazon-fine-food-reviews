#Predict the product review score

#Load required packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import NMF
from sklearn.pipeline import make_pipeline

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
vectorizer = CountVectorizer(min_df = 0.0005, max_df = 1.0, ngram_range = (1, 3), stop_words = 'english')
vectorizer.fit(train['Summary'])
train_text_dtm = vectorizer.transform(train['Summary'])
test_text_dtm = vectorizer.transform(test['Summary'])

train = train_text_dtm
test = test_text_dtm

#Create a parameter grid: map the parameter names to the values that should be searched
param_grid = dict(max_depth = [10, 20])

#Instantiate the grid
rf = RandomForestRegressor(n_estimators = 100, random_state = 44, verbose = 3, n_jobs = -1)
print(rf)
grid = GridSearchCV(rf, param_grid, cv = 5, scoring = 'neg_mean_squared_error')

#Fit the models
grid.fit(train, train_score)

#Check the scores
scores = grid.cv_results_

#Print the best score - 19/02/2017 - 1.369 (rmse - 1.170)
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
    
test_preds_df.to_csv('model/rf1.csv', index=False)


"""

#Define the parameter values that should be searched
l1_ratio_range = np.arange(0, 1.2, 0.2)
alpha_range = 10.0**-np.arange(0,4)

#Create a parameter grid: map the parameter names to the values that should be searched
param_grid = dict(l1_ratio = l1_ratio_range, alpha = alpha_range)
print(param_grid)

#Instantiate the grid
lm = SGDRegressor(loss = 'squared_loss', n_iter = np.ceil(10**6 /train.shape[0]), random_state = 44, \
                   verbose = 3, n_jobs = -1)

grid = GridSearchCV(lm, param_grid, cv = 5, scoring = 'neg_mean_squared_error')

#Fit the model
grid.fit(train, train_score)

#Check the scores
scores = grid.cv_results_

#Print the best score
print("The best score is %s" % grid.best_score_)
print("The best model parameters are: %s" % grid.best_params_)

#Get the best model
rf_mod = grid.best_estimator_

#Fit the model on the entire training set
rf_mod.fit(train, Popular)

#Make test set predictions
test_preds = rf_mod.predict_proba(test)[:,1]

#Save the test predictions
test_preds = pd.DataFrame({
        "UniqueID": TestID,
        "Probability1": test_preds
})
    
test_preds.to_csv('python_practice/rf.csv', index=False)








#Convert to array
text_arr = text.values

#Instantiate Tfidf
vectorizer = TfidfVectorizer(min_df = 5, ngram_range = (1, 1), \
                             stop_words = 'english')

#Instantiate nmf
nmf = NMF(n_components = 6, random_state = 44)

#Pipeline
pipeline = make_pipeline(vectorizer, nmf)
components = pipeline.fit_transform(text_arr)

#Determine top words for each topic
def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()

#Call function
print_top_words(nmf, vectorizer.get_feature_names(), 5)

#For each topic, find the text that is the best representation of that topic
components_scaled = (components - components.mean(axis=0))/components.std(axis=0)

top_idx = np.argsort(components_scaled, axis=0)[-3:]

count = 0
for idxs in top_idx.T: 
    print("\nTopic {}:".format(count))
    for idx in idxs:
        print(text.iloc[idx])
    count += 1

#Name topics
topics_dict = {'topic' : {0 : 'generic', 1 : 'tea', 2 : 'coffee', 3: 'pets', 4 : 'shopping', 5 : 'chocolate'}}

#Create a DataFrame with the nmf components for each observation
components_df = pd.DataFrame(components, index = text.index)
components_explore_df = pd.DataFrame(components, index = text)

#DataFrame with the highest scoring topic for each observation
topics = pd.DataFrame(components_df.idxmax(axis = 'columns').rename('topic'))

#Replace topic column with their given name
topics = topics.replace(topics_dict)

#Left outer join reviews on topics
reviews = reviews.join(topics, how = 'left')

###

#t-Distributed Stochastic Neighbor Embedding
np.random.seed(44)
indices = np.unique(np.random.randint(low = 0, high=components.shape[0], size=10000))

tsne = TSNE(random_state = 44)
tsne_embedding = tsne.fit_transform(components[indices])
tsne_embedding = pd.DataFrame(tsne_embedding,columns=['x','y'])
tsne_embedding['topic'] = components[indices].argmax(axis=1)

#Need to create a copy - feather can't handle strided data
tsne_feather = tsne_embedding.copy()
feather.write_dataframe(tsne_feather, 'tsne_embedding.feather')

scatter = plt.scatter(data=tsne_embedding,x='x',y='y',s=6,c=tsne_embedding['topic'],cmap="Set1")
plt.axis('off')
plt.savefig('tsne.png', dpi = 500)
plt.show()


###

#Quick recommender system
from sklearn.preprocessing import normalize 
components_normalised = normalize(components)

#Create df
df = pd.DataFrame(components_normalised, index = text)

# Select the row corresponding to ... review
article = df.iloc[0]

# Compute the dot products: similarities
similarities = df.dot(article)

# Display those with the largest cosine similarity
print(similarities.nlargest())

"""
