#Predict the product review score

#Load required packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.externals import joblib
import random

#Read in csv
reviews_score = pd.read_csv('Reviews.csv', index_col = 'Id', usecols = ['Id', 'Score'])

#Split train and test - only for target variable
train_score_indices, test_score_indices = train_test_split(reviews_score, train_size = 0.8,
                                                           stratify = reviews_score['Score'], random_state = 44)

#Picke train_score
joblib.dump(train_score_indices, 'train_score_indices.pkl')

#Take a random sample of train_score_indices to read in for the next section
train_indices_load = train_score_indices.sample(n = 200000).index

#Number of rows in reviews
n = reviews_score.shape[0]

#Desired sample size from the whole reviews dataset - only the ~80% of these in train will be used
s = 250000

#Random sample of the row numbers to be skipped
skip = sorted(random.sample(range(1, n), n-s))

#Read in csv
reviews = pd.read_csv('Reviews.csv', index_col = 'Id', usecols = ['Id', 'Summary', 'Text', 'Score'],
                      skiprows = skip)

#Only need to train vectorizers on training data
reviews = reviews.iloc[reviews.index.isin(train_score_indices.index)]

#Drop duplicate score-text values
reviews = reviews.drop_duplicates(subset = ['Summary', 'Text', 'Score'])

#Create text - remove line breaks
reviews['Summary'] = reviews['Summary'].str.replace('<br />', ' ')
reviews['Text'] = reviews['Text'].str.replace('<br />', ' ')

#Replace nan with ""
reviews['Summary'] = reviews['Summary'].fillna(value = "")
reviews['Text'] = reviews['Text'].fillna(value = "")

#Remove score from reviews
reviews = reviews.drop('Score', axis = 'columns')

#Summary dtm
summary_vectorizer = CountVectorizer(min_df = 0.0005, max_df = 1.0, ngram_range = (1, 3))
summary_vectorizer.fit(reviews['Summary'])

#Pickle the model
joblib.dump(summary_vectorizer, 'summary_vectorizer.pkl')

#Text dtm
text_vectorizer = CountVectorizer(min_df = 0.001, max_df = 1.0, ngram_range = (1, 3))
text_vectorizer.fit(reviews['Text'])

#Pickle the model
joblib.dump(text_vectorizer, 'text_vectorizer.pkl')

#Non-negative matrix factorisation to identify topics in Text
#Remove punctuation
reviews_text = reviews['Text'].str.replace('[^\w\s]','')

#Instantiate Tfidf
nmf_vectorizer = TfidfVectorizer(min_df = 5, ngram_range = (1, 1), stop_words = 'english')

#Instantiate NMF
nmf = NMF(n_components = 6, random_state = 44)

#Pipeline
nmf_pipeline = make_pipeline(nmf_vectorizer, nmf)
nmf_pipeline.fit(reviews_text)

#Pickle the model
joblib.dump(nmf_pipeline, 'nmf_pipeline.pkl')

#Transform so that the OHE can be fit
reviews_text = nmf_pipeline.transform(reviews_text)

#Find most likely topic for each observation
reviews_text_topics = reviews_text.argmax(axis=1)

#One hot encode topics
topic_ohe = OneHotEncoder()
topic_ohe.fit(reviews_text_topics.reshape(-1, 1))

#Pickle the model
joblib.dump(topic_ohe, 'topic_ohe.pkl')

