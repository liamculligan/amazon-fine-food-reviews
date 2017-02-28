#Non-negative Matrix Factorisation

#Load required packages
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.pipeline import make_pipeline
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import feather

#Read in csv
reviews = pd.read_csv('Reviews.csv', index_col = 'Id')

#Create text - drop duplicates and convert to lower case
text = reviews['Text'].drop_duplicates().str.lower().str.replace('<br />','').str.replace('[^\w\s]','')

#Replace nan with ""
text = text.fillna(value = "")

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

"""
scatter = plt.scatter(data=tsne_embedding,x='x',y='y',s=6,c=tsne_embedding['topic'],cmap="Set1")
plt.axis('off')
plt.savefig('tsne.png', dpi = 500)
plt.show()
"""

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
