#Word Cloud

#Load required packages
from wordcloud import WordCloud, ImageColorGenerator
import pandas as pd
import numpy as np
import string
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from PIL import Image

#Read in csv
reviews = pd.read_csv('Reviews.csv', index_col = 'Id')

#Create text
summary = reviews['Summary'].str.cat(sep=' ')

#Convert to lower case
summary = summary.lower()

#Remove punctuation

# This uses the 3-argument version of str.maketrans
# with arguments (x, y, z) where 'x' and 'y'
# must be equal-length strings and characters in 'x'
# are replaced by characters in 'y'. 'z'
# is a string (string.punctuation here)
# where each character in the string is mapped
# to None
translator = str.maketrans('', '', string.punctuation)

summary = summary.translate(translator)

#Create stop words
stop_words = stopwords.words("english")

# Generate a word cloud image
wordcloud = WordCloud(random_state = 44).generate(summary)

# Display the generated image:
# the matplotlib way:
plt.imshow(wordcloud)
plt.axis("off")
plt.savefig('eda/summary-word-cloud-py.png', dpi = 500)

###

#Overlay the word cloud on the Amazon Fresh logo with the same colouring

# read the mask image
mask = np.array(Image.open('amazon-fresh-logo.jpg'))

# create coloring from image
image_colors = ImageColorGenerator(mask)

# Generate a word cloud image
wordcloud = WordCloud(background_color = "white", max_words = 500, mask = mask, \
                      random_state = 44).generate(summary)

#Colour wordcloud
# we could also give color_func=image_colors directly in the constructor
plt.imshow(wordcloud.recolor(color_func = image_colors))
plt.axis("off")
plt.savefig('eda/summary-word-cloud-coloured.png', dpi = 500)
plt.figure()
plt.imshow(mask, cmap = plt.cm.gray)
plt.axis("off")
plt.show()

###

#Create text
text = reviews['Text'].str.cat(sep=' ')

#Convert to lower case
text = text.lower()

#Remove punctuation
translator = str.maketrans('', '', string.punctuation)

text = text.translate(translator)

#Create stop words
stop_words = stopwords.words("english")
stop_words.append("br")

# Generate a word cloud image
wordcloud = WordCloud(stopwords = stop_words, random_state = 44).generate(text)

# Display the generated image:
# the matplotlib way:
plt.imshow(wordcloud)
plt.axis("off")
plt.savefig('eda/text-word-cloud.png', dpi = 500)

#Overlay the word cloud on the Amazon Fresh logo with the same colouring

# read the mask image
mask = np.array(Image.open('amazon-fresh-logo.jpg'))

# create coloring from image
image_colors = ImageColorGenerator(mask)

# Generate a word cloud image
wordcloud = WordCloud(background_color = "white", max_words = 1000, mask = mask, \
                      random_state = 44).generate(text)

#Colour wordcloud
# we could also give color_func=image_colors directly in the constructor
plt.imshow(wordcloud.recolor(color_func = image_colors))
plt.axis("off")
plt.savefig('eda/text-word-cloud-coloured.png', dpi = 500)
plt.figure()
plt.imshow(mask, cmap = plt.cm.gray)
plt.axis("off")
plt.show()