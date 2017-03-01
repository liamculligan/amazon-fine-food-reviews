#EDA

#Load required packages
import pandas as pd
import feather

#Read in csv
reviews = pd.read_csv('Reviews.csv', index_col = 'Id')

#Convert timestamp to date
reviews['date_time'] = pd.to_datetime(reviews['Time'], unit = 's')

# Setting the date as the index
reviews.set_index('date_time', inplace=True)

# Resampling data by week (only looking backwards)
reviews1 = reviews.resample('10080T', label = 'right').sum().fillna(value = 0)

# Calculate moving average
reviews1 = reviews1.assign(HelpfulnessNumeratorRollingAverage = \
                           reviews1['HelpfulnessNumerator'].rolling(window=4).mean().shift(periods = 1),
    HelpfulnessDenominatorRollingAverage = \
                           reviews1['HelpfulnessDenominator'].rolling(window=4).mean().shift(periods = 1))

#Reindex reviews1 based on nearest index of reviews
reviews1 = reviews1.reindex(reviews.index, method='nearest')

#Sort both DataFrames by date_time
reviews = reviews.sort_index()
reviews1 = reviews1.sort_index()

#Drop unecessary columns in reviews1
reviews1 = reviews1.drop(['Score', 'Time', 'HelpfulnessNumerator', 'HelpfulnessDenominator'], \
                         axis = 'columns')

#Concatenate DataFrames
reviews_merged = pd.concat([reviews, reviews1], axis = 'columns')

#Need date_time index to be column for feather
reviews_merged.reset_index(level = 0, inplace = True)

#Write DataFrame to feather file
feather.write_dataframe(reviews_merged, 'eda/reviews_rolling.feather')