#Exploratory Data Analysis

#Amazon fine food reviews

#Load required packages
library(data.table)
library(dplyr)
library(dtplyr)
library(lubridate)
library(xts)
library(feather)
library(ggplot2)

#Read in feather file - from eda.py
reviews_merged = read_feather('eda/reviews_rolling.feather') %>%
  as.data.table()

summary(reviews_merged)

#Manipulate Data
reviews_merged = reviews_merged %>%
  mutate(ProductId = as.factor(ProductId), UserId = as.factor(UserId), ProfileName = as.factor(ProfileName)) %>%
  mutate(HelpfulnessRatio = HelpfulnessNumerator/HelpfulnessDenominator) %>%
  mutate(HelpfulnessRollingAverageRatio = HelpfulnessNumeratorRollingAverage/HelpfulnessDenominatorRollingAverage)

#Summarise
summary(reviews_merged)

#Set theme for ggplot2
themes_data = {
  x = list()
  
  x$colours =
    c(dkgray = rgb(60, 60, 60, max = 255),
      medgray = rgb(210, 210, 210, max = 255),
      ltgray = rgb(240, 240, 240, max = 255),
      red = rgb(255, 39, 0, max = 255),
      blue = rgb(0, 143, 213, max = 255),
      green = rgb(119, 171, 67, max = 255))
  
  x
}
ggplot_theme = theme(
  line = element_line(colour = "black"),
  rect = element_rect(fill = themes_data$colours["ltgray"], linetype = 0, colour = NA),
  text = element_text(colour = themes_data$colours["dkgray"]),
  axis.ticks = element_blank(),
  axis.line = element_blank(),
  legend.background = element_rect(),
  legend.position = "top",
  legend.direction = "horizontal",
  legend.box = "vertical",
  panel.grid = element_line(colour = NULL),
  panel.grid.major =
    element_line(colour = themes_data$colours["medgray"]),
  panel.grid.minor = element_blank(),
  plot.title = element_text(hjust = 0.5, size = 20),
  axis.title = element_text(size = 16),
  axis.text = element_text(size = 14),
  plot.margin = unit(c(1, 1, 1, 1), "lines"),
  strip.background = element_rect()
)

#Plots
ggplot(reviews_merged, aes(x = date_time, y = HelpfulnessNumeratorRollingAverage)) + 
  geom_line(col = "blue") + 
  ggplot_theme + 
  labs(list(x = "Date", y = "Rolling Average", title = "Rolling Average of Helpful Review \n Count per Month")) 

ggplot(reviews_merged, aes(x = date_time, y = HelpfulnessDenominatorRollingAverage)) + 
  geom_line(col = "blue") + 
  ggplot_theme + 
  labs(list(x = "Date", y = "Rolling Average", title = "Rolling Average of Review \n Count per Month"))

ggplot(reviews_merged, aes(x = date_time, y = HelpfulnessRollingAverageRatio)) + 
  geom_line(col = "green") + 
  ggplot_theme + 
  labs(list(x = "Date", y = "Rolling Average", title = "Rolling Average of Helpful Review \n Ratio per Month"))