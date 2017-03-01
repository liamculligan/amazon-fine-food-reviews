#Exploratory Data Analysis

#Amazon fine food reviews

#Load required packages
library(data.table)
library(dplyr)
library(dtplyr)
library(lubridate)
library(xts)
library(ggplot2)

#Read in csv
reviews = fread("Reviews.csv")

#Summarise
summary(reviews)

#Manipulate Data
reviews = reviews %>%
  mutate(ProductId = as.factor(ProductId), UserId = as.factor(UserId), ProfileName = as.factor(ProfileName)) %>%
  mutate(HelpfulnessRatio = HelpfulnessNumerator/HelpfulnessDenominator) %>%
  mutate(date = as.POSIXct(Time,  origin = "1970-01-01", tz = "UTC"))

#Summarise
summary(reviews)

#Summarise data for each data
each_date = reviews %>%
  group_by(date) %>%
  summarise(SumHelpfulnessNumerator = sum(HelpfulnessNumerator), SumHelpfulnessDenominator = sum(HelpfulnessDenominator))

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

#Plot hepful review count by timestamp
ggplot(each_date, aes(x = date, y = SumHelpfulnessNumerator)) + 
  geom_line(col = "blue") + 
  ggplot_theme + 
  labs(list(x = "Date", y = "Helpful Review Count", title = "Helpful Review Count over Time"))

#The above plot is not particularly informative - rather compute rolling averages per month and plot these

#Function to create xts object with the given rolling function applied to it
rolling_function = function(data, cols, date_index, time_step_period, time_step_value, smallest_datetime_period, summary_function, 
                            rolling_period) {
  
  #data - data frame
  #cols - vector of column names. Must contain at least 1 name (strings)
  #date_index - date column name (string)
  #time_step_period - produce one row for each period... e.g("seconds" through to "weeks")
  #time_step_value - the value related to the above period
  #smallest_datetime_period - related to the date index. Smallest resolution of time. Only "days" for date and "seconds" for datetime
  #summary_function - summary function applied to each period (string)
  #rolling_period - how many periods to roll the summary_function back
  
  #Check for at least one defined column
  if (length(cols) == 0) {
    stop('Please provide at least one column name in "cols"')
  }
  
  #Since we have at least one input column, create a vector of this column
  x = data[[cols[1]]]
  
  #cbind all other columns onto the first column
  if (length(cols) > 1) {
    for (i in 2:length(cols)) {
      x = cbind(x, data[[cols[i]]])
    }
  }
  
  #Create xts object
  xts_each_date = xts(x = x, order.by = data[[date_index]])
  
  #Define endpoints
  endpoints = endpoints(xts_each_date, time_step_period, time_step_value)
  
  #Summarise by endpoints
  
  #Since we have at least one input column, find endpoints for this column
  xts_summary = period.apply(xts_each_date[, 1], endpoints, sum)
  
  #cbind all other columns onto the first column
  if (length(cols) > 1) {
    for (i in 2:length(cols)) {
      xts_summary = cbind(xts_summary, period.apply(xts_each_date[, i], endpoints, sum))
    }
  }
  
  #Align to 1-week periods - must give the number of seconds number of seconds to adjust by
  
  #To get to seconds
  if (time_step_period == "seconds") {
    multiplier = 1
  } else if (time_step_period == "minutes") {
    multiplier = 60
  } else if ((time_step_period == "hours")) {
    multiplier = 60*60
  } else if ((time_step_period == "days")) {
    multiplier = 60*60*24
  } else if ((time_step_period == "weeks")) {
    multiplier = 60*60*24*7
  }
  
  xts_align = align.time(xts_summary, time_step_value * multiplier)
  
  #To get from seconds to input value of smallest_datetime_period
  if (smallest_datetime_period == "seconds") {
    merge_multiplier = multiplier
  } else if (smallest_datetime_period == "days") {
    merge_multiplier = multiplier/(60*60*24)
  }
  
  #If no attempts occured in a 1 week period, this period will be missing from the resulting xts object - need to correct
  #by argument must match finest date/time resolution - days in this case
  all_times = xts(x = NULL, seq(start(xts_align), end(xts_align), by = time_step_value * merge_multiplier))
  
  xts_each_date = merge(all_times, xts_align, fill = 0)
  
  #Count of the rolling average per month
  xts_rolling = rollapply(xts_each_date, rolling_period, summary_function)
  
  #Rename columns
  names(xts_rolling) = cols
  
  return(xts_rolling)
  
}

#Call function
xts_rolling = rolling_function(data = reviews, cols = c("HelpfulnessNumerator", "HelpfulnessDenominator"),
                               date_index = "date", time_step_period = "weeks", time_step_value = 1,
                               smallest_datetime_period = "days", summary_function = "mean", rolling_period = 4)

#Compute ratio
xts_rolling$HelpfulnessRatio = xts_rolling[, 2]/xts_rolling[, 1]

#Plots
ggplot(xts_rolling, aes(x = Index, y = HelpfulnessNumerator)) + 
  geom_line(col = "blue") + 
  ggplot_theme + 
  labs(list(x = "Date", y = "Rolling Average", title = "Rolling Average of Helpful Review \n Count per Month")) 

ggplot(xts_rolling, aes(x = Index, y = HelpfulnessDenominator)) + 
  geom_line(col = "blue") + 
  ggplot_theme + 
  labs(list(x = "Date", y = "Rolling Average", title = "Rolling Average of Review \n Count per Month"))

ggplot(xts_rolling, aes(x = Index, y = HelpfulnessRatio)) + 
  geom_line(col = "green") + 
  ggplot_theme + 
  labs(list(x = "Date", y = "Rolling Average", title = "Rolling Average of Helpful Review \n Ratio per Month"))

#Function to create xts object with the given rolling function applied to it
rolling_function_merge = function(data, cols, date_index, time_step_period, time_step_value, smallest_datetime_period,
                                  summary_function, rolling_period) {
  
  #data - data frame
  #cols - vector of column names. Must contain at least 1 name (strings)
  #date_index - date column name (string)
  #time_step_period - produce one row for each period... e.g("seconds" through to "weeks")
  #time_step_value - the value related to the above period
  #smallest_datetime_period - related to the date index. Smallest resolution of time. Only "days" for date and "seconds" for datetime
  #summary_function - summary function applied to each period (string)
  #rolling_period - how many periods to roll the summary_function back
  
  #Check for at least one defined column
  if (length(cols) == 0) {
    stop('Please provide at least one column name in "cols"')
  }
  
  #Since we have at least one input column, create a vector of this column
  x = data[[cols[1]]]
  
  #cbind all other columns onto the first column
  if (length(cols) > 1) {
    for (i in 2:length(cols)) {
      x = cbind(x, data[[cols[i]]])
    }
  }
  
  #Create xts object
  xts_each_date = xts(x = x, order.by = data[[date_index]], tzone = "UTC")
  
  #Define endpoints
  endpoints = endpoints(xts_each_date[, 1], time_step_period, time_step_value)
  
  #We need the length to be greater than 2 to use the rollapply function
  if (length(endpoints) > rolling_period) {
    
    #Summarise by endpoints
    
    #Since we have at least one input column, find endpoints for this column
    xts_summary = period.apply(xts_each_date[, 1], endpoints, sum)
    
    #cbind all other columns onto the first column
    if (length(cols) > 1) {
      for (i in 2:length(cols)) {
        xts_summary = cbind(xts_summary, period.apply(xts_each_date[, i], endpoints, sum))
      }
    }
    
    #Align to 1-week periods - must give the number of seconds number of seconds to adjust by
    
    #To get to seconds
    if (time_step_period == "seconds") {
      multiplier = 1
    } else if (time_step_period == "minutes") {
      multiplier = 60
    } else if ((time_step_period == "hours")) {
      multiplier = 60*60
    } else if ((time_step_period == "days")) {
      multiplier = 60*60*24
    } else if ((time_step_period == "weeks")) {
      multiplier = 60*60*24*7
    }
    
    xts_align = align.time(xts_summary, time_step_value * multiplier)
    
    #Create an aligned date vector for merging
    date_index_aligned = align.time(as.POSIXct(data[[date_index]], tz = "UTC"), n = time_step_value * multiplier)
    
    #To get from seconds to input value of smallest_datetime_period
    if (smallest_datetime_period == "seconds") {
      merge_multiplier = multiplier
    } else if (smallest_datetime_period == "days") {
      merge_multiplier = multiplier/(60*60*24)
    }
    
    #If no attempts occured in a 1 week period, this period will be missing from the resulting xts object - need to correct
    #by argument must match finest date/time resolution - days in this case
    all_times = xts(x = NULL, seq(start(xts_align), end(xts_align), by = time_step_value * merge_multiplier))
    
    xts_each_date = merge(all_times, xts_align, fill = 0)
    
    #Count of the rolling average per month
    xts_rolling = rollapply(xts_each_date, width = rolling_period, summary_function, fill = NA, align = "right")
    
    #Now need to lag xts_rolling columns by 1 period so that no future information leaks
    
    for (i in 1:length(cols)) {
      xts_rolling[, i] = lag( xts_rolling[, i], k =1)
    }
    
    #Rename columns
    names(xts_rolling) = paste(cols, "roll", rolling_period, time_step_period, sep = "_")
    
    #Convert xts object to data frame for merging
    df_rolling = as.data.table(xts_rolling)
    
    #Define date_aligned column name for data
    date_index_aligned_name = paste0(date_index, "_aligned")
    
    #Add data_aligned vector to data
    data[[date_index_aligned_name]] = date_index_aligned
    
    #Set required keys prior to joining
    # set2keyv(data, date_index_aligned_name)
    # setkey(df_rolling, index)
    # 
    # #Left outer join data on rolling data frame
    # data = df_rolling[data, on = date_index_aligned_name]
    
    data = merge(data, df_rolling, by.x = date_index_aligned_name, by.y = "index", all.x = T, all.y = F)
    
    #Remove date_aligned variable from data
    data[[date_index_aligned_name]] = NULL
  
  } else {
    
    #Create NA columns
    new_cols = paste(cols, "roll", rolling_period, time_step_period, sep = "_")
    
    for (new_col in new_cols) {
      data[[new_col]] = as.numeric(NA)
    }
    
  }
  
  #Return data
  return(data)
  
}

#Call function
reviews_merged = rolling_function_merge(data = reviews, cols = c("HelpfulnessNumerator", "HelpfulnessDenominator"),
                                        date_index = "date", time_step_period = "weeks", time_step_value = 1,
                                        smallest_datetime_period = "seconds", summary_function = "mean", rolling_period = 4)

#Call the above function but grouped by UserId

#Select only the top 5 profiles
top_6_profiles = reviews %>%
  group_by(UserId) %>%
  summarise(profile_count = n()) %>%
  arrange(desc(profile_count)) %>%
  head(6)

profile_reviews = reviews %>%
  filter(UserId %in% top_6_profiles$UserId)

profile_reviews_merged = profile_reviews[, 
                                         rolling_function_merge(data = .SD,
                                                                 cols = c("HelpfulnessNumerator", "HelpfulnessDenominator"),
                                                                 date_index = "date", time_step_period = "weeks", 
                                                                 time_step_value = 1, smallest_datetime_period = "days", 
                                                                 summary_function = "mean", rolling_period = 4),
                                         by = UserId]

#Calculate ratio
profile_reviews_merged[, HelpfulnessRatio_roll_4_weeks := HelpfulnessNumerator_roll_4_weeks/HelpfulnessDenominator_roll_4_weeks]

#Plot
ggplot(profile_reviews_merged, aes(x = date, y = HelpfulnessNumerator_roll_4_weeks, col = UserId)) +
  geom_line()

ggplot(profile_reviews_merged, aes(x = date, y = HelpfulnessNumerator_roll_4_weeks, col = UserId)) +
  geom_line() +
  facet_wrap(~ UserId)

#Repeat but for whole data frame

#First select only users with more than 1 reviews
users = reviews %>%
  group_by(UserId) %>%
  summarise(profile_count = n()) %>%
  filter(profile_count > 94)

user_reviews = reviews %>%
  filter(UserId %in% users$UserId)

user_reviews_merged = user_reviews[, 
                                         rolling_function_merge(data = .SD,
                                                                cols = c("HelpfulnessNumerator", "HelpfulnessDenominator"),
                                                                date_index = "date", time_step_period = "weeks", 
                                                                time_step_value = 1, smallest_datetime_period = "seconds", 
                                                                summary_function = "mean", rolling_period = 4),
                                         by = UserId]

