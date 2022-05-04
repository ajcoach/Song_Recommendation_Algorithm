library(ggplot2)
library(dplyr)
library(tidyverse)

user_play_count_df <- read_csv("C:\\Users\\coach\\OneDrive - University of Canterbury\\MADS\\DATA420 Assignment 2 Code\\user_play_count.csv\\user_play_count.csv", col_names = TRUE)


ggplot(user_play_count_df, aes(x=plays_count)) + 
  geom_histogram(binwidth=500, colour = "midnightblue", fill="lightskyblue") +
  ggtitle("User Play Count", subtitle = "Distribution of User Activity") +
  xlab("Count") +
  ylab("Frequency") +
  scale_x_continuous(labels = scales::comma)

