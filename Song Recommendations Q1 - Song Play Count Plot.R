library(ggplot2)
library(dplyr)
library(tidyverse)

song_play_count_df <- read_csv("C:\\Users\\coach\\OneDrive - University of Canterbury\\MADS\\DATA420 Assignment 2 Code\\song_play_count.csv\\song_play_count.csv", col_names = TRUE)
ggplot(song_play_count_df, aes(x=plays_count)) + 
  geom_histogram(binwidth=5000, colour = "midnightblue", fill="lightskyblue") +
  ggtitle("Song Play Count", subtitle = "Distribution of Song Popularity") +
  xlab("Count") +
  ylab("Frequency") +
  scale_x_continuous(labels = scales::comma)

