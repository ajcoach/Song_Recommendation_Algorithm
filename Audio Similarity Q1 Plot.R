library(ggplot2)
library(dplyr)
library(tidyverse)


genre_df <- read_csv("C:\\Users\\coach\\OneDrive - University of Canterbury\\MADS\\DATA420 Assignment 2 Code\\genre_types.csv\\combined_genre_types.csv", col_names = TRUE)


ggplot(genre_df, aes(x=Genre, y=Count)) + 
  geom_bar(stat = "identity", colour = "midnightblue", fill="lightskyblue") + 
  coord_flip() +
  ggtitle("Genre Count", subtitle = "Distribution of Genres")



