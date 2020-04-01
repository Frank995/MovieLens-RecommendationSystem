###########################################################
# HarvardX PH125.9x Data Science Capstone Movielens Project
#
# Francesco Pudda, 2020
###########################################################

# IMPORTANT NOTES
# Since this project was carried out using a
# low-performance latptop (as much as I can afford)
# I tried to free as much memory as possible
# whenever a variable was not needed anymore
# Moreover, given the size of the dataset,  I was
# also restricted to functions that did not need
# too much memory.
# In this file I am going to describe every piece
# of code bu I will leave discussion and equations
# to the attached report.

################################
# Library import
################################

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
if(!require(Metrics)) install.packages("Metrics", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)
library(lubridate)
library(Metrics)

################################
# Create edx set, validation set
################################

# Note: this process could take a couple of minutes.

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data.
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set.
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set.
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

# Free RAM
invisible(gc())

################################
# Exploratory analysis
################################

# Check data frame structure and classes.
head(edx, 3)
sapply(edx, class)

# Let us first reduce computation time and
# memory usage by deleting users and films
# in edx that are not present in validation,
# and title, because we will need only the
# id of the film.
# Moreover, convert movieId to integer for
# less memory usage.
edx$movieId <- as.integer(edx$movieId)
validation <- validation %>% 
  select(-title)
edx <- edx %>% select(-title) %>%
  semi_join(validation, by = "userId") %>%
  semi_join(validation, by = "movieId")
	
# Add a couple of columns that will be useful later
edx <- edx %>% 
  mutate(time = round_date(as_datetime(timestamp), unit = "week"),
         week = week(as_datetime(timestamp))) %>%
  select(-timestamp)
validation <- validation %>%
  mutate(time = round_date(as_datetime(timestamp), unit = "week"),
         week = week(as_datetime(timestamp))) %>%
  select(-timestamp)

# To begin let's check ratings distributions by
# the most important variables

# Ratings per film
edx %>% 
  group_by(movieId) %>%
  summarise(count = n()) %>%
  ggplot(aes(count)) + 
  geom_histogram(bins = 30, color = "blue") +
  scale_x_log10() + 
  ggtitle("Number of ratings per film") +
  xlab("Film ID") + 
  ylab("Number of ratings")

# Ratings per user
edx %>% 
  group_by(userId) %>%
  summarise(count = n()) %>%
  ggplot(aes(count)) + 
  geom_histogram(bins = 30, color = "red") +
  scale_x_log10() + 
  ggtitle("Number of ratings per user") +
  xlab("User ID") + 
  ylab("Number of ratings")

# Ratings per genre
edx %>% 
  group_by(genres) %>%
  summarise(count = n()) %>%
  ggplot(aes(count)) + 
  geom_histogram(bins = 30, color = "green") +
  scale_x_log10() + 
  ggtitle("Number of ratings per genre") +
  xlab("Genre") + 
  ylab("Number of ratings") +
  theme(axis.text.x=element_blank(),
        axis.ticks.x=element_blank())


# This will check whether it is present any time effect.
# I am mutating the dataset to include timestamp even though
# I had previously deleted it. This is because this is the
# only time I will use it.
regression <- edx %>%
  mutate(timestamp = as.numeric(time)) %>%
  lm(rating ~ timestamp, data = .)

# Plotting the time series.
cols <- c("Smooth interpolation"="#0000ff","Regression line"="#ff0000")
edx %>%
  group_by(time) %>%
  summarise(avg = mean(rating)) %>%
  ggplot(aes(time, avg)) +
  geom_point(size=1.1) +
  geom_smooth(aes(colour = "Smooth interpolation"), size = 1.2) +
  geom_abline(aes(colour = "Regression line",
                  intercept = regression$coefficients["(Intercept)"],
                  slope = regression$coefficients["timestamp"]),
              size = 1.3) +
  scale_colour_manual(name="Legend", values=cols) +
  ggtitle("Average rating per week") + 
  xlab("Year") + 
  ylab("Average rating")

# Checking also any intra-year effect, by
# grouping by week of the year instead of rounding
# date to the nearest week like above.
edx %>%
  group_by(week) %>%
  summarise(avg = mean(rating)) %>%
  ggplot(aes(week, avg)) +
  geom_point() +
  geom_smooth() +
  ggtitle("Average rating throughout the year") + 
  xlab("Week of the year") + 
  ylab("Average rating")

# These pieces of code below will study other
# biases by plotting the number of ratings
# vs. average ratings calculated by grouping by
# userId and movieId.
edx %>%
  group_by(movieId) %>%
  summarise(count = n(), avg = mean(rating)) %>%
  ggplot(aes(count, avg)) +
  geom_point() +
  geom_smooth() +
  ggtitle("Average ratings based on films number of reviews") + 
  xlab("Reviews count") + 
  ylab("Average rating")

edx %>%
  group_by(userId) %>% summarise(count = n(), avg = mean(rating)) %>%
  ggplot(aes(count, avg)) +
  geom_point() +
  geom_smooth() +
  ggtitle("Average ratings based on users' number of reviews") + 
  xlab("Reviews count") + 
  ylab("Average rating")

# To conclude this is to check any possible
# genre effect, firstly with 'compound genres'
# and secondly by considering 'only single' ones.
# By compound genres I mean genres as given in the
# dataset, e.g Drama|War|Action.
edx %>%
  group_by(genres) %>%
  summarise(n = n(), avg = mean(rating), se = sd(rating)/sqrt(n())) %>%
  filter(n >= quantile(n, 0.98)) %>% 
  mutate(genres = reorder(genres, avg)) %>%
  ggplot(aes(x = genres, y = avg, ymin = avg - 2*se, ymax = avg + 2*se)) + 
  geom_point() +
  geom_errorbar() + 
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  ggtitle("Average ratings by compound genre") + 
  xlab("Genres") + 
  ylab("Average rating")

# Splitting by and then grouping by genres would
# would result in a computer freeze because
# the data frame would become too huge.
# For this reason in this case I am first 
# grouping by movieId.
edx %>%
  group_by(movieId) %>%
  summarise(count = n(), avg = mean(rating), genres = first(genres)) %>%
  separate_rows(genres, sep="\\|") %>%
  group_by(genres) %>%
  summarise(avg = mean(avg)) %>%
  mutate(genres = reorder(genres, avg)) %>%
  ggplot(aes(x = genres, y = avg)) +
  geom_point() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  ggtitle("Average ratings by genre") + 
  xlab("Genre") + 
  ylab("Average rating")

# Clean up some RAM
rm(regression, cols)
invisible(gc())

#################################
# Data analysis and preprocessing
#################################

# For training, using the lm() function would be
# just impractical given the size of the dataset.
# It is possible, though, to compute users and
# movies biases by studying one term of the
# linear regression at a time. The chosen
# metrics for performance evaluation is rmse.
# Important note: to avoid overtraining I
# will be using the validation set just for the
# final rmse calculus, whereas cross validation
# will be being used for any intermediate step.

# Create bootstrap indexes. 5 bootstraps were chosen
# for performance reasons.
n_bootstraps <- 5
bs_index <- createDataPartition(edx$rating, times=n_bootstraps, p=0.1, list=FALSE)

# Unfortunately, given the equation in the report,
# for each following step it is needed the bias vectors
# computed in the previous ones.
# The cleanest approach I thought of, was to call apply
# twice for each step. The first one is to declare
# global variables for biases that will be used right
# after that, with the second call to compute the rmse.
# IMPORTANT: instead of passing the bootstrap indexes
# columns as argument, I will pass the index of the column
# so that I can access again the index of the global variable
# in the next call.
# Another solution could have been to declare an empty vectors
# of length 5 for each bias, and to call apply only once.
# Inside the call I would have then filled such vectors with
# components computed for each bootstrap.
# Let's start building our model now.

# For first, we could neglect every sort of bias and
# assume the same rating for all movies and user,
# so that differences are just given by randomness.
mu <- lapply(1:n_bootstraps, function(i){
  mean(edx[-bs_index[,i], "rating"])
})

naive_rmse <- mean(sapply(1:n_bootstraps, function(i){
  y_hat <- mu[[i]]
  rmse(edx[bs_index[,i],"rating"], y_hat)
}))

cat("Naive rmse: ", naive_rmse, "\n")

# We can improve it by actually considering that some
# movies are just considered better than others and rated
# averagely higher or lower.
movie_bias <- lapply(1:n_bootstraps, function(i){
  edx[-bs_index[,i],] %>%
  group_by(movieId) %>% 
  summarise(b_i = mean(rating - mu[[i]]))
})

movie_rmse <- mean(sapply(1:n_bootstraps, function(i){
  y_hat <- mu[[i]] + edx[bs_index[,i],] %>%
    group_by(movieId) %>%
    left_join(movie_bias[[i]], by = "movieId") %>%
    pull(b_i)
  nas <- which(is.na(y_hat))
  rmse(edx[bs_index[,i],"rating"][-nas], y_hat[-nas])
}))

cat("Movie effect rmse: ", movie_rmse, "\n")

# One more step forward. Let us consider the user
# effect. Some users may, in fact, be more harsh
# in their ratings or just have preferences for
# some films.
user_bias <- lapply(1:n_bootstraps, function(i){
  edx[-bs_index[,i],] %>%
    left_join(movie_bias[[i]], by='movieId') %>%
    group_by(userId) %>% 
    summarise(b_u = mean(rating - mu[[i]] - b_i))
})

user_rmse <- mean(sapply(1:n_bootstraps, function(i){
  y_hat <- mu[[i]] + edx[bs_index[,i],] %>%
    group_by(movieId) %>%
		left_join(movie_bias[[i]], by = "movieId") %>%
		left_join(user_bias[[i]], by = "userId") %>%
		ungroup() %>%
		select(b_i, b_u) %>%
    rowSums()
  nas <- which(is.na(y_hat))
	rmse(edx[bs_index[,i],"rating"][-nas], y_hat[-nas])
}))

cat("Movie + user effect rmse: ", user_rmse, "\n")

# We noted an intra year effect on the average
# rating. This will check whether it will affect
# the RMSE.
week_bias <- lapply(1:n_bootstraps, function(i){
  edx[-bs_index[,i],] %>%
    left_join(movie_bias[[i]], by='movieId') %>%
	  left_join(user_bias[[i]], by='userId') %>%
    group_by(week) %>% 
    summarise(b_w = mean(rating - mu[[i]] - b_i - b_u))
})

week_rmse <- mean(sapply(1:n_bootstraps, function(i){
  y_hat <- mu[[i]] + edx[bs_index[,i],] %>%
    group_by(movieId) %>%
    left_join(movie_bias[[i]], by = "movieId") %>%
		left_join(user_bias[[i]], by = "userId") %>%
		left_join(week_bias[[i]], by = "week") %>%
		ungroup() %>%
		select(b_i, b_u, b_w) %>% 
		rowSums()
	nas <- which(is.na(y_hat))
	rmse(edx[bs_index[,i],"rating"][-nas], y_hat[-nas])
}))

cat("Movie + user + week effect rmse: ", week_rmse, "\n")

# And then since the regression calculated in the
# first plot had a significant slope, the effect of
# must be taken into account.
time_bias <- lapply(1:n_bootstraps, function(i){
  edx[-bs_index[,i],] %>%
    left_join(movie_bias[[i]], by='movieId') %>%
	  left_join(user_bias[[i]], by='userId') %>%
	  left_join(week_bias[[i]], by='week') %>%
    group_by(time) %>% 
    summarise(b_t = mean(rating - mu[[i]] - b_i - b_u - b_w))
})

time_rmse <- mean(sapply(1:n_bootstraps, function(i){
  y_hat <- mu[[i]] + edx[bs_index[,i],] %>%
    group_by(movieId) %>%
		left_join(movie_bias[[i]], by = "movieId") %>%
		left_join(user_bias[[i]], by = "userId") %>%
		left_join(week_bias[[i]], by = "week") %>%
		left_join(time_bias[[i]], by = "time") %>%
		ungroup() %>%
		select(b_i, b_u, b_w, b_t) %>% 
		rowSums()
	nas <- which(is.na(y_hat))
	rmse(edx[bs_index[,i],"rating"][-nas], y_hat[-nas])
}))

cat("Movie + user + time effect rmse: ", time_rmse, "\n")

# Finally, let's consider the effect of the genre
# on ratings.
genre_bias <- lapply(1:n_bootstraps, function(i){
  edx[-bs_index[,i],] %>%
    left_join(movie_bias[[i]], by='movieId') %>%
	  left_join(user_bias[[i]], by='userId') %>%
	  left_join(week_bias[[i]], by='week') %>%
	  left_join(time_bias[[i]], by='time') %>%
    group_by(genres) %>% 
    summarise(b_g = mean(rating - mu[[i]] - b_i - b_u - b_w - b_t))
})

genre_rmse <- mean(sapply(1:n_bootstraps, function(i){
  y_hat <- mu[[i]] + edx[bs_index[,i],] %>%
    group_by(movieId) %>%
		left_join(movie_bias[[i]], by = "movieId") %>%
		left_join(user_bias[[i]], by = "userId") %>%
		left_join(week_bias[[i]], by = "week") %>%
		left_join(time_bias[[i]], by = "time") %>%
		left_join(genre_bias[[i]], by = "genres") %>%
		ungroup() %>%
		select(b_i, b_u, b_w, b_t, b_g) %>% 
		rowSums()
	nas <- which(is.na(y_hat))
	rmse(edx[bs_index[,i],"rating"][-nas], y_hat[-nas])
}))

cat("Movie + user + time + genre effect rmse: ", genre_rmse, "\n")

# Free memory
rm(mu, movie_bias, user_bias, week_bias, time_bias, genre_bias)
invisible(gc())

# Our model has really improved but it can still
# get better.
# In fact, as shown at the beginning of exploratory
# analysis, there are many movies, users and genres with
# with very few ratings that could produce more uncertainty.												 
# We are then going to use regularisation to penalise
# large estimates resulted by such few samples.
# The best regularisation parameter is unknown so
# we are going to use a trial and error approach.
# NOTE: At first I tried with a wider range but
# computational time was very high. So, after having
# obtained the best parameter I reduced the interval
# to render the script faster
lambdas <- seq(3.5, 7.5, 0.5)

rmse_lambda <- sapply(lambdas, function(lambda){
  cat("Testing lambda = ", lambda, "\n")
  
  mu <- lapply(1:n_bootstraps, function(i){
    mean(edx[-bs_index[,i], "rating"])
  })
  
  movie_bias <- lapply(1:n_bootstraps, function(i){
    edx[-bs_index[,i],] %>%
      group_by(movieId) %>% 
      summarise(b_i = sum(rating - mu[[i]]) / (lambda + n()))
  })
  
  user_bias <- lapply(1:n_bootstraps, function(i){
    edx[-bs_index[,i],] %>%
      left_join(movie_bias[[i]], by='movieId') %>%
      group_by(userId) %>% 
      summarise(b_u = sum(rating - mu[[i]] - b_i) / (lambda + n()))
  })
  
  week_bias <- lapply(1:n_bootstraps, function(i){
    edx[-bs_index[,i],] %>%
      left_join(movie_bias[[i]], by='movieId') %>%
      left_join(user_bias[[i]], by='userId') %>%
      group_by(week) %>% 
      summarise(b_w = mean(rating - mu[[i]] - b_i - b_u))
  })
  
  time_bias <- lapply(1:n_bootstraps, function(i){
    edx[-bs_index[,i],] %>%
      left_join(movie_bias[[i]], by='movieId') %>%
      left_join(user_bias[[i]], by='userId') %>%
      left_join(week_bias[[i]], by='week') %>%
      group_by(time) %>% 
      summarise(b_t = mean(rating - mu[[i]] - b_i - b_u - b_w))
  })
  
  genre_bias <- lapply(1:n_bootstraps, function(i){
    edx[-bs_index[,i],] %>%
      left_join(movie_bias[[i]], by='movieId') %>%
      left_join(user_bias[[i]], by='userId') %>%
      left_join(week_bias[[i]], by='week') %>%
      left_join(time_bias[[i]], by='time') %>%
      group_by(genres) %>% 
      summarise(b_g = sum(rating - mu[[i]] - b_i - b_u - b_w - b_t) / (lambda + n()))
  })
  
  mean(sapply(1:n_bootstraps, function(i){
    y_hat <- mu[[i]] + edx[bs_index[,i],] %>%
      group_by(movieId) %>%
      left_join(movie_bias[[i]], by = "movieId") %>%
      left_join(user_bias[[i]], by = "userId") %>%
      left_join(week_bias[[i]], by = "week") %>%
      left_join(time_bias[[i]], by = "time") %>%
      left_join(genre_bias[[i]], by = "genres") %>%
      ungroup() %>%
      select(b_i, b_u, b_w, b_t, b_g) %>% 
      rowSums()
    nas <- which(is.na(y_hat))
    rmse(edx[bs_index[,i],"rating"][-nas], y_hat[-nas])
  }))
})

# Plot obtained RMSEs
plot(lambdas, rmse_lambda, main="RMSE by regularisation parameter",
     xlab="Lambda", ylab="RMSE")

best_lambda <- lambdas[which.min(rmse_lambda)]
regularised_rmse <- min(rmse_lambda)

cat("Best regularised rmse: ", regularised_rmse, "\n")
cat("Best regularisation parameter: ", best_lambda, "\n")

# Clean up
rm(n_bootstraps, bs_index, rmse_lambda, lambdas)
invisible(gc())

# Now that best lambda was calculated we can finally
# validate our model with the validation set.
# We used cross validation for model checking and now
# we can build the final model by using all the data.
mu <- mean(edx$rating)

movie_bias <- edx %>%
  group_by(movieId) %>% 
  summarise(b_i = sum(rating - mu) / (best_lambda + n()))

user_bias <- edx %>%
  left_join(movie_bias, by='movieId') %>%
  group_by(userId) %>%
  summarise(b_u = sum(rating - mu - b_i) / (best_lambda + n()))

week_bias <- edx %>%
  left_join(movie_bias, by='movieId') %>%
  left_join(user_bias, by='userId') %>%
  group_by(week) %>% 
  summarise(b_w = mean(rating - mu - b_i - b_u))

time_bias <- edx %>%
  left_join(movie_bias, by='movieId') %>%
  left_join(user_bias, by='userId') %>%
  left_join(week_bias, by='week') %>%
  group_by(time) %>%
  summarise(b_t = mean(rating - mu - b_i - b_u - b_w))

genre_bias <- edx %>%
  left_join(movie_bias, by='movieId') %>%
  left_join(user_bias, by='userId') %>%
  left_join(week_bias, by='week') %>%
  left_join(time_bias, by='time') %>%
  group_by(genres) %>% 
  summarise(b_g = sum(rating - mu - b_i - b_u - b_w - b_t) / (best_lambda + n()))

y_hat <- mu + validation %>%
  group_by(movieId) %>%
  left_join(movie_bias, by = "movieId") %>%
  left_join(user_bias, by = "userId") %>%
  left_join(week_bias, by = "week") %>%
  left_join(time_bias, by = "time") %>%
  left_join(genre_bias, by = "genres") %>%
  ungroup() %>%
  select(b_i, b_u, b_w, b_t, b_g) %>% 
  rowSums()

final_rmse <- rmse(validation$rating, y_hat)

cat("Final rmse: ", final_rmse, "\n")

# Best RMSE obtained
# Clean up everything
rm(list=ls())
invisible(gc())