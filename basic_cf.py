# CSE546 Project
# Part 2: User-based and item-based collaborative filtering
# Hongbin Qu & Hao Fang
# This program implements a user based and a item based 
# collaborative filtering algorithm

import numpy as np
from math import sqrt
from dataset_analysis import df_r
from dataset_analysis import unique_user_num
from dataset_analysis import unique_movie_num
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import pairwise_distances

# Splits the dataset as training data and testing data
train_df, test_df = train_test_split(df_r, test_size = 0.1)

# Computes the R matrix on training data
R_train = train_df.pivot(index = 'userId', columns = 'movieId',
                           values = 'rating').fillna(0).values
appendix_row = np.zeros((unique_user_num - R_train.shape[0], R_train.shape[1]))
R_train = np.concatenate((R_train, appendix_row), axis = 0)
appendix_col = np.zeros((R_train.shape[0], unique_movie_num - R_train.shape[1]))
R_train = np.concatenate((R_train, appendix_col), axis = 1)

# Computes the R matrix on testing data
R_test = test_df.pivot(index = 'userId', columns = 'movieId',
                           values = 'rating').fillna(0).values
appendix_row = np.zeros((unique_user_num - R_test.shape[0], R_test.shape[1]))
R_test = np.concatenate((R_test, appendix_row), axis = 0)
appendix_col = np.zeros((R_test.shape[0], unique_movie_num - R_test.shape[1]))
R_test = np.concatenate((R_test, appendix_col), axis = 1)

########### user-based collaborative filtering ################################

# Computes user-based similarity matrix
S_user = pairwise_distances(R_train, metric = 'euclidean')

# Training set prediction
#mean_user_rating = R_train.mean(axis = 1)
mean_user_rating = np.zeros(R_train.shape[0])
for i in range(R_train.shape[0]):
    mean_user_rating[i] = R_train[i].mean()
mean_user_rating_train = mean_user_rating[:, np.newaxis]
difference_train = R_train - mean_user_rating_train
numerator_train_ub = np.dot(S_user, difference_train)
denominator_train_ub = np.abs(S_user).sum(axis = 1)[:, np.newaxis]
prediction_train_ub = mean_user_rating[:, np.newaxis] + numerator_train_ub\
                        / denominator_train_ub

# Computes root mean squate error of training set
prediction_specified_train_ub = prediction_train_ub[R_train.nonzero()]
R_specified_train_ub = R_train[R_train.nonzero()]
error_train_ub = sqrt(mean_squared_error(prediction_specified_train_ub,
                                         R_specified_train_ub))
print('User_based training set prediction rmse:', error_train_ub)

# Testing Set Prediction
difference_test = R_test - mean_user_rating[:, np.newaxis]
numerator_test_ub = np.dot(S_user, difference_test)
denominator_test_ub = np.abs(S_user).sum(axis = 1)[:, np.newaxis]
prediction_test_ub = mean_user_rating[:, np.newaxis] + numerator_test_ub\
                        / denominator_test_ub

# Computes root mean squate error of training set
prediction_specified_test_ub = prediction_test_ub[R_test.nonzero()]
R_specified_test_ub = R_test[R_test.nonzero()]
error_test_ub = sqrt(mean_squared_error(prediction_specified_test_ub,
                                        R_specified_test_ub))
print('User-based testing set prediction rmse:', error_test_ub)


########### movie-based collaborative filtering ###############################

# Computes movie-based similarity matrix
S_movie = pairwise_distances(R_train.T, metric = 'euclidean')

# Training set prediction
numerator_train_mb = np.dot(R_train, S_movie)
denominator_train_mb = np.abs(S_movie).sum(axis = 1)
prediction_train_mb = numerator_train_mb / denominator_train_mb

# Computes root mean squate error of training set
prediction_specified_train_mb = prediction_train_mb[R_train.nonzero()]
R_specified_train_mb = R_train[R_train.nonzero()]
error_train_mb = sqrt(mean_squared_error(prediction_specified_train_mb,
                                         R_specified_train_mb))
print('Movie-based training set prediction rmse:', error_train_mb)

# Testing Set Prediction
numerator_test_mb = np.dot(R_test, S_movie)
denominator_test_mb = np.abs(S_movie).sum(axis = 1)
prediction_test_mb = numerator_test_mb / denominator_test_mb

# Computes root mean squate error of training set
prediction_specified_test_mb = prediction_test_mb[R_test.nonzero()]
R_specified_test_mb = R_test[R_test.nonzero()]
error_test_mb = sqrt(mean_squared_error(prediction_specified_test_mb,
                                        R_specified_test_mb))
print('Movie-based testing set prediction rmse:', error_test_mb)
