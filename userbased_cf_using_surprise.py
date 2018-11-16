# CSE546 Project
# Part 2: User-based collaborative filtering
# Hongbin Qu & Hao Fang
# This program implements a user based collaborative filtering 
# algorithm using kNN algorithm with the help of surprise package

import numpy as np
import matplotlib.pyplot as plt
from surprise import Reader
from surprise import Dataset
from surprise.model_selection import cross_validate
from surprise.prediction_algorithms.knns import KNNWithMeans

reader = Reader(line_format = 'user item rating timestamp',
                sep = ',', skip_lines = 1)
data = Dataset.load_from_file('ratings.csv', reader = reader)

# Calculate root mean square error using k nearest neighbor method
# with k starting from 2 to 50 in step sizes of 2
rmse_train = []
rmse_test = []
for k in range(2, 51, 2):
    print('k =', k)
    sim_options = {'name': 'cosine'}
    algo = KNNWithMeans(k = k, sim_options = sim_options, verbose = False)
    result = cross_validate(algo, data, measures = ['RMSE'], cv = 10,
                            return_train_measures = True, verbose = False)
    rmse_train.append(np.mean(result['train_rmse']))
    rmse_test.append(np.mean(result['test_rmse']))

plt.figure(1)
plt.plot(range(2, 51, 2), rmse_train)
plt.plot(range(2, 51, 2), rmse_test)
plt.xlabel('k')
plt.ylabel('Root Mean Square Error')
plt.title('kNN: The Result of Average RMSE versus k')
plt.legend(('train rmse', 'test rmse'))
plt.show()
