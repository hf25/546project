#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 21:58:48 2018

@author: haofang
"""
import sqlite3
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


connection = sqlite3.connect("/Users/haofang/Desktop/546project/small_dataset.db")
cursor = connection.cursor()
# Using SQL qurery statement to findout how the frequency of each rating score and plot:
cursor.execute("select ratings.userId, ratings.movieId,a.total_watch_times, ratings.rating, \
                           ratings.timestamp,movies.title,movies.genres\
                           from (select movieId, count(movieId) as total_watch_times\
                                      from ratings\
                                     group by movieId) as a, movies, ratings\
                                     where movies.movieId=a.movieId and ratings.movieId=a.movieId\
                                     order by userId;")
data= cursor.fetchall()

cursor.execute("select userId from ratings;")
users=cursor.fetchall()
userId = [x[0] for x in users]

cursor.execute("select movieId from ratings;")
movies=cursor.fetchall()
movieId = [x[0] for x in movies]

cursor.close()
connection.close()


user_item_data=pd.DataFrame(list(data), index=userId,\
                            columns=["userId","movieId","total_watch_times","rating","timestamp","title","genres"])


print(user_item_data['total_watch_times'].describe())
print(user_item_data['rating'].describe())
print(user_item_data['total_watch_times'].quantile(np.arange(0.2, 1, .05)), )
# Found that 0.7 of all of the movies was watched more than 10 times. 

def reshape_dataset(index,columns,values,data):
    reshaped_data=data.pivot(index,columns,values).fillna(0)
    return reshaped_data

def knn_fitting(data):
    knn_fit = NearestNeighbors(metric = 'cosine', algorithm = 'auto')
    fitting=knn_fit.fit(data)
    return fitting


def predict_and_recommandation(who_predict_for,knn_fit,the_reshaped_data,k):
    the_column=the_reshaped_data.iloc[who_predict_for,:]
    the_column=the_column.values.reshape(1,-1)
    cosine_distance, recmovie=knn_fit.kneighbors(the_column,k)
#    print("Recommand for:",who_predict_for)
#    for v in range(1,k):
#             print("recommanded movieId:",recmovie[0,v],"the cosine distance:",cosine_distance[0,v])
    return cosine_distance, recmovie


index = 'movieId'
columns = 'userId'
values = 'rating'

item_item_data=pd.DataFrame(list(data), index=userId,\
                           columns=["userId","movieId","total_watch_times","rating","timestamp","title","genres"])


reshaped_data=reshape_dataset(index,columns,values,item_item_data)
# Split into train set, validation set and test set.
movieId=list(set(movieId))
train_data, test_data, train_Id, test_Id = train_test_split( reshaped_data, movieId, test_size=0.25)
val_data,test_data,val_Id,test_Id=train_test_split(test_data,test_Id,test_size=0.5)
userId=list(set(userId))
knn_fit=knn_fitting(train_data)
cosine_distances,recmovie=predict_and_recommandation(userId[1],knn_fit,train_data,6)

#
def squared_cosine_distance(cosine_distance):
    i=0
    for v in range(1,len(cosine_distance[0])):
        i=i+cosine_distance[0][v]**2
    i=i/(len(cosine_distances[0])-1)
    return i

#knn_fit=knn_fitting(train_data)
#cosine_distances,recmovie=predict_and_recommandation(pick_one_userId,knn_fit,train_data,6)


#val_error=[]
train_error=[]
K=[]
for k in np.arange(2,40,3):
      print("k=",k)
      i=0
      j=0
      for v in userId:
          #using the training fit to predict validation 
            train_cosine_distances,train_recmovie=predict_and_recommandation(v,knn_fit,val_data,k)
           # val_cosine_distances,val_recmovie=predict_and_recommandation(v,knn_fit,train_data,k)
           # i=i+squared_cosine_distance(val_cosine_distances)
            j=j+squared_cosine_distance(train_cosine_distances)
      train_error.append(j)
     # val_error.append(i)
      K.append(k)
      
plt.plot(K, train_error)
#plt.plot(K, val_error)
plt.xlabel('Value of K')
plt.ylabel('Squared cosine distances')
plt.title("Squared cosine distances vs. value of K")
plt.show()
    











