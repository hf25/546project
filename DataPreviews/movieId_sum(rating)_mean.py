#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 21:41:21 2018

@author: haofang
"""
import matplotlib.pyplot as plt
import sqlite3
import numpy as np

# Using SQL qurery statement to findout how the how many ratings that each movie got and their mean rating

connection = sqlite3.connect("/Users/haofang/Desktop/546project/546Data.db")
cursor = connection.cursor()
cursor.execute("select movieId, sum(rating)/count(rating) as mean_rating\
                             from ratings\
                             group by movieId\
                             order by mean_rating;")
movieId_countRate_meanRate= cursor.fetchall()
cursor.close()
connection.close()
movieId= [x[0] for x in movieId_countRate_meanRate]
mean_rate= [x[1] for x in movieId_countRate_meanRate]
y_pos = np.arange(len(mean_rate))
plt.figure(1)
plt.hist(mean_rate, bins=50, normed=True, alpha=0.5,
         histtype='stepfilled', color='steelblue',
         edgecolor='none')
plt.xlabel('Movie ID')
plt.ylabel('Mean Rating')
plt.title('The distribution of mean ratings')
plt.show()
