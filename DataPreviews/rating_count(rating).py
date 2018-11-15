#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 13:52:05 2018

@author: haofang
"""

import matplotlib.pyplot as plt
import sqlite3
import numpy as np


connection = sqlite3.connect("/Users/haofang/Desktop/546project/small_dataset.db")
cursor = connection.cursor()
# Using SQL qurery statement to findout how the frequency of each rating score and plot:
cursor.execute("select rating, count(rating) from ratings group by rating;")
count_countRating= cursor.fetchall()
rating= [x[0] for x in count_countRating]
countRating= [x[1] for x in count_countRating]
y_pos = np.arange(len(countRating))
plt.bar(y_pos, countRating)
plt.xticks(y_pos, rating)
plt.xlabel('Rating scores')
plt.ylabel('Frequency of rating scores')
plt.title('Rating scores frequency')
plt.grid(True)
plt.show()

cursor.close()
connection.close()