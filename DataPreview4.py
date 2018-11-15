#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 22:01:18 2018

@author: haofang
"""

import sqlite3
# Using SQL qurery statement to findout the top 30 frequently appeared genres
connection = sqlite3.connect("/Users/haofang/Desktop/546project/546Data.db")
cursor = connection.cursor()
cursor.execute("select genres, count(movieId)\
                             from movies\
                             group by genres\
                             order by count(movieId) desc;")
genres_countMovie= cursor.fetchall()
cursor.close()
connection.close()

for v in range(30):
    print(genres_countMovie[v])



