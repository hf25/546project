#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 21:52:47 2018

@author: haofang
"""
import sqlite3
# Using SQL query statement to find out the top 20 frequent appeared genres in high rated movies(>4)
connection = sqlite3.connect("/Users/haofang/Desktop/546project/546Data.db")
cursor = connection.cursor()
cursor.execute("select count(a.movieId) as numvers_of_movie, a.genres as genres\
                             from (select movies.movieId, movies.genres\
                                        from movies , meanrating \
                                        where movies.movieId=meanrating.movieId and\
                                        meanrating.mean_rating>4 group by movies.movieId, movies.genres) as a\
                             group by a.genres \
                             order by count(a.movieId) desc;")
countMovie_genres= cursor.fetchall()
cursor.close()
connection.close()

for v in range(20):
    print(countMovie_genres[v])