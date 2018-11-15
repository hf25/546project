#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 22:06:33 2018

@author: haofang
"""

import sqlite3
# Using SQL qurery statement to findout the top 20 frequent tags(higher than 0.9 relevance) of  top rated movies(4.5,5)
connection = sqlite3.connect("/Users/haofang/Desktop/546project/546Data.db")
cursor = connection.cursor()
cursor.execute("select count(a.movieId), b.tag, min(b.relevance)\
                             from (select movies.movieId as movieId, movies.genres as genres\
                                       from movies , meanrating\
                                       where movies.movieId=meanrating.movieId and \
                                       meanrating.mean_rating>4 group by movies.movieId, movies.genres) as a, \
		                              (select genomescores.movieId as movieId, genometags.tag as tag, \
                                      genomescores.relevance as relevance\
                                     from genomescores, genometags\
                                      where genomescores.tagId=genometags.tagId and genomescores.relevance>0.9) as b\
                            where a. movieId=b.movieId\
                            group by b.tag\
                            order by count(a.movieId) desc;")
countMovie_tag= cursor.fetchall()
for v in range(20):
    print(countMovie_tag[v])
cursor.close()
connection.close()