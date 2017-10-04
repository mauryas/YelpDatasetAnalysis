# -*- coding: utf-8 -*-
"""
Created on Mon May  8 20:07:54 2017

@author: ShivamMaurya
"""
'''
Excercise 2: Consider the scenario of Exercise 1, Task 2. We still consider the 
reviews as a stream with the timepoint being the 1st of each month. Let the 
window size be 3 months. Calculate
a. For timepoint ti, the number of reviews in the window
b. For timepoint ti, the number of positive/negative reviews in the window
c. For timepoint ti, the number of users in the window
Use a Sliding window and Landmark window for the above calculations.
'''
import numpy as np
import pandas as pd
from datetime import datetime as dt
from sklearn.linear_model import LogisticRegression
from datetime import timedelta
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
#%matplotlib inline

def positiveWords(text,pos_words):
    posW = 0
    for w in text:
        if pos_words.issuperset({w.lower()}):
            posW+=1
    return posW

def negetiveWords(text,neg_words):
    negW = 0
    for w in text:
        if neg_words.issuperset({w.lower()}):
            negW+=1
    return negW    

def Review(streamVector):
    posReview = 0
    negReview = 0
    for instance in streamVector:
        if(instance[3]>2):
            posReview+=1
        else:
            negReview+=1
    return posReview,negReview

def sumReviewWords(streamVector):
    sumWords = 0
    for instance in streamVector:
        sumWords += instance[6]
        
    return sumWords

def calSumSqWords(streamVector):
    sumSquare = 0
    for instance in streamVector:
        sumSquare += instance[6]**2
    return sumSquare
        
def sumSqPos(streamVector):
    sumSquare = 0
    for instance in streamVector:
        sumSquare+=instance[7]**2
    return sumSquare

def sumSqNeg(streamVector):
    sumSquare = 0
    for instance in streamVector:
        sumSquare += instance[8]**2
    return sumSquare

def newUsers(streamVector,userSet):
    for instance in streamVector:
        userSet.add(instance[1])
    return userSet
    
#Yelp Dataset Analysis
dataReview = pd.read_csv('../data/city/Gilbert/yelp_academic_dataset_review.csv')
dataReview = np.array(dataReview)

#Positive Lexicon
pos_words = set()
for words in open('../data/opinion lexicon/positive-words.txt', 'r').readlines()[35:]:
    pos_words.add(words.rstrip())

#Negative Lexicon
neg_words = set()
for words in open('../data/opinion lexicon/negative-words.txt', 'r').readlines()[35:]:
    neg_words.add(words.rstrip())

reviewVector = []
reviewLine = [0,0,0,0,0,0,0,0]
for line in dataReview:
    text = line[5][3:len(line[5])-2].split(' ')
    date = dt(int(line[2][2:6]),int(line[2][7:9]),int(line[2][10:len(line[2])-1]))
    busId = line[6][2:len(line[6])-1]
    revId = line[4][2:len(line[3])-1]
    userId = line[3][2:len(line[3])-1]
    reviewVector.append(np.array([revId,userId,busId,int(line[8]),line[5][3:len(line[5])-2],date,len(text), positiveWords(text,pos_words),negetiveWords(text,neg_words)]))
    #review,user,business,rating,,date,size review, pos word no, neg word no

#reviewVector = np.array(reviewVector)
reviewVector = np.random.permutation(reviewVector)
#reviewVector = reviewVector[reviewVector[:,5].argsort()]
#print(reviewVector[0,[3,6,7,8]])
#Logistic Regression

model = LogisticRegression().fit(reviewVector[:,6:8], np.int_(reviewVector[:,3]))
print(model.score(reviewVector[:,6:8], np.int_(reviewVector[:,3])))