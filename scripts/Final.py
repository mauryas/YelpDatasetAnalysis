#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import division
"""
Created on Tue Jun 27 22:08:33 2017

@author: mukul
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics, cross_validation,svm
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
import string
from string import digits
from nltk.corpus import stopwords
from nltk import word_tokenize
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics, cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt
from sklearn.learning_curve import learning_curve



tokenize = lambda doc: doc.lower().split(' ')

#sklearn_tfidf = TfidfVectorizer(norm='l2',min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True)

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


def revRatVal(rating):
    if(rating<=2):
        return 0
    elif rating>=2 and rating<=3.5:
        return 1
    else:
        return 2

dataReview = pd.read_csv('../data/city/Gilbert/yelp_academic_dataset_review.csv')
dataReview = np.array(dataReview)
dataReview = np.random.permutation(dataReview)


BusReview = pd.read_csv('../data/city/Gilbert/yelp_academic_dataset_business.csv')
BusReview = np.array(BusReview)

UsrReview = pd.read_csv('../data/city/Gilbert/yelp_academic_dataset_user.csv')
UsrReview = np.array(UsrReview)


#Positive Lexicon
pos_words = set()
for words in open('../data/opinion-lexicon-English/positive-words.txt', 'r').readlines()[35:]:
    pos_words.add(words.rstrip())

#Negative Lexicon
neg_words = set()
for words in open('../data/opinion-lexicon-English/negative-word.txt', 'r').readlines()[35:]:
    neg_words.add(words.rstrip())
        

totalText=[]
for line in dataReview:
    text = line[9][3:len(line[9])-3]
    totalText.append(text)
    
totalTextS=[]    
for line in totalText:
   rex = line.translate(line.maketrans('','',digits))
   rex = rex.translate(rex.maketrans('\n',' '))
   rex = rex.translate(rex.maketrans('','', string.punctuation))
   totalTextS.append(rex)

totalTextS = [word for word in totalTextS if word not in stopwords.words('english')]


totalWords=[]
for word in totalTextS:
    text=word.split(' ')
    totalWords.append(text)
    
sklearn_tfidf = TfidfVectorizer(norm='l2',min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True,tokenizer=tokenize)
sklearn_representation = sklearn_tfidf.fit_transform(totalTextS)


RevVec = []
userInfo = []
for line,tfidf in zip(dataReview,sklearn_representation):
    text = line[9][3:len(line[9])-3].split(' ')
    date = line[10][2:len(line[9])-1].split(' ')
    busId = line[5][2:len(line[5])-1]
    revId = line[6][2:len(line[6])-1]
    userInfo = UsrReview[np.where(UsrReview[:,-10]==line[1]),[-12,-2]]
    busInfo = BusReview[np.where(BusReview[:,-4]==line[5]),[4,-2]]
    RevVec.append(np.array([int(userInfo[0,0]),int(userInfo[0,1]),revRatVal(float(line[8])),int(busInfo[0,0]),int(busInfo[0,1]),len(text),positiveWords(text,pos_words),negetiveWords(text,neg_words)],tfidf))
    
RevVec = np.array(RevVec)

X=RevVec[:,[0,1,3,4,5,6,7]]

Y=np.int_(np.array(RevVec[:,2]))    

# Logistic Regression
model = LogisticRegression()
predicted = cross_validation.cross_val_predict(model, X,Y, cv=5)
print ("\nMetrics for Logistic Regression\n")
print (metrics.accuracy_score(Y, predicted))
print (metrics.classification_report(Y, predicted))

#==============================================================================
# train_sizes, train_scores, test_scores = learning_curve(
#         model, X, Y, cv=5, n_jobs=-1, train_sizes=np.linspace(.1, 1., 10), verbose=0)
# 
# train_scores_mean = np.mean(train_scores, axis=1)
# train_scores_std = np.std(train_scores, axis=1)
# test_scores_mean = np.mean(test_scores, axis=1)
# test_scores_std = np.std(test_scores, axis=1)
# plt.figure()
# plt.title("Logistic Regression")
# plt.legend(loc="best")
# plt.xlabel("Training examples")
# plt.ylabel("Score")
# plt.ylim((0.6, 1.01))
# plt.gca().invert_yaxis()
# plt.grid()
# 
# # Plot the average training and test score lines at each training set size
# plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label="Training score")   
# plt.plot(train_sizes, test_scores_mean, 'o-', color="r", label="Test score")
# 
# # Plot the std deviation as a transparent range at each training set size
# plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, 
#                  alpha=0.1, color="b")
# plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, 
#                  alpha=0.1, color="r")
# plt.draw()
# plt.show()
# plt.gca().invert_yaxis()
#==============================================================================
   

# KNN Classifier
neigh = KNeighborsClassifier(n_neighbors=25)
print ("\nMetrics for KNN Classifier\n")
predicted = cross_validation.cross_val_predict(neigh, X,Y, cv=5)
print (metrics.accuracy_score(Y, predicted))
print (metrics.classification_report(Y, predicted))


#==============================================================================
# train_sizes, train_scores, test_scores = learning_curve(
#         neigh, X, Y, cv=5, n_jobs=-1, train_sizes=np.linspace(.1, 1., 10), verbose=0)
# 
# train_scores_mean = np.mean(train_scores, axis=1)
# train_scores_std = np.std(train_scores, axis=1)
# test_scores_mean = np.mean(test_scores, axis=1)
# test_scores_std = np.std(test_scores, axis=1)
# plt.figure()
# plt.title("KNN")
# plt.legend(loc="best")
# plt.xlabel("Training examples")
# plt.ylabel("Score")
# plt.ylim((0.6, 1.01))
# plt.gca().invert_yaxis()
# plt.grid()
# 
# # Plot the average training and test score lines at each training set size
# plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label="Training score")   
# plt.plot(train_sizes, test_scores_mean, 'o-', color="r", label="Test score")
# 
# # Plot the std deviation as a transparent range at each training set size
# plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, 
#                  alpha=0.1, color="b")
# plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, 
#                  alpha=0.1, color="r")
# plt.draw()
# plt.show()
# plt.gca().invert_yaxis()
#==============================================================================

#Random Forest
clf = RandomForestClassifier(n_estimators=250)
print ("\nMetrics for Random Forest Classifier\n")
predicted = cross_validation.cross_val_predict(clf, X,Y, cv=5)
print (metrics.accuracy_score(Y, predicted))
print (metrics.classification_report(Y, predicted))

#==============================================================================
# 
# train_sizes, train_scores, test_scores = learning_curve(
#         clf, X, Y, cv=5, n_jobs=-1, train_sizes=np.linspace(.1, 1., 10), verbose=0)
# 
# train_scores_mean = np.mean(train_scores, axis=1)
# train_scores_std = np.std(train_scores, axis=1)
# test_scores_mean = np.mean(test_scores, axis=1)
# test_scores_std = np.std(test_scores, axis=1)
# plt.figure()
# plt.title("Random Forest")
# plt.legend(loc="best")
# plt.xlabel("Training examples")
# plt.ylabel("Score")
# plt.ylim((0.6, 1.01))
# plt.gca().invert_yaxis()
# plt.grid()
# 
# # Plot the average training and test score lines at each training set size
# plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label="Training score")   
# plt.plot(train_sizes, test_scores_mean, 'o-', color="r", label="Test score")
# 
# # Plot the std deviation as a transparent range at each training set size
# plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, 
#                  alpha=0.1, color="b")
# plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, 
#                  alpha=0.1, color="r")
# plt.draw()
# plt.show()
# plt.gca().invert_yaxis()
# 
# 
#==============================================================================

#print(model.score(X_train,y_train))
#print(model.score(X_test,y_test))
#UsrReview[np.where(UsrReview[:,-10]==line[1])]
#UsrReview[np.where(UsrReview[:,-10]==line[1]),[-12,-2]]







