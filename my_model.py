#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 15:13:43 2015

@author: ddboline
"""

import gzip
import cPickle as pickle
import numpy as np
#import pandas as pd

from load_data import load_data, transform_to_class, STATUS_GROUP

from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

NCLASSES = 3

def train_model_parallel(model, xtrain, ytrain, index=0, prefix=''):
    xTrain, xTest, yTrain, yTest = train_test_split(xtrain, ytrain[:, index],
                                                    test_size=0.25)
    model.fit(xTrain, yTrain)
    print 'score', model.score(xTest, yTest)
    with gzip.open('model_%s_%d.pkl.gz' % (prefix, index), 'wb') as pklfile:
        pickle.dump(model, pklfile, protocol=2)

def test_model_parallel(xtrain, ytrain, prefix=''):
    xTrain, xTest, yTrain, yTest = train_test_split(xtrain, ytrain, 
                                                    test_size=0.25)
    yprob = np.zeros(yTest.shape)
    for index in range(NCLASSES):
        with gzip.open('model_%s_%d.pkl.gz' % (prefix, index), 'rb') as pklfile:
            model = pickle.load(pklfile)
            ypred = model.predict(xTest)
            yprob[:, index] = model.predict_proba(xTest)[:, 1]
            print model.score(xTest, yTest[:, index])
            print accuracy_score(yTest[:, index], ypred)
    
    ytest_final = transform_to_class(yTest)
    ypred_final = transform_to_class(yprob)

    print ytest_final.shape, ypred_final.shape
    print accuracy_score(ytest_final, ypred_final)

def prepare_submission_parallel(xtest, ytest, prefix=''):
    ytest_prob = np.zeros((ytest.shape[0], NCLASSES))
    for n in range(NCLASSES):
        with gzip.open('model_%s_%d.pkl.gz' % (prefix, n), 'rb') as mfile:
            model = pickle.load(mfile)
            ytest_prob[:, n] = model.predict_proba(xtest)[:, 1]
    ytest2 = transform_to_class(ytest_prob).astype(np.int64)

    ytest['status_group'] = ytest2
    ytest['status_group'] = ytest['status_group'].map({n: c for (n,c) 
                                                in enumerate(STATUS_GROUP)})

    ytest.to_csv('submission.csv', index=False)

    return


if __name__ == '__main__':
    xtrain, ytrain, xtest, ytest = load_data()

    model = RandomForestClassifier(n_estimators=10, n_jobs=-1)
#    for idx in range(3):
#        train_model_parallel(model, xtrain, ytrain, index=idx, prefix='rf10')
#    test_model_parallel(xtrain, ytrain, prefix='rf10')
    prepare_submission_parallel(xtest, ytest, prefix='rf10')
