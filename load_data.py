#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 21:10:41 2015

@author: ddboline
"""

import numpy as np
import pandas as pd
import datetime
from dateutil.parser import parse

STATUS_GROUP = ['non functional', 'functional needs repair', 'functional']

def transform_from_classes(inp):
    y = np.zeros((inp.shape[0], 3), dtype=np.int64)
    for index, cidx in enumerate(inp):
        y[index, cidx] = 1.0
    return y

def transform_to_class(vec):
    return np.argmax(vec, axis=1)

def extract_categories(df):
    category_dict = {}
    
    for category in ('basin', 'region', 'scheme_management', 
                     'extraction_type', 'extraction_type_group', 
                     'extraction_type_class', 'management', 
                     'management_group', 'payment', 'payment_type', 
                     'water_quality', 'quality_group', 'quantity', 
                     'quantity_group', 'source', 'source_type', 
                     'source_class', 'waterpoint_type', 
                     'waterpoint_type_group', 'lga'):
        category_dict[category] = sorted(df[category].unique())
    
    return category_dict

def clean_data(df, cat_dict):
    if 'status_group' in df.columns:
        df['status_group'] = df['status_group'].map({c: n for (n, c) in
                                                     enumerate(STATUS_GROUP)})
    
    df['date_recorded'] = df['date_recorded']\
                          .apply(lambda x: (parse(x).date()
                                            - datetime.date(year=2000, 
                                                            month=1, 
                                                            day=1)).days)
    
    for cat in cat_dict:
        if cat not in df.columns:
            continue
        df[cat] = df[cat].map({c: n for (n, c) in enumerate(cat_dict[cat])})
    
    for label in 'public_meeting', 'permit':
        df.loc[df[label].isnull(), label] = -1
        df[label] = df[label].astype(np.int64)
        
    
    drop_labels = ['funder', 'installer', 'wpt_name', 'subvillage', 'ward', 
                   'recorded_by', 'scheme_name']
    df = df.drop(labels=drop_labels, axis=1)
    
    for col in df.columns:
        if np.any(df[col].isnull()):
            print col, np.sum(df[col].isnull())
    
    return df

def load_data(do_plots=False):
    train_df = pd.read_csv('train_values.csv.gz', compression='gzip')
    train_df_labels = pd.read_csv('train_labels.csv.gz', compression='gzip')
    test_df = pd.read_csv('test_values.csv.gz', compression='gzip')
    submit_format = pd.read_csv('submit_format.csv.gz', compression='gzip')
    
    train_df['status_group'] = train_df_labels['status_group']
    
    cat_dict = extract_categories(train_df)
    train_df = clean_data(train_df, cat_dict)
    test_df = clean_data(test_df, cat_dict)
    
    print train_df.columns
    
    if do_plots:
        from plot_data import plot_data
        plot_data(train_df, prefix='html_train')
        plot_data(test_df, prefix='html_test')

    xtrain = train_df.drop(labels=['id', 'status_group'], axis=1).values
    ytrain = transform_from_classes(train_df['status_group'].values)
    xtest = test_df.drop(labels=['id'], axis=1).values
    ytest = submit_format

    return xtrain, ytrain, xtest, ytest

if __name__ == '__main__':
    xtrain, ytrain, xtest, ytest = load_data(do_plots=True)
    
    print [x.shape for x in xtrain, ytrain, xtest, ytest]
