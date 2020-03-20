#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 15:35:56 2020

@author: tauro
"""


# from sklearn.datasets import load_boston
# import numpy as np

# boston = load_boston()

# X = pd.DataFrame(boston.data, columns = boston.feature_names)
# y = pd.DataFrame(boston.target, columns = ["target"])

import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction import FeatureHasher


df, y = make_regression(n_samples=1000, n_features=4, n_informative=4, random_state = 0)

df = pd.DataFrame(df)

high_card_col = pd.DataFrame(np.random.choice(range(4000,4500), 1000), columns = ['cat1'])
low_card_col = pd.DataFrame(np.random.choice(range(1,10), 1000), columns = ['cat2'])

df = pd.concat([df, high_card_col, low_card_col], axis=1)

df['cat1'] = df['cat1'].astype('str')
df['cat2'] = df['cat2'].astype('str')

len(df['cat1'].unique())
len(df['cat2'].unique())




def column_encoder(y, X, i):
    
    """
    Returns the log-odds of the count values for a categorical column in the input dataframe.
    
    Args ->
        y (pd.Series or pd.DataFrame): The target variable
        X (pd.Series or pd.DataFrame): The input features
        i (int): The index of the column to be transformed
    
    Returns ->
        encoded (pd.Series): A pd.Series object which is an encoded form of the
        original column.
    
    Raises ->
        AssertionError: On various conditions
    """
    
    assert X.shape[0] > 0
    assert y.shape[0] > 0
    assert isinstance(i, int)
    
    col = X.iloc[:,i]
    
    col_name = X.columns[i]
    
    if col.dtype not in ['str', 'object', 'O']:
        return
    
    counts = col.value_counts()
    
    cardinality = len(col.unique())
    
    if cardinality > 10:
    
        prop = counts/X.shape[0]
        not_prop = (X.shape[0] - counts)/X.shape[0]
        log_odds_ratio = np.log(prop) - np.log(not_prop)                  
        encoded = col.map(log_odds_ratio.to_dict())
        
        return encoded
    
    else:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse=True)
        encoded = ohe.fit_transform(np.array(col).reshape(-1,1))
        encoded = pd.DataFrame(encoded.toarray())
        return encoded
        
    


encoded_col = column_encoder(y, df, 4)
set(encoded_col)

encoded_col = column_encoder(y, df, 5)
set(encoded_col)




h = FeatureHasher(n_features = 10, input_type='string')

hashed = h.transform(df['cat'])



def column_encoder(y, X, i):
    
    """
    Returns the log-odds of the count values for a categorical column in the input dataframe.
    
    Args ->
        y (pd.Series or pd.DataFrame): The target variable
        X (pd.Series or pd.DataFrame): The input features
        i (int): The index of the column to be transformed
    
    Returns ->
        encoded (pd.Series): A pd.Series object which is an encoded form of the
        original column.
    
    Raises ->
        AssertionError: On various conditions
    """
    
    assert X.shape[0] > 0
    assert y.shape[0] > 0
    assert isinstance(i, int)
    
    col = X.iloc[:,i]
    
    col_name = X.columns[i]
    
    if col.dtype not in ['str', 'object', 'O']:
        return
    
    counts = col.value_counts()
    
    cardinality = len(col.unique())
    
    if cardinality > 10:
        
        h = FeatureHasher(n_features = 10, input_type='string')
        encoded = h.transform(col).toarray()
        encoded = pd.DataFrame(encoded)
        
        return encoded
    
    else:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse=True)
        encoded = ohe.fit_transform(np.array(col).reshape(-1,1))
        encoded = pd.DataFrame(encoded.toarray())
        return encoded



encoded_col = column_encoder(y, df, 4)
set(encoded_col)
