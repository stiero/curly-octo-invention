#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 11:43:24 2020

@author: tauro
"""

from sklearn.datasets import make_blobs
import seaborn as sns
import pandas as pd
import numpy as np
import scipy.stats as stats
from pprint import PrettyPrinter

pp = PrettyPrinter(depth = 5)


def simulate_data(N, P, S, K):
    
    """
    Simulates a dataset with clusters, including relevant and irrelevant columns.
    
    Args -> 
        N: Number of rows in the data.
        P: Total number of columns in the data.
        S: Number of columns that are relevant to the clustering.
        K: Number of clusters.
           
    Returns -> 
        data: A Pandas DataFrame according to the supplied specifications.
        
        
    Raises ->
        AssertionError: When P < S, When K < 1.  
    
    """
    assert K > 0
    assert P > S
    
    relevant_col_names = ["col_"+str(i) for i in range(S)]
    relevant_col_names.append('cluster')
    
    X, y = make_blobs(n_samples=N, n_features=S, random_state=0, centers=K)
    data = np.column_stack((X, y))
    data = pd.DataFrame(data, columns = relevant_col_names)
    
    num_synth_cols = P-S
    synth_data = pd.DataFrame()
    for i in range(num_synth_cols):
        col = np.random.random(N)
        synth_data["synth_col_"+str(i)] = col

    data = pd.concat((data, synth_data), axis=1)
    
    print("Generated {} clustered dataset with {} rows and {} columns ({} useful)".
          format(K, N, P, S))
    
    return data
    


def rank_columns(data):
    
    """
    Assigns a score to every column of the input dataframe based on its likelihood
    on being a useful feature during clustering.
    
    Args ->
        data (pd.DataFrame): A dataframe ideally generated as an output by simulate_data()
    
    Returns ->
        column_scores (dict()): A dictionary object of format {column name: (F-statistic, pvalue)}
        containing the F-statistic and p-values
        for each column.
    
    Raises ->
        AssertionError: To ensure a non-empty DataFrame.
        
    
    """
    
    assert data.shape[0] != 0
    assert data.shape[1] != 0
    
    columns = data.columns
    cluster_labels = set(data['cluster'])
    column_scores = {}
    
    for col in columns:
        if col != 'cluster':
            subsets = []
            for lab in cluster_labels:
                subsets.append(data[col][data['cluster'] == lab])
            column_scores[col] = stats.f_oneway(*subsets)
    
    pp.pprint(column_scores)
    
    return column_scores




# Testing it out
    
data = simulate_data(N = 1000, P = 10, S = 6, K = 2)
sns.scatterplot(x = data['col_1'], y = data['col_2'], hue=data['cluster']);
ranks = rank_columns(data)

data = simulate_data(N = 1000, P = 10, S = 3, K = 4)
sns.scatterplot(x = data['col_1'], y = data['col_2'], hue=data['cluster']);
ranks = rank_columns(data)
